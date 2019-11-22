import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import os
import multiprocessing
import math
import tqdm
import time
import numpy as np

# Loading with torchaudio takes around 43 seconds,
# loading from npys is significantly faster (fastest run was 18 seconds)

DATA_DIR = "soundcloud/"
OUTPUT_DIR = "outputs/"
SONG_NPY_DIR = "npys/"

if len(os.listdir(SONG_NPY_DIR)) == 0:
    # There aren't any npys, so we have to load the data
    # with torchaudio.load and create the npys for future loads
    print('Loading data with torchaudio...')

    # Search the data directory for audio files
    audio_file_names = []
    for idx, f in enumerate(os.listdir(DATA_DIR)):
        # Only load every other mp3 file
        if idx % 2 == 0:
            audio_file_names.append(os.path.join(DATA_DIR, f))

    # Load the audio in a multithreaded manner.
    def load_audio(filename):
        return torchaudio.load(filename)

    data = multiprocessing.Pool(os.cpu_count()).map(load_audio, audio_file_names)

    # Save the ndarrays to npys
    # Need to save array itself and sample
    # rate in a 'numpy tuple'
    print('Saving npys...')
    for idx, d in enumerate(tqdm.tqdm(data)):
        np_tuple = np.array([d[0].numpy(), d[1]])
        np.save(os.path.join(SONG_NPY_DIR, 'song{}'.format(idx+1)), np_tuple)

else:
    # We have npys we can read from, so lets do that
    print('Reading data from npy files...')

    data = []
    for npy in tqdm.tqdm(os.listdir(SONG_NPY_DIR)):
        loaded = np.load(os.path.join(SONG_NPY_DIR, npy), allow_pickle=True)
        data.append((torch.from_numpy(loaded[0]), loaded[1]))


# TODO: Start creating window pairs out of the data: (prev, next)
# TODO: Look at any useful transforms of the audio before processing
# TODO: Make sure all audio has the same sample rate
# TODO: Explore a feasible window size

# Can make this multithreaded, which we will want to do since each worker will
# be blocked by the torchaudio.load() call. See:
# https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset

# An interator for a single audio.
class AudioIterator:
  def __init__(self, audio, segment_size):
    self.audio = audio
    self.segment_size = segment_size
    self.idx = 0
  
  def __iter__(self):
    return self
  
  def __next__(self):
    # Currently, returns previous and next segment, and iterates over the audio just
    # like a conv layer where kernel_size == stride. E.g.:
    # first_iter:
    #     [ s_0 | s_1 | s_2 | s_3 | s_4 | ... ]
    #      --p-- --n--
    # second_iter:
    #     [ s_0 | s_1 | s_2 | s_3 | s_4 | ... ]
    #            --p-- --n--
    # if we want to have much more examples, we might want to explore shifting
    # the p and n window by a "stride".
    if (self.idx + 2) * self.segment_size <= self.audio.shape[0]:
      p, n = self.audio[self.idx*self.segment_size:(self.idx+1)*self.segment_size], self.audio[(self.idx+1)*self.segment_size:(self.idx+2)*self.segment_size]
      self.idx += 1
      return p, n
    else:
      raise StopIteration

# The iterator each worker will get. Iterates through multiple audios.
class MultiAudioIterator:
  def __init__(self, audio_list, segment_size):
    self.audio_list = audio_list
    self.segment_size = segment_size
    self.audio_iters = [iter(AudioIterator(audio, segment_size)) for audio in audio_list]
    self.idx = 0
  
  def __iter__(self):
    return self

  def __next__(self):
    while True and len(self.audio_iters) != 0:
      self.idx %= len(self.audio_iters)
      try:
        ret = next(self.audio_iters[self.idx])
        self.idx += 1
        return ret
      except StopIteration:
        del self.audio_iters[self.idx]
    raise StopIteration

# The overaching dataset. We only want to deal with mono.
# TODO(ivamon): Pass in list of filenames here; in a multiprocessing manner,
# read in the data, process into necessary format, then delete the orignal data.
# This will save space.
class AudioDataset(torch.utils.data.IterableDataset):
  def __init__(self, data, segment_size, sample_rate):
    # files: a list of file names
    # segment_size: Size of each audio segment, in waveframe length
    # sample_rate
    super(AudioDataset).__init__()
    self.segment_size = segment_size
    self.sample_rate = sample_rate
    self.audio = []
    self.length = 0
    # This is the for loop that we want to multiprocess, and add in file loading.
    for d in data:
      if d[1] != sample_rate:
        self.audio.append(torchaudio.transforms.Resample(d[1], sample_rate)(d[0])[0])
      else:
        self.audio.append(d[0][0])
      self.length += self.audio[-1].shape[0] // segment_size - 1

  # Audio file num == idx % len(self.files)
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      return iter(MultiAudioIterator(self.audio, self.segment_size))
    else:
      per_worker = int(math.ceil(len(self.audio) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = per_worker*worker_id
      iter_end = min(iter_start + per_worker, len(self.audio))
      return iter(MultiAudioIterator(self.audio[iter_start:iter_end], self.segment_size))




# The input of the generator will be the previous audio segment, and noise. The output
# will be the next audio segment to coninue the previous
class Generator(nn.Module):
  def __init__(self, noise_dim, segment_size):
    super(Generator, self).__init__()
    # TODO: Specify network layers here

    self.noise_dim = noise_dim
    self.segment_size = segment_size
    
    # input shape will be (n x d)
    # output shape will be (n x d/8)
    # Each conv layer has a single input and output channel
    # The kernel will obviously be 3x1 as per 1d convolutions
    # Stride is 2 with one layer of padding to downsample by a factor
    # of 1/2 in each layer
    self.conv1 = nn.Conv1d(1, 1, 3, stride=2, padding=1)
    self.conv2 = nn.Conv1d(1, 1, 3, stride=2, padding=1)
    self.conv3 = nn.Conv1d(1, 1, 3, stride=2, padding=1)

    self.fc1 = nn.Linear((segment_size // 8) + noise_dim + 1, segment_size)

  def forward(self, prev, next_):
    # print(self.fc1)
    # print(prev.size())

    d1, d2 = prev.size()
    prev = prev.view(d1, 1, d2)

    # print(prev.size())

    prev = self.conv3(self.conv2(self.conv1(prev)))
    prev = prev.view(d1, -1)

    # print(prev.size())

    x = torch.cat((prev, next_), dim=-1)
    # print(x.size())
    x = self.fc1(x)
    return x




# The input of the discriminator will be the previous and next audio segments. The
# output a classification of the input signal concatenation being true audio 
class Discriminator(nn.Module):
  def __init__(self, segment_size):
    super(Discriminator, self).__init__()
    self.segment_size = segment_size
    # TODO: Specify network layers here
    # Note: We should probably use the nn.Sequential
    # paradigm since it makes coding up the forward pass
    # much easier
    self.model = nn.Sequential(
        nn.Linear(segment_size*2, 1),

        # Using sigmoid for binary classification
        # of whether or not an audio segment is "real"
        nn.Sigmoid(),
    )

  def forward(self, prev, next_):
    x = torch.cat((prev, next_), dim=-1)
    x = self.model(x)
    return x

SAMPLE_RATE = 44100 // 4  # Samples per second
SEGMENT_SIZE = int(SAMPLE_RATE)    
LEARNING_RATE = 0.0002
BATCH_SIZE = 4
NOISE_DIM = int(SAMPLE_RATE/4)
EPOCHS = 100



dataset = AudioDataset(data, SEGMENT_SIZE, SAMPLE_RATE)

train_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=BATCH_SIZE,
                                          num_workers=os.cpu_count())



# TODO: Write training code
# TODO: Write code to save the model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

generator = Generator(NOISE_DIM, SEGMENT_SIZE).to(device)
discriminator = Discriminator(SEGMENT_SIZE).to(device)

generator_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Loss function
adversarial_loss = torch.nn.BCELoss()

g_losses = []
d_losses = []


for epoch in range(EPOCHS):
  for batch_idx, (prev_, next_) in enumerate(tqdm.tqdm(train_loader, total=math.ceil(dataset.length/BATCH_SIZE))):
    prev_, next_ = prev_.to(device), next_.to(device)

    # print(batch_idx)

    # Ground truths
    valid = torch.ones(prev_.shape[0], 1).to(device)
    fake = torch.zeros(prev_.shape[0], 1).to(device)

    # ========= Train generator =========

    # Zero the gradient buffers
    generator_opt.zero_grad()

    # Sample noise as generator input
    noise = torch.randn(prev_.shape[0], NOISE_DIM).to(device)

    # Generate the next audio segment
    gen_next = generator(prev_, noise)
  
    # Compute the loss 
    generator_loss = adversarial_loss(discriminator(prev_, gen_next), valid)

    # Compute the gradient of the loss and update weights
    generator_loss.backward()
    generator_opt.step()

    # ========= Train discriminator =========

    # Zero the gradient buffers
    discriminator_opt.zero_grad()

    # Compute the loss
    real_loss = adversarial_loss(discriminator(prev_, next_), valid)
    fake_loss = adversarial_loss(discriminator(prev_, gen_next.detach()), fake)
    discriminator_loss = (real_loss + fake_loss) / 2

    # Compute the gradient of the loss and update weights
    discriminator_loss.backward()
    discriminator_opt.step()

    # Save loss data
    d_losses.append(discriminator_loss.item())
    g_losses.append(generator_loss.item())

  # Print statistics
  print(f"\n[Epoch: {epoch+1}] [Discriminator Loss: {discriminator_loss.item()}] [Generator Loss: {generator_loss.item()}]\n")

end = time.time()
duration = round(end - start, 2)
print(f'Finished training with {EPOCHS} epochs in {duration}s')

# Graph losses
plt.plot(np.arange(len(d_losses)), d_losses, label='Discriminator Loss')
plt.plot(np.arange(len(g_losses)), g_losses, label='Generator Loss')
plt.title('Train Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Train loss')
plt.legend()
plt.show()


SEGMENTS = 100

segments = [torch.zeros(1, SEGMENT_SIZE).to(device)]
for t in range(SEGMENTS):
    noise = torch.randn(1, NOISE_DIM).to(device)
    segments.append(generator(segments[-1], noise))

result = torch.cat(segments[1:], axis=1).to("cpu")
print(result.shape)
print(result)

min_ = torch.min(result)
max_ = torch.max(result)

result = (result - min_) / (max_ - min_)
result = 2*result - 1

print(result)

torchaudio.save(os.path.join(OUTPUT_DIR, "test.mp3"), torch.cat((result, result), axis=0), SAMPLE_RATE)




import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchaudio
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
# We have npys we can read from, so lets do that
print('Reading data from npy files...')

data = []
for npy in tqdm.tqdm(os.listdir(SONG_NPY_DIR)):
    loaded = np.load(os.path.join(SONG_NPY_DIR, npy), allow_pickle=True)[()]
    key = str(44100)
    data.append((torch.from_numpy(loaded[key][0]), loaded[key][1]))

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
  def __init__(self, sigma, seconds, device):
    super(Generator, self).__init__()
    # TODO: Specify network layers here
    self.sigma = sigma
    self.seconds = seconds
    
    self.c_int_sum= 8  # channels intermediate summary
    self.c_sum = 16     # channels summarized

    self.input_sample_rate = 44100
    self.sum_sample_rate = 100

    self.conv1 = nn.Conv1d(1, self.c_int_sum, kernel_size=21, stride=21).to(device)
    self.conv2 = nn.Conv1d(self.c_int_sum, self.c_sum, kernel_size=21, stride=21).to(device)

    self.fc1 = nn.Linear(self.sum_sample_rate * self.seconds, self.sum_sample_rate * self.seconds).to(device)
    self.fc2 = nn.Linear(self.sum_sample_rate * self.seconds, self.sum_sample_rate * self.seconds).to(device)

    self.deconv1 = nn.ConvTranspose1d(self.c_sum, self.c_int_sum, kernel_size=21, stride=21).to(device)
    self.deconv2 = nn.ConvTranspose1d(self.c_int_sum, 4, kernel_size=7, stride=7).to(device)
    self.deconv3 = nn.ConvTranspose1d(4, 1, kernel_size=3, stride=3).to(device)

  def forward(self, prev):
    # Shrink
    x = torch.tanh(self.conv2(torch.tanh(self.conv1(prev.view(prev.shape[0], 1, -1)))))
    # Shape: (batch_size, self.c_sum, sum_sample_rate)
    conv_shape = x.shape

    # Manipulate
    x = torch.tanh(self.fc1(x.view(x.shape[0] * x.shape[1], -1)))
    x = x + torch.randn_like(x) * self.sigma
    x = torch.tanh(self.fc2(x)).view(conv_shape)

    # Expand
    ret = self.deconv3(torch.tanh(self.deconv2(torch.tanh(self.deconv1(x))))).view(prev.shape[0], -1)

    return torch.clamp(ret, -1, 1)

# The input of the discriminator will be the previous and next audio segments. The
# output a classification of the input signal concatenation being true audio 
class Discriminator(nn.Module):
  def __init__(self, seconds, device):
    super(Discriminator, self).__init__()
    self.seconds = seconds
    
    self.c_int_sum= 4  # channels intermediate summary
    self.c_sum = 8     # channels summarized

    self.input_sample_rate = 44100
    self.sum_sample_rate = 100

    self.conv1 = nn.Conv1d(1, self.c_int_sum, kernel_size=21, stride=21).to(device)
    self.conv2 = nn.Conv1d(self.c_int_sum, self.c_sum, kernel_size=21, stride=21).to(device)

    self.flatten_size = self.seconds * self.c_sum * self.sum_sample_rate * 2
    self.fc1 = nn.Linear(self.flatten_size, self.flatten_size//2).to(device)
    self.fc2 = nn.Linear(self.flatten_size//2, 1).to(device)


  def forward(self, p, n):
    # x1 = F.relu(self.conv2(F.relu(self.conv1())))
    x = torch.cat((p, n), dim=-1)
    x = F.relu(self.conv2(F.relu(self.conv1(x.view(p.shape[0], 1, -1))))).view(p.shape[0], -1)
    x = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
    return x

SECONDS = 5
SAMPLE_RATE = 44100
SEGMENT_SIZE = int(SAMPLE_RATE)*SECONDS
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
SIGMA = 0.001
EPOCHS = 10000

dataset = AudioDataset(data, SEGMENT_SIZE, SAMPLE_RATE)

train_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

generator = Generator(SIGMA, SECONDS, device).to(device)
discriminator = Discriminator(SECONDS, device).to(device)

generator_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Loss function
adversarial_loss = torch.nn.BCELoss()

g_losses = []
d_losses = []


for epoch in range(EPOCHS):
  for batch_idx, (p, n) in enumerate(tqdm.tqdm(train_loader, total=math.ceil(dataset.length/BATCH_SIZE))):
    p, n = p.to(device), n.to(device)

    # print(batch_idx)

    # Ground truths
    valid = torch.ones(p.shape[0], 1).to(device)
    fake = torch.zeros(p.shape[0], 1).to(device)

    # ========= Train generator =========

    # Zero the gradient buffers
    generator_opt.zero_grad()

    # Generate the next audio segment
    g_n = generator(p)
  
    # Compute the loss 
    generator_loss = adversarial_loss(discriminator(p, g_n), valid)

    # Compute the gradient of the loss and update weights
    generator_loss.backward()
    generator_opt.step()

    # ========= Train discriminator =========

    # Zero the gradient buffers
    discriminator_opt.zero_grad()

    # Compute the loss
    real_loss = adversarial_loss(discriminator(p, n), valid)
    fake_loss = adversarial_loss(discriminator(p, g_n.detach()), fake)
    discriminator_loss = (real_loss + fake_loss) / 2

    # Compute the gradient of the loss and update weights
    discriminator_loss.backward()
    discriminator_opt.step()

    # Save loss data
    d_losses.append(discriminator_loss.item())
    g_losses.append(generator_loss.item())

  # Print statistics
  print(f"\n[Epoch: {epoch+1}] [Discriminator Loss: {discriminator_loss.item()}] [Generator Loss: {generator_loss.item()}]\n")

# end = time.time()
# duration = round(end - start, 2)
# print(f'Finished training with {EPOCHS} epochs in {duration}s')

# Graph losses
plt.plot(np.arange(len(d_losses)), d_losses, label='Discriminator Loss')
plt.plot(np.arange(len(g_losses)), g_losses, label='Generator Loss')
plt.title('Train Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Train loss')
plt.legend()
plt.show()


# SEGMENTS = 100

# segments = [torch.zeros(1, SEGMENT_SIZE).to(device)]
# for t in range(SEGMENTS):
#     noise = torch.randn(1, NOISE_DIM).to(device)
#     segments.append(generator(segments[-1], noise))

# result = torch.cat(segments[1:], axis=1).to("cpu")
# print(result.shape)
# print(result)

# min_ = torch.min(result)
# max_ = torch.max(result)

# result = (result - min_) / (max_ - min_)
# result = 2*result - 1

# print(result)

# torchaudio.save(os.path.join(OUTPUT_DIR, "test.mp3"), torch.cat((result, result), axis=0), SAMPLE_RATE)




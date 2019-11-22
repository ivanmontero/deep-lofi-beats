
"""deep_boi_beats.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rVcWyZXcR2-Q0WIti9GfRt2kfSZs_5gB

# ***Deep LoFi Beats***
---
* Ivan Montero
* Cameron "The Meissiah" Meissner
* Ani Canumalla
---
## Research questions
- Which size segment allows most creativity?
- Which neural network architecture lends itself best to music generation?

# Infrastructure Setup

Here, we install dependencies and setup the runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import multiprocessing
import math
import tqdm
import numpy as np

DATA_DIR = "soundcloud/"
OUTPUT_DIR = "outputs/"
SONG_NPY_DIR = "npys/"

SAMPLE_RATE = 44100 // 4  # Samples per second
SEGMENT_SIZE = int(SAMPLE_RATE)
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 4
NOISE_DIM = int(SAMPLE_RATE/4)
FEATURE_SIZE = 1
EPOCHS = 100

"""# Loading the data

Here, we load the data either via torchaudio to create
a repository of npy files each corresponding to a single song within the dataset, or directly from the existing npy files themselves.
"""

# Loading with torchaudio takes around 43 seconds,
# loading from npys is significantly faster (fastest run was 18 seconds)

def load():
    data = []
    for npy in tqdm.tqdm(os.listdir(SONG_NPY_DIR)):
        loaded = np.load(os.path.join(SONG_NPY_DIR, npy), allow_pickle=True)
        data.append((torch.from_numpy(loaded[0]), loaded[1]))

    return data


"""# Preprocessing the data
Here, we perform all the necessary steps to preprocess the data and define the multi-threaded dataloader for training and testing.
"""

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


""" LSTM
We have to flesh the section out...
"""

# https://miro.medium.com/max/1981/1*sO-SP58T4brE9EHazHSeGA.png


# Cam: Ani - I tried implementing this with my interpretation of what
# we talked about earlier. Prob gonna have to fix, but this should be a start\
class Seq2SeqAudioModel(nn.Module):

    def __init__(self, feature_size):
        super(Seq2SeqAudioModel, self).__init__()
        self.feature_size = feature_size

        # Encoder: stacked unidirectional LSTM
        self.encoder = nn.LSTM(self.feature_size, self.feature_size, num_layers=2, batch_first=True)

        # Decoder: stacked unidirectional LSTM
        self.decoder = nn.LSTM(self.feature_size, self.feature_size, num_layers=2, batch_first=True)

        # Fully connected layer to transform decoder outputs to a single float
        self.dense_layer = nn.Linear(self.feature_size, 1)

    def forward(self, prev_sequence):
        hidden = self.run_encoder(prev_sequence)
        x = self.decoder(prev_sequence, hidden)
        x = self.dense_layer(x)

    def run_encoder(self, prev_sequence):
        """
        Returns the final hidden state of the encoder after
        running over the entire previous sequence
        """
        hidden = None
        for sample in torch.squeeze(prev_sequence):
            _, hidden = self.encoder(sample, hidden)
        return hidden

    # This defines the function that gives a probability distribution and implements the temperature computation.
    def inference(self, x, states=None, temperature=1):
        x = x.view(-1, 1)
        x, states = self.forward(x, states)
        x = x.view(1, -1)
        x = x / max(temperature, 1e-20)
        x = F.softmax(x, dim=1)
        return x, states

    # Predefined loss function
    def loss(self, prediction, label, reduction='mean'):
        return F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)


"""# Training the LSTM"""


def train(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = AudioDataset(data, SEGMENT_SIZE, SAMPLE_RATE)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=os.cpu_count())

    model = Seq2SeqAudioModel(FEATURE_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    losses = []

    for epoch in range(EPOCHS):
        batch_losses = []
        for batch_idx, (prev_sequence, next_sequence) in enumerate(tqdm.tqdm(train_loader, total=math.ceil(dataset.length/BATCH_SIZE))):
            prev_sequence, next_sequence = prev_sequence.to(device), next_sequence.to(device)

            optimizer.zero_grad()
            predicted_next_sequence, hidden = model(prev_sequence)

            loss = model.loss(predicted_next_sequence, next_sequence)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss)
        avg_batch_loss = np.mean(batch_losses)
        losses.append(avg_batch_loss)
        print(f'[Epoch: {epoch+1}] [Loss: {avg_batch_loss}]')


    plt.plot(np.arange(len(losses)), losses)
    plt.title('Train Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = load()
    train(data)


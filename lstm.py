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


# https://miro.medium.com/max/1981/1*sO-SP58T4brE9EHazHSeGA.png

class AudioLSTM(nn.Module):

    # What will be the analog for vocab size?
    def __init__(self, feature_size):
        super(AudioLSTM, self).__init__()
        self.feature_size = feature_size
        self.encoder = nn.Embedding(self.vocab_size, self.feature_size)
        self.lstm = nn.LSTM(self.feature_size, self.feature_size, batch_first=True)
        self.decoder = nn.Linear(self.feature_size, self.vocab_size)
        
        # This shares the encoder and decoder weights as described in lecture.
        self.decoder.weight = self.encoder.weight
        self.decoder.bias.data.zero_()
        
        self.best_accuracy = -1
    
    def forward(self, x, states=None):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        
        x = self.encoder(x)
        x, states = self.lstm(x, states)
        x = self.decoder(x)

        return x, states

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

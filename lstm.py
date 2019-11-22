
"""
-------------LSTM-------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math
import tqdm
import numpy as np
from loader import load
from audio_processing import AudioDataset

# DATA_DIR = "soundcloud/"
# OUTPUT_DIR = "outputs/"
# SONG_NPY_DIR = "npys/"

# Cam: Ani - I tried implementing this with my interpretation of what
# we talked about earlier. Prob gonna have to fix, but this should be a start
SAMPLE_RATE = 44100 // 4  # Samples per second
SEGMENT_SIZE = int(SAMPLE_RATE)
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 4
NOISE_DIM = int(SAMPLE_RATE/4)
FEATURE_SIZE = 1
EPOCHS = 100

class EncoderRNN(nn.Module):
    def __init__(self, input_size, feature_size):
        self.input_size = input_size # instead of vocab size
        self.feature_size = feature_size
        #vself.embedding = nn.Embedding(input_size, self.feature_size)
        self.lstm = nn.LSTM(self.input_size, self.feature_size, num_layers=2, batch_first=True)

    def forward(self, sample, hidden):
        # embedded = self.embedding(sample)
        # output = embedded
        output, hidden = self.lstm(sample, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, feature_size):
        self.feature_size = feature_size
        self.lstm = nn.LSTM(self.feature_size, self.feature_size, batch_first=True)
        self.fc1 = nn.Linear(self.feature_size, 1) # should we be returning decoder hidden

    def forward(self, prev, hidden):
        output, hidden = self.lstm(prev, hidden)
        output = self.fc1(output)
        return output, hidden

class Seq2SeqAudioNet(nn.Module):
    def __init__(self,  encoder, decoder):
        super(Seq2SeqAudioNet, self).__init__()
        self.encoderRNN = encoder
        self.decoderRNN = decoder

    def forward(self, prev_seq):
        hidden = None
        for sample in torch.squeeze(prev_seq):
            _, hidden = self.encoderRNN(sample, hidden)
        output, decoder_hidden = self.decoderRNN(prev_seq, hidden)
        return output, decoder_hidden # should we be returning decoder hidden

    def loss(self, prediction, label):
        # prediction and label size is single float in a tensor
        return F.cross_entropy(prediction, label)


# class Seq2SeqAudioModel(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2SeqAudioModel, self).__init__()
#         self.encoderRNN = encoder
#         self.decoderRNN = decoder
#         # self.feature_size = feature_size
#         #
#         # # Encoder: stacked unidirectional LSTM
#         # self.encoder = nn.LSTM(self.feature_size, self.feature_size, num_layers=2, batch_first=True)
#         #
#         # # Decoder: stacked unidirectional LSTM
#         # self.decoder = nn.LSTM(self.feature_size, self.feature_size, num_layers=2, batch_first=True)
#         #
#         # # Fully connected layer to transform decoder outputs to a single float
#         # self.dense_layer = nn.Linear(self.feature_size, 1)
#
#     def forward(self, prev_sequence):
#         hidden = self.run_encoder(prev_sequence)
#         x = self.decoder(prev_sequence, hidden)
#         x = self.dense_layer(x)
#
#     def run_encoder(self, prev_sequence):
#         """
#         Returns the final hidden state of the encoder after
#         running over the entire previous sequence
#         """
#         hidden = None
#         for sample in torch.squeeze(prev_sequence):
#             _, hidden = self.encoder(sample, hidden)
#         return hidden
#
#     # Predefined loss function
#     def loss(self, prediction, label, reduction='mean'):
#         return F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)


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


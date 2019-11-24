
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
from audio_processing import get_batch

# DATA_DIR = "soundcloud/"
# OUTPUT_DIR = "outputs/"
# SONG_NPY_DIR = "npys/"

# Cam: Ani - I tried implementing this with my interpretation of what
# we talked about earlier. Prob gonna have to fix, but this should be a start
SAMPLE_RATE = 44100  # Samples per second
MAX_LEN = SAMPLE_RATE * 4 # 4 seconds max length for sequence
# SEGMENT_SIZE = int(SAMPLE_RATE)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 100
NOISE_DIM = int(SAMPLE_RATE/4)
HIDDEN_SIZE = 1
EPOCHS = 100
SEQ_IN_EPOCH = 1e5

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, device):
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(1, self.hidden_size, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(1, self.hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 1)
        self.device = device
        self.teacher_forcing_ratio = 0.5

    def forward(self, prev, next):
        output, hidden = self.lstm(prev, hidden)
        output = self.fc1(output)
        return output, hidden

    def encode(self, prev):
        hidden =  ( torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device),
                    torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device))
        for t in range(prev.shape[1]):
            _, hidden = self.encoder(prev[t], hidden)

        return hidden
    
    def decode_train(self, hidden, n):
        predictions = []

        next_input = torch.zeros(n.shape[0], 1, 1)
        for t in range(n.shape[1]):
            output, hidden = self.decoder(next_input, hidden)
            
            predictions.append(output.view(n.shape[0], 1))

            if random.random() < self.teacher_forcing_ratio:
                next_input = n[:,t].reshape(n.shape[0], 1, 1)
            else:
                next_input = predictions[-1].view(n.shape[0], 1, 1)
        
        predictions = torch.stack(predictions).permute(1, 0, 2)

        return predictions
    
    def decode(self, hidden, length):
        predictions = []

        next_input = torch.zeros(hidden.shape[0], 1, 1)
        for t in range(length):
            output, hidden = self.decoder(next_input, hidden)
            
            predictions.append(output.view(hidden.shape[0], 1))

            next_input = predictions[-1].view(hidden.shape[0], 1, 1)
        
        predictions = torch.stack(predictions).permute(1, 0, 2)

        return predictions

def train(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = load()

    model = Seq2SeqAudioModel(HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []

    for epoch in range(EPOCHS):
        batch_losses = []
        # for batch_idx, (prev_sequence, next_sequence) in enumerate(tqdm.tqdm(train_loader, total=math.ceil(dataset.length/BATCH_SIZE))):
        for i in range(SEQ_IN_EPOCH):
            # prev_sequence, next_sequence = prev_sequence.to(device), next_sequence.to(device)
            p, n = get_batch(data, MAX_LEN, BATCH_SIZE, device)

            optimizer.zero_grad()
            pred_n, hidden = model(p)

            loss = model.loss(pred_n, n)
            print(loss)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss)
        avg_batch_loss = np.mean(batch_losses)
        losses.append(avg_batch_loss)
        print(f"[Epoch: {epoch+1}] [Loss: {avg_batch_loss}]")

    plt.plot(np.arange(len(losses)), losses)
    plt.title('Train Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.legend()
    plt.show()

    return model


if __name__ == '__main__':
    data = load()
    model = train(data)
    torch.save(model.state_dict(), "lstm.pt")

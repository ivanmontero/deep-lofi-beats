
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
import random

# TODO: Play with below
SAMPLE_RATE = 44100  # Samples per second. Must match data
MAX_LEN = int(SAMPLE_RATE * 4) # 4 seconds max length for sequence
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 1
HIDDEN_SIZE = 2048
EPOCHS = 100
SEQ_IN_EPOCH = 10000

class Seq2Seq(nn.Module):
    def __init__(self, hidden_size, device):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(1, self.hidden_size, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(1, self.hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.device = device
        self.teacher_forcing_ratio = 0.5

    def forward(self, prev, next):
        encoded = self.encode(prev)
        return self.decode_train(encoded, next)

    def encode(self, prev):
        hidden =  ( torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device),
                    torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device))

        for t in range(prev.shape[1]):
            _, hidden = self.encoder(prev[:,t].view(prev.shape[0], 1, 1), hidden)

        return hidden
    
    def decode_train(self, hidden, n):
        predictions = []

        next_input = torch.zeros(n.shape[0], 1, 1)
        for t in range(n.shape[1]):
            output, hidden = self.decoder(next_input, hidden)

            pred = self.fc(output.view(n.shape[0], self.hidden_size))
            
            predictions.append(pred.view(n.shape[0], 1))

            if random.random() < self.teacher_forcing_ratio:
                next_input = n[:,t].reshape(n.shape[0], 1, 1)
            else:
                next_input = predictions[-1].view(n.shape[0], 1, 1)
        
        predictions = torch.stack(predictions).permute(1, 0, 2).view(n.shape)

        return predictions
    
    def decode(self, hidden, length):
        predictions = []

        next_input = torch.zeros(hidden.shape[0], 1, 1)
        for t in range(length):
            output, hidden = self.decoder(next_input, hidden)
            
            pred = self.fc(output.view(n.shape[0], self.hidden_size))
            
            predictions.append(pred.view(n.shape[0], 1))

            next_input = predictions[-1].view(hidden.shape[0], 1, 1)
        
        predictions = torch.stack(predictions).permute(1, 0, 2).view(next_input.shape[0], length)

        return predictions
    
    def loss(self, pred, real):
        return F.mse_loss(pred, real)

def train(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data, sample_rates = data

    model = Seq2Seq(HIDDEN_SIZE, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    all_losses = []

    for epoch in range(EPOCHS):
        batch_losses = []
        for i in range(SEQ_IN_EPOCH):
            p, n = get_batch(data, MAX_LEN, BATCH_SIZE, device)

            optimizer.zero_grad()
            pred_n = model(p, n)

            loss = model.loss(pred_n, n)
            print(loss)
            loss.backward()
            optimizer.step()
            all_losses.append(loss.detach().item())
            batch_losses.append(loss.detach().item())
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from loader import load
from audio_processing import get_batch
import random

SAMPLE_RATE = 44100  # Samples per second. Must match data
assert SAMPLE_RATE % 44100 == 0
# TODO if you change sample rate change the kernel size
MAX_LEN = int(SAMPLE_RATE * 2)  # 2 seconds max length for sequence
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 8 # change to 4-5 for inference, use
HIDDEN_SIZE = 512
EPOCHS = 10
SEQ_IN_EPOCH = 25


class ConvSeq2Seq(nn.Module):
    """
    :class ConvSeq2Seq is a variant of the standard sequence to sequence model, which uses
           convolutional layers in the encoder and deconvolutional layers in the decoder
    """
    def __init__(self, hidden_size, device):
        """
        :param hidden_size:
        :param device:
        """
        #TODO: can we get away with not moving any of this stuff to the GPU? @Cameron
        super(ConvSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(1, 20, kernel_size=21, stride=21).to(device)
        self.conv2 = nn.Conv1d(20, 64, kernel_size=21, stride=21).to(device)
        self.rnn_encoder = nn.LSTM(1, self.hidden_size, num_layers=2, batch_first=True, dropout=0.1).to(device)
        # self.bnorm1 = nn.BatchNorm1d(64).to(device)

        self.deconv1 = nn.ConvTranspose1d(64, 20, kernel_size=21, stride=21).to(device)
        self.deconv2 = nn.ConvTranspose1d(20, 1, kernel_size=21, stride=21).to(device)
        self.rnn_decoder = nn.LSTM(1, self.hidden_size, num_layers=2, batch_first=True, dropout=0.1).to(device)
        self.fc1 = nn.Linear(self.hidden_size, 1).to(device)

        self.device = device
        self.teacher_forcing_ratio = 0.5

    def forward(self, prev_tensor, next_tensor):
        """
        Architecture:
        [encoder]
        input ->
        conv1 ->
        conv2 ->
        rnn ->
        (encoded)


        [decoder]
        (encoded) ->
        {begin timestep} lstm ->
        (output of size hidden_size) ->
        linear layer ->
        deconv1 ->
        deconv2 ->
        (actual song segment, as prediction) ->
        conv1 ->
        conv2 ->
        (input to next timestep)
        """
        encoded = self.encoder(prev_tensor)
        decoded = self.decoder_train(encoded, next_tensor)
        return decoded

    def encoder(self, prev):
        """
        Same idea as encode except conforms to architecture we discussed
        :return: hidden state
        """
        hidden = (torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device),
                  torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device))
        for step in tqdm.tqdm(range(prev.shape[1]):
            x = self.conv1(prev)
            x = self.conv2(prev)
            _, hidden = self.rnn_encoder(prev[:, step].view(prev.shape[0], 1, 1), hidden)
        return hidden

    def decoder_train(self, hidden, next_tensor):
        """
        """
        predictions = []
        next_input = torch.zeros(next_tensor.shape[0], 1, 1, device=self.device)
        for t in range(next_tensor.shape[1]):
            output, hidden = self.rnn_decoder(next_input, hidden)

            pred = self.fc(output.view(next_tensor.shape[0], self.hidden_size))

            predictions.append(pred.view(next_tensor.shape[0], 1))

            if random.random() < self.teacher_forcing_ratio:
                next_input = next_tensor[:, t].reshape(next_tensor.shape[0], 1, 1)
            else:
                next_input = predictions[-1].view(next_tensor.shape[0], 1, 1)

        predictions = torch.stack(predictions).permute(1, 0, 2).view(next_tensor.shape)
        return predictions

    def encode(self, prev):
        # TODO: delete once encoder method starts to work
        hidden = (torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device),
                  torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device))

        for t in tqdm.tqdm(range(prev.shape[1])):
            _, hidden = self.rnn_encoder(prev[:, t].view(prev.shape[0], 1, 1), hidden)

        return hidden

    def inference(self, prev, next_len):
        encoded = self.encode(prev)
        return self.decode_train(encoded, next_len)

    def loss(self, prediction, label):
        """
        square loss used for predicting real values
        :param prediction:
        :param label:
        :return:
        """
        return F.mse_loss(prediction, label)

    def decode_train(self, hidden, n):
        predictions = []

        next_input = torch.zeros(n.shape[0], 1, 1, device=self.device)
        for t in range(n.shape[1]):
            output, hidden = self.rnn_decoder(next_input, hidden)

            pred = self.fc(output.view(n.shape[0], self.hidden_size))

            predictions.append(pred.view(n.shape[0], 1))

            if random.random() < self.teacher_forcing_ratio:
                next_input = n[:, t].reshape(n.shape[0], 1, 1)
            else:
                next_input = predictions[-1].view(n.shape[0], 1, 1)

        predictions = torch.stack(predictions).permute(1, 0, 2).view(n.shape)
        return predictions

    def decode(self, hidden, length):
        pass

def train(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Using device: {}'.format(device))

    data, sample_rates = data

    # print(data.shape)

    model = ConvSeq2Seq(HIDDEN_SIZE, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    all_losses = []

    for epoch in range(EPOCHS):
        batch_losses = []
        for _ in tqdm.tqdm(range(SEQ_IN_EPOCH)):
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
        losses.append(np.mean(batch_losses))
        print(f"[Epoch: {epoch+1}] [Loss: {avg_batch_loss}]")
        torch.save(model.state_dict(), "checkpoints/conv_lstm{}.pt".format(epoch+1))

    plt.plot(np.arange(len(losses)), losses)
    plt.title('Train Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.legend()
    plt.show()

    return model


if __name__ == '__main__':
    audio_data = load()
    model = train(audio_data)
    torch.save(model.state_dict(), "checkpoints/conv_lstm_final.pt")

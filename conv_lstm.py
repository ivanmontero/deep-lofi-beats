import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from loader import load
from audio_processing import get_batch
import random

MAX_LEN = 10 # seconds
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 1 # change to 4-5 for inference, use
HIDDEN_SIZE = 512
EPOCHS = 10
SEQ_IN_EPOCH = 25

# INPUT SIZE IS 44100
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
        self.c_int_sums = [2, 8, 32]  # channels intermediate summary
        self.c_sum = 64     # channels summarized (channels in summarized)

        self.input_sample_rate = 44100
        self.sum_sample_rate = 42
        self.sum_seg_size = self.input_sample_rate // self.sum_sample_rate

        # 3 -> 5 -> 7 -> 10

        self.conv1 = nn.Conv1d(1, self.c_int_sums[0], kernel_size=3, stride=3).to(device)
        self.conv2 = nn.Conv1d(self.c_int_sums[0], self.c_int_sums[1], kernel_size=5, stride=5).to(device)
        self.conv3 = nn.Conv1d(self.c_int_sums[1], self.c_int_sums[2], kernel_size=7, stride=7).to(device)
        self.conv4 = nn.Conv1d(self.c_int_sums[2], self.c_sum, kernel_size=10, stride=10).to(device)
        # self.bnorm1 = nn.BatchNorm1d(64).to(device)
        # TODO: How to think about input_size relative to the number of output channels from the conv layers
        self.rnn_encoder = nn.LSTM(self.c_sum, self.hidden_size, num_layers=2, batch_first=True).to(device)
        self.rnn_decoder = nn.LSTM(self.c_sum, self.hidden_size, num_layers=2, batch_first=True).to(device)

        self.deconv1 = nn.ConvTranspose1d(self.c_sum, self.c_int_sums[2], kernel_size=10, stride=10).to(device)
        self.deconv2 = nn.ConvTranspose1d(self.c_int_sums[2], self.c_int_sums[1], kernel_size=7, stride=7).to(device)
        self.deconv3 = nn.ConvTranspose1d(self.c_int_sums[1], self.c_int_sums[0], kernel_size=5, stride=5).to(device)
        self.deconv4 = nn.ConvTranspose1d(self.c_int_sums[0], 1, kernel_size=3, stride=3).to(device)

        self.fc = nn.Linear(self.hidden_size, self.c_sum*1).to(device)

        self.device = device
        self.teacher_forcing_ratio = 0.0

    def forward(self, prev_tensor, next_tensor):
        encoded = self.encode(prev_tensor)
        decoded = self.decode_train(encoded, next_tensor)
        return decoded
    
    # Summarizes raw audio
    # in: (batch_size, 1, seq_len)
    # out: (batch_size, self.c_sum, seq_len // (21*21))
    def summarize(self, x):
        # Resize to fit into conv layer
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.conv4(x)
        return x
    
    # in: (batch_size, self.c_sum, seq_len // (21*21))
    # out: (batch_size, 1, seq_len)
    def desummarize(self, x):
        x = self.deconv1(x)
        x = torch.tanh(x)
        x = self.deconv2(x)
        x = torch.tanh(x)
        x = self.deconv3(x)
        x = torch.tanh(x)
        x = self.deconv4(x)
        return x


    def encode(self, prev):
        hidden = (torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device),
                  torch.zeros(2, prev.shape[0], self.hidden_size, device=self.device))

        x = self.summarize(prev.view(prev.shape[0], 1, -1))
        # x: (batch_size, self.c_sum, seq_len // (21*21))

        # lstm expects: (batch_size, seq, self.c_sum)
        x = x.permute(0, 2, 1)

        # Loop to prevent cuda error
        for t in range(x.shape[1]):
            _, hidden = self.rnn_encoder(x[:, t].view(x.shape[0], 1, x.shape[2]), hidden)

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
        # each prediction takes shape, from loop: (n.shape[1] // self.sum_seg_size, batch_size, self.sum_seg_size)
        # permute s. t. (batch_size, n.shape[1] // self.sum_seg_size, self.sum_seg_size)

        next_input = torch.randn(n.shape[0], 1, self.c_sum, device=self.device)
        for t in range(n.shape[1] // self.sum_seg_size):
            # lstm expects: (batch_size, 1, self.c_sum)
            output, hidden = self.rnn_decoder(next_input, hidden)

            # deconv expects: (batch_size, self.c_sum, 1)
            hidden_to_deconv = self.fc(torch.tanh(output.view(n.shape[0], self.hidden_size))).view(n.shape[0], self.c_sum, 1)

            pred = self.desummarize(hidden_to_deconv)

            predictions.append(pred.view(n.shape[0], -1))

            # if random.random() < self.teacher_forcing_ratio:
            #     next_input = self.summarize(n[:,t*self.sum_seg_size:(t+1)*self.sum_seg_size].view(n.shape[0], 1, self.sum_seg_size))
            # else:
            #     next_input = self.summarize(pred)
            # next_input = next_input.permute(0, 2, 1)
            next_input = torch.randn(n.shape[0], 1, self.c_sum, device=self.device)

        predictions = torch.stack(predictions).permute(1, 0, 2).reshape(n.shape)
        return predictions

    # Length on the scale of seconds.
    def decode(self, hidden, length):
        batch_size = hidden[0].shape[1]
        predictions = []
        # each prediction takes shape, from loop: (n.shape[1] // self.sum_seg_size, batch_size, self.sum_seg_size)
        # permute s. t. (batch_size, n.shape[1] // self.sum_seg_size, self.sum_seg_size)

        next_input = torch.randn(batch_size, 1, self.c_sum, device=self.device)
        for t in range(length*self.sum_sample_rate):
            # lstm expects: (batch_size, 1, self.c_sum)
            output, hidden = self.rnn_decoder(next_input, hidden)

            # deconv expects: (batch_size, self.c_sum, 1)
            hidden_to_deconv = self.fc(torch.tanh(output.view(batch_size, self.hidden_size))).view(batch_size, self.c_sum, 1)

            pred = self.desummarize(hidden_to_deconv)

            predictions.append(pred.view(batch_size, -1))

            next_input = torch.randn(batch_size, 1, self.c_sum, device=self.device)

        predictions = torch.stack(predictions).permute(1, 0, 2).reshape((batch_size, -1))
        return predictions

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
            p, n = get_batch(data, MAX_LEN, BATCH_SIZE, device, segment_size=44100, full=True)

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
    audio_data = load(enforce_samplerate=44100)
    model = train(audio_data)
    torch.save(model.state_dict(), "checkpoints/conv_lstm_final.pt")

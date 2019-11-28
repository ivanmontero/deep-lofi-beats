import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from loader import load
from audio_processing import get_batch
import random

AUTOENCODE = False

# SMMRY PARAMS
SMMRY_EPOCHS = 1
SMMRY_SEQ_IN_EPOCH = 1000
SMMRY_MAX_LEN = 10
SMMRY_BATCH_SIZE = 64


# NET PARAMS
MAX_LEN = 10 # seconds
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
BATCH_SIZE = 64 # change to 4-5 for inference, use
HIDDEN_SIZE = 512
EPOCHS = 10
SEQ_IN_EPOCH = 25

# INPUT SIZE IS 44100
class ConvSeq2Seq(nn.Module):
    """
    :class ConvSeq2Seq is a variant of the standard sequence to sequence model, which uses
           convolutional layers in the encoder and deconvolutional layers in the decoder
    """
    def __init__(self, hidden_size, device, smmry=None):
        """
        :param hidden_size:
        :param device:
        """
        #TODO: can we get away with not moving any of this stuff to the GPU? @Cameron
        super(ConvSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.c_int_sum= 16  # channels intermediate summary
        self.c_sum = 64     # channels summarized

        self.input_sample_rate = 44100
        self.sum_sample_rate = 100
        self.sum_seg_size = self.input_sample_rate // self.sum_sample_rate

        # self.conv1 = nn.Conv1d(1, self.c_int_sum, kernel_size=21, stride=21).to(device)
        # self.conv2 = nn.Conv1d(self.c_int_sum, self.c_sum, kernel_size=21, stride=21).to(device)
        # self.bnorm1 = nn.BatchNorm1d(64).to(device)
        # TODO: How to think about input_size relative to the number of output channels from the conv layers
        self.rnn_encoder = nn.LSTM(self.c_sum, self.hidden_size, num_layers=2, batch_first=True, dropout=0.1).to(device)
        self.rnn_decoder = nn.LSTM(self.c_sum, self.hidden_size, num_layers=2, batch_first=True, dropout=0.1).to(device)

        # self.deconv1 = nn.ConvTranspose1d(self.c_sum, self.c_int_sum, kernel_size=21, stride=21).to(device)
        # self.deconv2 = nn.ConvTranspose1d(self.c_int_sum, 1, kernel_size=21, stride=21).to(device)

        if smmry is not None:
            for param in smmry.parameters():
                param.requires_grad = False
            self.smmry = smmry
        else:
            self.smmry = ConvSummary(device)

        self.fc = nn.Linear(self.hidden_size, self.c_sum*1).to(device)

        self.device = device
        self.teacher_forcing_ratio = 0.5

    def forward(self, prev_tensor, next_tensor):
        encoded = self.encode(prev_tensor)
        decoded = self.decode_train(encoded, next_tensor)
        return decoded
    
    # # Summarizes raw audio
    # # in: (batch_size, 1, seq_len)
    # # out: (batch_size, self.c_sum, seq_len // (21*21))
    # def summarize(self, x):
    #     # Resize to fit into conv layer
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     return x
    
    # # in: (batch_size, self.c_sum, seq_len // (21*21))
    # # out: (batch_size, 1, seq_len)
    # def desummarize(self, x):
    #     x = self.deconv1(x)
    #     x = self.deconv2(x)
    #     return x


    def encode(self, prev):
        x = self.smmry.summarize(prev.view(prev.shape[0], 1, -1))
        # x: (batch_size, self.c_sum, seq_len // (21*21))

        # lstm expects: (batch_size, seq, self.c_sum)
        x = x.permute(0, 2, 1)

        # This hopefully wont hit a cuda error now
        _, hidden = self.rnn_encoder(x)

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

        next_input = torch.zeros(n.shape[0], 1, self.c_sum, device=self.device)
        for t in range(n.shape[1] // self.sum_seg_size):
            # lstm expects: (batch_size, 1, self.c_sum)
            output, hidden = self.rnn_decoder(next_input, hidden)

            # deconv expects: (batch_size, self.c_sum, 1)
            hidden_to_deconv = self.fc(output.view(n.shape[0], self.hidden_size)).view(n.shape[0], self.c_sum, 1)

            pred = self.smmry.desummarize(hidden_to_deconv)

            predictions.append(pred.view(n.shape[0], -1))

            if random.random() < self.teacher_forcing_ratio:
                next_input = self.smmry.summarize(n[:,t*self.sum_seg_size:(t+1)*self.sum_seg_size].view(n.shape[0], 1, self.sum_seg_size))
            else:
                next_input = self.smmry.summarize(pred.detach())
            next_input = next_input.permute(0, 2, 1)

        predictions = torch.stack(predictions).permute(1, 0, 2).reshape(n.shape)
        return predictions

    # Length on the scale of seconds.
    def decode(self, hidden, length):
        batch_size = hidden[0].shape[1]
        predictions = []
        # each prediction takes shape, from loop: (n.shape[1] // self.sum_seg_size, batch_size, self.sum_seg_size)
        # permute s. t. (batch_size, n.shape[1] // self.sum_seg_size, self.sum_seg_size)

        next_input = torch.zeros(n.shape[0], 1, self.c_sum, device=self.device)
        for t in range(length*self.sum_sample_rate):
            # lstm expects: (batch_size, 1, self.c_sum)
            output, hidden = self.rnn_decoder(next_input, hidden)

            # deconv expects: (batch_size, self.c_sum, 1)
            hidden_to_deconv = self.fc(output.view(batch_size, self.hidden_size)).view(batch_size, self.c_sum, 1)

            pred = self.smmry.desummarize(hidden_to_deconv)

            predictions.append(pred.view(batch_size, -1))

            next_input = self.smmry.summarize(pred.detach()).permute(0, 2, 1)

        predictions = torch.stack(predictions).permute(1, 0, 2).reshape((batch_size, -1))
        return predictions


class ConvSummary(nn.Module):
    def __init__(self, device):
        super(ConvSummary, self).__init__()
        self.c_int_sum= 16  # channels intermediate summary
        self.c_sum = 64     # channels summarized

        self.conv1 = nn.Conv1d(1, self.c_int_sum, kernel_size=21, stride=21).to(device)
        self.conv2 = nn.Conv1d(self.c_int_sum, self.c_sum, kernel_size=21, stride=21).to(device)

        self.deconv1 = nn.ConvTranspose1d(self.c_sum, self.c_int_sum, kernel_size=21, stride=21).to(device)
        self.deconv2 = nn.ConvTranspose1d(self.c_int_sum, 1, kernel_size=21, stride=21).to(device)

        self.device = device
    
    def forward(self, x):
        return self.desummarize(self.summarize(x))
    
    def summarize(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def desummarize(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x
    
    def loss(self, pred, true):
        return F.mse_loss(pred, true)

def train(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Using device: {}'.format(device))

    data, sample_rates = data

    # print(data.shape)

    # Stage one, train summarizer
    if AUTOENCODE:
        smmry_losses = []
        smmry = ConvSummary(device)
        smmry_optimizer = torch.optim.Adam(smmry.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for epoch in range(SMMRY_EPOCHS):
            for _ in tqdm.tqdm(range(SMMRY_SEQ_IN_EPOCH)):
                p, n = get_batch(data, SMMRY_MAX_LEN, SMMRY_BATCH_SIZE, device, segment_size=44100)

                smmry_optimizer.zero_grad()

                s_p = smmry(p.view(p.shape[0], 1, -1)).view(p.shape[0], -1)
                s_n = smmry(n.view(n.shape[0], 1, -1)).view(n.shape[0], -1)

                loss = (smmry.loss(s_p, p) + smmry.loss(s_n, n))/2
                print(loss)
                loss.backward()
                smmry_optimizer.step()
                smmry_losses.append(loss.detach().item())
            print(f"[Epoch: {epoch+1}]") 
    else:
        smmry = None


    model = ConvSeq2Seq(HIDDEN_SIZE, device, smmry)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    all_losses = []

    for epoch in range(EPOCHS):
        batch_losses = []
        for _ in tqdm.tqdm(range(SEQ_IN_EPOCH)):
            p, n = get_batch(data, MAX_LEN, BATCH_SIZE, device, segment_size=44100)

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

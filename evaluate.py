import torch
import numpy as np
from lstm import Seq2Seq, HIDDEN_SIZE
from loader import SONG_NPY_DIR, OUTPUT_DIR
import os
from loader import load
from audio_processing import get_batch
import matplotlib.pyplot as plt

TRAINED_STATE = 'checkpoints/lstm_final.pt'
IS_WINDOWS = True

def load_model_from_checkpoint(checkpoint, device):
    model = Seq2Seq(HIDDEN_SIZE, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))

    data, sample_rates = load()
    # 10 seconds of audio
    seq_len = 10 * sample_rates[0]

    print('Loading trained model...')
    model = load_model_from_checkpoint(TRAINED_STATE, device)
    print('Performing inference...')

    prev, _ = get_batch(data, 10*sample_rates[0], 1, device)

    print('Encoding seed sequence')
    hidden = model.encode(prev)

    print('Producing sequence')
    audio = model.decode(hidden, seq_len)

    print('Saving result')
    np.save(os.path.join(OUTPUT_DIR, 'prediction.npy'), audio[0].detach().numpy())
    if not IS_WINDOWS:
        import torchaudio
        torchaudio.save("prediction.mp3", torch.stack((audio[0], audio[0])), sample_rates[0])

    plt.plot(audio[0].detach().numpy())
    plt.show()

    return audio


if __name__ == '__main__':
    pred = main()
    print(pred)





# def train(data):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     print('Using device: {}'.format(device))

#     data, sample_rates = data

#     # print(data.shape)

#     model = Seq2Seq(HIDDEN_SIZE, device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     losses = []
#     all_losses = []

#     for epoch in range(EPOCHS):
#         batch_losses = []
#         for _ in tqdm.tqdm(range(SEQ_IN_EPOCH)):
#             p, n = get_batch(data, MAX_LEN, BATCH_SIZE, device)

#             optimizer.zero_grad()
#             pred_n = model(p, n)

#             loss = model.loss(pred_n, n)
#             print(loss)
#             loss.backward()
#             optimizer.step()
#             all_losses.append(loss.detach().item())
#             batch_losses.append(loss.detach().item())
#         avg_batch_loss = np.mean(batch_losses)
#         losses.append(np.mean(batch_losses))
#         print(f"[Epoch: {epoch+1}] [Loss: {avg_batch_loss}]")
#         torch.save(model.state_dict(), "checkpoints/lstm{}.pt".format(epoch+1))

#     plt.plot(np.arange(len(losses)), losses)
#     plt.title('Train Loss vs. Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Train loss')
#     plt.legend()
#     plt.show()

#     return model


# if __name__ == '__main__':
#     audio_data = load()
#     model = train(audio_data)
#     torch.save(model.state_dict(), "checkpoints/lstm_final.pt")


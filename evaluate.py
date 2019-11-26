import torch
import numpy as np
from lstm import Seq2Seq, HIDDEN_SIZE
from loader import SONG_NPY_DIR, OUTPUT_DIR
import os

TRAINED_STATE = 'checkpoints/lstm_final.pt'


def load_model_from_checkpoint(checkpoint, device):
    model = Seq2Seq(HIDDEN_SIZE, device)
    model.load_state_dict(torch.load(checkpoint))
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: {}'.format(device))

    audio, samples_per_second = np.load(
        os.path.join(SONG_NPY_DIR, np.random.choice(os.listdir(SONG_NPY_DIR), 1)[0]),
        allow_pickle=True)

    print(audio.shape)

    # Get the left channel of audio
    audio = audio[0].reshape(1, -1)

    print(audio.shape, samples_per_second)

    audio = torch.from_numpy(audio)
    # 10 seconds of audio
    seq_len = 1 * samples_per_second

    # Get 0s - 30s in the input song
    # and predict 30s - 40s
    model_input = audio[:, :1*samples_per_second].to(device)
    print(model_input.size())

    print('Loading trained model...')
    model = load_model_from_checkpoint(TRAINED_STATE, device)
    print('Performing inference...')

    pred = model.inference(model_input, seq_len).numpy()
    print(pred.shape)

    np.save(os.path.join(OUTPUT_DIR, 'prediction.npy'), pred)
    return pred


if __name__ == '__main__':
    pred = main()
    print(pred)



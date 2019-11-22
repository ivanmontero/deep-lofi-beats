
import torch
import torchaudio
import os
import multiprocessing
import tqdm
import numpy as np

DATA_DIR = "soundcloud/"
OUTPUT_DIR = "outputs/"
SONG_NPY_DIR = "npys/"


def load():
    if len(os.listdir(SONG_NPY_DIR)) == 0:
        # There aren't any npys, so we have to load the data
        # with torchaudio.load and create the npys for future loads
        print('Loading data with torchaudio...')

        # Search the data directory for audio files
        audio_file_names = []
        for idx, f in enumerate(os.listdir(DATA_DIR)):
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

    return data



import torch
import os
import tqdm
import numpy as np

DATA_DIR = "soundcloud/"
OUTPUT_DIR = "outputs/"
SONG_NPY_DIR = "npys/"


# TODO: Create .npy files that are downsampled (see colab)
def load():
    # We have npys we can read from, so lets do that
    print('Reading data from npy files...')

    data = []
    sample_rates = []
    for npy in tqdm.tqdm(os.listdir(SONG_NPY_DIR)):
        loaded = np.load(os.path.join(SONG_NPY_DIR, npy), allow_pickle=True)
        data.append(loaded[0][0])
        sample_rates.append(loaded[1])

    return np.array(data), np.array(sample_rates)




import torch
import os
import tqdm
import numpy as np

DATA_DIR = "soundcloud/"
OUTPUT_DIR = "outputs/"
SONG_NPY_DIR = "npys/"


def load():
    # We have npys we can read from, so lets do that
    print('Reading data from npy files...')

    data = []
    for npy in tqdm.tqdm(os.listdir(SONG_NPY_DIR)):
        loaded = np.load(os.path.join(SONG_NPY_DIR, npy), allow_pickle=True)
        data.append((torch.from_numpy(loaded[0]), loaded[1]))

    return data


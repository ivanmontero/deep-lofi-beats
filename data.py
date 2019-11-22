

import numpy as np
import random




# Assumes data is n x seq, where n is the amount of audio files and seq varies
def get_batch(data, max_len, batch_size):
    batch = data[np.random.choice(data.shape[0], batch_size)]

    len1 = random.randint(1, max_len - 1)
    len2 = random.randint(1, max_len - len2)

    p, n = np.zeros(batch_size, len1), np.zeros(batch_size, len2)

    for i in range(batch_size):
        start = random.randint(0, batch[i].shape[0] - len2 - len2)
        p[i] = batch[i][start:start+len1]
        n[i] = batch[i][start+len1:start+len1+len2]

    return p, n


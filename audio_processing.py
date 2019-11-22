
import torch
import torchaudio
import math

# An interator for a single audio.
class AudioIterator:
    def __init__(self, audio, segment_size):
        self.audio = audio
        self.segment_size = segment_size
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Currently, returns previous and next segment, and iterates over the audio just
        # like a conv layer where kernel_size == stride. E.g.:
        # first_iter:
        #     [ s_0 | s_1 | s_2 | s_3 | s_4 | ... ]
        #      --p-- --n--
        # second_iter:
        #     [ s_0 | s_1 | s_2 | s_3 | s_4 | ... ]
        #            --p-- --n--
        # if we want to have much more examples, we might want to explore shifting
        # the p and n window by a "stride".
        if (self.idx + 2) * self.segment_size <= self.audio.shape[0]:
            p, n = self.audio[self.idx*self.segment_size:(self.idx+1)*self.segment_size], self.audio[(self.idx+1)*self.segment_size:(self.idx+2)*self.segment_size]
            self.idx += 1
            return p, n
        else:
            raise StopIteration


# The iterator each worker will get. Iterates through multiple audios.
class MultiAudioIterator:
    def __init__(self, audio_list, segment_size):
        self.audio_list = audio_list
        self.segment_size = segment_size
        self.audio_iters = [iter(AudioIterator(audio, segment_size)) for audio in audio_list]
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True and len(self.audio_iters) != 0:
            self.idx %= len(self.audio_iters)
            try:
                ret = next(self.audio_iters[self.idx])
                self.idx += 1
                return ret
            except StopIteration:
                del self.audio_iters[self.idx]
        raise StopIteration


# The overaching dataset. We only want to deal with mono.
# TODO(ivamon): Pass in list of filenames here; in a multiprocessing manner,
# read in the data, process into necessary format, then delete the orignal data.
# This will save space.
class AudioDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, segment_size, sample_rate):
        # files: a list of file names
        # segment_size: Size of each audio segment, in waveframe length
        # sample_rate
        super(AudioDataset).__init__()
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.audio = []
        self.length = 0
        # This is the for loop that we want to multiprocess, and add in file loading.
        for d in data:
            if d[1] != sample_rate:
                self.audio.append(torchaudio.transforms.Resample(d[1], sample_rate)(d[0])[0])
            else:
                self.audio.append(d[0][0])
            self.length += self.audio[-1].shape[0] // segment_size - 1

    # Audio file num == idx % len(self.files)
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return iter(MultiAudioIterator(self.audio, self.segment_size))
        else:
            per_worker = int(math.ceil(len(self.audio) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = per_worker*worker_id
            iter_end = min(iter_start + per_worker, len(self.audio))
            return iter(MultiAudioIterator(self.audio[iter_start:iter_end], self.segment_size))
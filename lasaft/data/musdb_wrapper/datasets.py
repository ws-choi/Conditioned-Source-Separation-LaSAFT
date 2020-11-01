import musdb
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import random

from lasaft.utils.fourier import get_trim_length


class MusdbLoader(object):

    def __init__(self, musdb_root='data/musdb18_wav/', is_wav=True):
        self.musdb_train = musdb.DB(root=musdb_root, subsets="train", split='train', is_wav=is_wav)
        self.musdb_valid = musdb.DB(root=musdb_root, subsets="train", split='valid', is_wav=is_wav)
        self.musdb_test = musdb.DB(root=musdb_root, subsets="test", is_wav=is_wav)

        if len(self.musdb_train) > 0:
            pass
        else:
            raise FileNotFoundError('musdb_root does exist or is not a valid path. \n'
                                    'please try a valid path for --musdb_root \n'
                                    'You should also check whether the given musdb path is wave-filed version'
                                    ' (i.e., --musdb_is_wav True) or not.')


class MusdbTrainSet(Dataset):

    def __init__(self, musdb_train, n_fft=2048, hop_length=1024, num_frame=64, target_names=None, cache_mode=True,
                 dev_mode=False, one_hot_mode=False):

        self.musdb_train = musdb_train
        self.window_length = hop_length * (num_frame - 1)

        self.lengths = [track.samples for track in self.musdb_train]
        self.source_names = ['vocals', 'drums', 'bass', 'other']  # == self.musdb_train.targets_names[:-2]

        if target_names is None:
            self.target_names = self.source_names
        else:
            self.target_names = target_names

        self.num_tracks = len(self.musdb_train)

        # development mode
        if dev_mode:
            self.num_tracks = 1
            self.lengths = self.lengths[:1]

        self.cache_mode = cache_mode
        if cache_mode:
            self.cache_dataset()

        self.one_hot_mode = one_hot_mode

    def cache_dataset(self):
        self.cache = {}
        print('cache audio files.')
        for idx in tqdm(range(self.num_tracks)):
            self.cache[idx] = {}
            for source in self.source_names:
                self.cache[idx][source] = self.musdb_train[idx].targets[source].audio.astype(np.float32)

    def __len__(self):
        return sum([length // self.window_length for length in self.lengths]) * len(self.target_names)

    def __getitem__(self, whatever):
        source_sample = {target: self.get_random_audio_sample(target) for target in self.source_names}
        rand_target = np.random.choice(self.target_names)

        mixture = sum(source_sample.values())
        target = source_sample[rand_target]

        if self.one_hot_mode:
            input_condition = np.zeros(len(self.target_names), dtype=np.float32)
            input_condition[self.target_names.index(rand_target)] = 1.
        else:
            input_condition = np.array(self.target_names.index(rand_target), dtype=np.long)

        return torch.from_numpy(mixture), torch.from_numpy(target), torch.tensor(input_condition, dtype=torch.long)

    def get_random_audio_sample(self, target_name):
        return self.get_audio_sample(random.randint(0, self.num_tracks - 1), target_name)

    def get_audio_sample(self, idx, target_name):
        length = self.lengths[idx] - self.window_length
        start_position = random.randint(0, length - 1)
        return self.get_audio(idx, target_name, start_position, self.window_length)

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode:
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_train[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]


class MusdbTestSet(Dataset):

    def __init__(self, musdb_test, n_fft=2048, hop_length=1024, num_frame=64, target_names=None, cache_mode=True,
                 dev_mode=False, one_hot_mode=False):

        self.hop_length = hop_length
        self.musdb_test = musdb_test
        self.window_length = hop_length * (num_frame - 1)
        self.trim_length = get_trim_length(self.hop_length)
        self.true_samples = self.window_length - 2 * self.trim_length

        self.lengths = [track.samples for track in self.musdb_test]
        self.source_names = ['vocals', 'drums', 'bass', 'other']  # == self.musdb_train.targets_names[:-2]

        if target_names is None:
            self.target_names = self.source_names
        else:
            self.target_names = target_names

        self.num_tracks = len(self.musdb_test)
        # development mode
        if dev_mode:
            self.num_tracks = 1
            self.lengths = self.lengths[:1]

        import math
        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.acc_chunk_final_ids = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        self.cache_mode = cache_mode
        if cache_mode:
            self.cache_dataset()

        self.one_hot_mode = one_hot_mode

    def cache_dataset(self):
        self.cache = {}
        print('cache audio files.')
        for idx in tqdm(range(self.num_tracks)):
            self.cache[idx] = {}
            self.cache[idx]['linear_mixture'] = self.musdb_test[idx].targets['linear_mixture'].audio.astype(
                np.float32)

    def __len__(self):
        return self.acc_chunk_final_ids[-1] * len(self.target_names)

    def __getitem__(self, idx):

        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)

        target_name = self.target_names[target_offset]

        mixture, mixture_idx, offset = self.get_mixture_sample(idx)

        if self.one_hot_mode:
            input_condition = np.zeros(len(self.target_names), dtype=np.float32)
            input_condition[target_offset] = 1.
        else:
            input_condition = np.array(target_offset, dtype=np.long)

        mixture = torch.from_numpy(mixture)
        input_condition = torch.tensor(input_condition, dtype=torch.long)
        window_offset = offset // self.true_samples

        return mixture, mixture_idx, window_offset, input_condition, target_name

    def get_mixture_sample(self, idx):
        mixture_idx, start_pos = self.idx_to_track_offset(idx)
        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        return mixture, mixture_idx, start_pos

    def idx_to_track_offset(self, idx):

        for mixture_idx, last_chunk in enumerate(self.acc_chunk_final_ids):
            if idx < last_chunk:
                if mixture_idx != 0:
                    offset = (idx - self.acc_chunk_final_ids[mixture_idx - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return mixture_idx, offset

        return None, None

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode and target_name == 'linear_mixture':
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_test[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]


class MusdbValidSet(Dataset):

    def __init__(self, musdb_valid, n_fft=2048, hop_length=1024, num_frame=64, target_names=None, cache_mode=True,
                 dev_mode=False, one_hot_mode=False):

        self.hop_length = hop_length
        self.musdb_valid = musdb_valid
        self.window_length = hop_length * (num_frame - 1)
        self.trim_length = get_trim_length(self.hop_length)

        self.true_samples = self.window_length - 2 * self.trim_length

        self.lengths = [track.samples for track in self.musdb_valid]
        self.source_names = ['vocals', 'drums', 'bass', 'other']  # == self.musdb_train.targets_names[:-2]

        if target_names is None:
            self.target_names = self.source_names
        else:
            self.target_names = target_names

        self.num_tracks = len(self.musdb_valid)
        # development mode
        if dev_mode:
            self.num_tracks = 1
            self.lengths = self.lengths[:1]

        import math
        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.acc_chunk_final_ids = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        self.cache_mode = cache_mode
        if cache_mode:
            self.cache_dataset()

        self.one_hot_mode = one_hot_mode

    def cache_dataset(self):
        self.cache = {}
        print('cache audio files.')
        for idx in tqdm(range(self.num_tracks)):
            self.cache[idx] = {}
            for source in self.source_names + ['linear_mixture']:
                self.cache[idx][source] = self.musdb_valid[idx].targets[source].audio.astype(np.float32)

    def __len__(self):
        return self.acc_chunk_final_ids[-1] * len(self.target_names)

    def __getitem__(self, idx):

        target_offset = idx % len(self.target_names)
        idx = idx // len(self.target_names)

        target_name = self.target_names[target_offset]
        mixture_idx, start_pos = self.idx_to_track_offset(idx)

        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)
        target = self.get_audio(mixture_idx, target_name, start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        target = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), target,
                                 np.zeros((right_padding_num, 2), dtype=np.float32)), 0)

        if self.one_hot_mode:
            input_condition = np.zeros(len(self.target_names), dtype=np.float32)
            input_condition[target_offset] = 1.
        else:
            input_condition = np.array(target_offset)

        mixture, target = [torch.from_numpy(output) for output in [mixture, target]]
        chunk_offset = start_pos // self.true_samples

        return mixture, mixture_idx, chunk_offset, torch.tensor(input_condition, dtype=torch.long), target_name, target

    def idx_to_track_offset(self, idx):

        for i, last_chunk in enumerate(self.acc_chunk_final_ids):
            if idx < last_chunk:
                if i != 0:
                    offset = (idx - self.acc_chunk_final_ids[i - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return i, offset

        return None, None

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode:
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_valid[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]


class MusdbUnmixedTrainSet(Dataset):

    def __init__(self, musdb_train, n_fft=2048, hop_length=1024, num_frame=64,
                 cache_mode=True, dev_mode=False):

        self.musdb_train = musdb_train
        self.window_length = hop_length * (num_frame - 1)

        self.lengths = [track.samples for track in self.musdb_train]
        self.source_names = ['vocals', 'drums', 'bass', 'other']

        self.num_tracks = len(self.musdb_train)

        # development mode
        if dev_mode:
            self.num_tracks = 1
            self.lengths = self.lengths[:1]

        self.cache_mode = cache_mode
        if cache_mode:
            self.cache = {}
            self.cache_dataset()

    def cache_dataset(self):
        print('cache audio files.')
        for idx in tqdm(range(self.num_tracks)):
            self.cache[idx] = {}
            for source in self.source_names:
                self.cache[idx][source] = self.musdb_train[idx].targets[source].audio.astype(np.float32)

    def __len__(self):
        return sum([length // self.window_length for length in self.lengths])

    def __getitem__(self, whatever):
        source_sample = [self.get_random_audio_sample(target) for target in self.source_names]
        return np.array(source_sample)

    def get_random_audio_sample(self, target_name):
        return self.get_audio_sample(random.randint(0, self.num_tracks - 1), target_name)

    def get_audio_sample(self, idx, target_name):
        length = self.lengths[idx] - self.window_length
        start_position = random.randint(0, length - 1)
        return self.get_audio(idx, target_name, start_position, self.window_length)

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode:
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_train[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]


class MusdbUnmixedEvalSet(Dataset):

    def __init__(self, musdb_eval, n_fft=2048, hop_length=1024, num_frame=64,
                 cache_mode=True, dev_mode=False):

        self.hop_length = hop_length
        self.musdb_test = musdb_eval
        self.window_length = hop_length * (num_frame - 1)
        self.trim_length = get_trim_length(self.hop_length)
        self.true_samples = self.window_length - 2 * self.trim_length

        self.lengths = [track.samples for track in self.musdb_test]
        self.source_names = ['vocals', 'drums', 'bass', 'other']  # == self.musdb_train.targets_names[:-2]

        self.num_tracks = len(self.musdb_test)

        # development mode
        if dev_mode:
            self.num_tracks = 1
            self.lengths = self.lengths[:1]

        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.acc_chunk_final_ids = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        self.cache_mode = cache_mode
        if cache_mode:
            self.cache = {}
            self.cache_dataset()

    def cache_dataset(self):
        print('cache audio files.')
        for idx in tqdm(range(self.num_tracks)):
            self.cache[idx] = {}
            self.cache[idx]['linear_mixture'] = self.musdb_test[idx].targets['linear_mixture'].audio.astype(
                np.float32)

    def __len__(self):
        return self.acc_chunk_final_ids[-1]

    def __getitem__(self, idx):

        track_idx, start_pos = self.idx_to_track_offset(idx)
        source_sample = [self.get_target_sample(track_idx, target, start_pos)
                         for target
                         in self.source_names]

        window_offset = start_pos // self.true_samples

        return np.array(source_sample), track_idx, window_offset

    def get_target_sample(self, track_idx, target, start_pos):
        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        target_length = self.lengths[track_idx]
        if start_pos + length > target_length:  # last
            right_padding_num += self.true_samples - (target_length - start_pos)
            length = None

        target = self.get_audio(track_idx, target, start_pos, length)

        target = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), target,
                                 np.zeros((right_padding_num, 2), dtype=np.float32)), 0)
        return target

    def idx_to_track_offset(self, acc_idx):

        for track_idx, last_chunk in enumerate(self.acc_chunk_final_ids):
            if acc_idx < last_chunk:
                if track_idx != 0:
                    offset = (acc_idx - self.acc_chunk_final_ids[track_idx - 1]) * self.true_samples
                else:
                    offset = acc_idx * self.true_samples
                return track_idx, offset

        return None, None

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode and target_name == 'linear_mixture':
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_test[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]


class SingleTrackSet(Dataset):

    def __init__(self, track, hop_length, num_frame):

        assert len(track.shape) == 2
        assert track.shape[1] == 2  # check stereo audio

        self.hop_length = hop_length
        self.window_length = hop_length * (num_frame - 1)
        self.trim_length = get_trim_length(self.hop_length)

        self.true_samples = self.window_length - 2 * self.trim_length

        self.lengths = [track.shape[0]]
        self.source_names = ['vocals', 'drums', 'bass', 'other']  # == self.musdb_train.targets_names[:-2]

        self.target_names = self.source_names

        self.num_tracks = 1

        import math
        num_chunks = [math.ceil(length / self.true_samples) for length in self.lengths]
        self.acc_chunk_final_ids = [sum(num_chunks[:i + 1]) for i in range(self.num_tracks)]

        self.cache_mode = True
        self.cache = {}
        self.cache[0] = {}
        self.cache[0]['linear_mixture'] = track.astype(np.float32)

    def __len__(self):
        return self.acc_chunk_final_ids[-1] #* len(self.target_names)

    def __getitem__(self, idx):

        target_offset = 0 #idx % len(self.target_names)

        target_name = self.target_names[target_offset]
        mixture_idx, start_pos = self.idx_to_track_offset(idx)

        length = self.true_samples
        left_padding_num = right_padding_num = self.trim_length
        if mixture_idx is None:
            raise StopIteration
        mixture_length = self.lengths[mixture_idx]
        if start_pos + length > mixture_length:  # last
            right_padding_num += self.true_samples - (mixture_length - start_pos)
            length = None

        mixture = self.get_audio(mixture_idx, 'linear_mixture', start_pos, length)

        mixture = np.concatenate((np.zeros((left_padding_num, 2), dtype=np.float32), mixture,
                                  np.zeros((right_padding_num, 2), dtype=np.float32)), 0)

        mixture = torch.from_numpy(mixture)

        return mixture

    def idx_to_track_offset(self, idx):

        for i, last_chunk in enumerate(self.acc_chunk_final_ids):
            if idx < last_chunk:
                if i != 0:
                    offset = (idx - self.acc_chunk_final_ids[i - 1]) * self.true_samples
                else:
                    offset = idx * self.true_samples
                return i, offset

        return None, None

    def get_audio(self, idx, target_name, pos=0, length=None):
        if self.cache_mode:
            track = self.cache[idx][target_name]
        else:
            track = self.musdb_valid[idx].targets[target_name].audio.astype(np.float32)
        return track[pos:pos + length] if length is not None else track[pos:]

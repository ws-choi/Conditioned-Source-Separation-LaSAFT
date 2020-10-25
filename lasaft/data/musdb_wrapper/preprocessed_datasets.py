from pathlib import Path
from warnings import warn

import soundfile

from lasaft.data.musdb_wrapper.datasets import *


class FiledMusdbTrainSet(MusdbTrainSet):

    def __init__(self, musdb_root='data/musdb18_wav/', is_wav=True, n_fft=2048, hop_length=1024, num_frame=64,
                 target_names=None, cache_mode=True,
                 dev_mode=False, one_hot_mode=False):

        musdb_loader = MusdbLoader(musdb_root, is_wav)
        musdb_root_parent = Path(musdb_root).parent
        self.train_root = musdb_root_parent.joinpath('librosa/train')

        if not musdb_root_parent.joinpath('librosa').is_dir():
            musdb_root_parent.joinpath('librosa').mkdir(parents=True, exist_ok=True)
        if not musdb_root_parent.joinpath('librosa/train').is_dir():
            musdb_root_parent.joinpath('librosa/train').mkdir(parents=True, exist_ok=True)
            warn('do not terminate now. if you have to do, please remove the librosa/train dir before re-initiating '
                 'LibrosaMusdbTrainSet ')

            for i, track in enumerate(tqdm(musdb_loader.musdb_train)):
                for target in ['linear_mixture', 'vocals', 'drums', 'bass', 'other']:
                    soundfile.write(file='{}/{}_{}.wav'.format(self.train_root, i, target),
                                    data=track.targets[target].audio.astype(np.float32),
                                    samplerate=track.rate
                                    )

        super().__init__(musdb_loader.musdb_train, n_fft, hop_length, num_frame, target_names, cache_mode, dev_mode,
                         one_hot_mode)

        # is_wav mode's lengths != not_is_wav mode's length.
        self.lengths = [self.get_audio(i, 'vocals').shape[0] for i in range(self.num_tracks)]

    def cache_dataset(self):
        warn('Librosa Musdbset does not need to be cached.')
        pass

    def get_audio(self, idx, target_name, pos=0, length=None):
        arg_dicts = {
            'file': self.train_root.joinpath('{}_{}.wav'.format(idx, target_name)),
            'start': pos,
            'dtype': 'float32'
        }

        if length is not None:
            arg_dicts['stop'] = pos + length

        return soundfile.read(**arg_dicts)[0]


class FiledMusdbTestSet(MusdbTestSet):

    def __init__(self, musdb_root='data/musdb18_wav/', is_wav=True, n_fft=2048, hop_length=1024, num_frame=64,
                 target_names=None, cache_mode=True,
                 dev_mode=False):

        musdb_loader = MusdbLoader(musdb_root, is_wav)
        musdb_root_parent = Path(musdb_root).parent
        self.test_root = musdb_root_parent.joinpath('librosa/test')

        if not musdb_root_parent.joinpath('librosa').is_dir():
            musdb_root_parent.joinpath('librosa').mkdir(parents=True, exist_ok=True)
        if not musdb_root_parent.joinpath('librosa/test').is_dir():
            musdb_root_parent.joinpath('librosa/test').mkdir(parents=True, exist_ok=True)
            warn('do not terminate now. if you have to do, please remove the librosa/test dir before re-initiating '
                 'LibrosaMusdbTestSet ')

            for i, track in enumerate(tqdm(musdb_loader.musdb_test)):
                for target in ['linear_mixture', 'vocals', 'drums', 'bass', 'other']:
                    soundfile.write(file='{}/{}_{}.wav'.format(self.test_root, i, target),
                                    data=track.targets[target].audio.astype(np.float32),
                                    samplerate=track.rate
                                    )

        super().__init__(musdb_loader.musdb_test, n_fft, hop_length, num_frame, target_names, cache_mode, dev_mode)
        # is_wav mode's lengths != not_is_wav mode's length.
        self.lengths = [self.get_audio(i, 'vocals').shape[0] for i in range(self.num_tracks)]

    def cache_dataset(self):
        warn('Librosa Musdbset does not need to be cached.')
        pass

    def get_audio(self, idx, target_name, pos=0, length=None):
        arg_dicts = {
            'file': self.test_root.joinpath('{}_{}.wav'.format(idx, target_name)),
            'start': pos,
            'dtype': 'float32'
        }

        if length is not None:
            arg_dicts['stop'] = pos + length

        return soundfile.read(**arg_dicts)[0]


class FiledMusdbValidSet(MusdbValidSet):

    def __init__(self, musdb_root='data/musdb18_wav/', is_wav=True, n_fft=2048, hop_length=1024, num_frame=64,
                 target_names=None, cache_mode=True,
                 dev_mode=False):

        musdb_loader = MusdbLoader(musdb_root, is_wav)
        musdb_root_parent = Path(musdb_root).parent
        self.valid_root = musdb_root_parent.joinpath('librosa/valid')

        if not musdb_root_parent.joinpath('librosa').is_dir():
            musdb_root_parent.joinpath('librosa').mkdir(parents=True, exist_ok=True)
        if not musdb_root_parent.joinpath('librosa/valid').is_dir():
            musdb_root_parent.joinpath('librosa/valid').mkdir(parents=True, exist_ok=True)
            warn('do not terminate now. if you have to do, please remove the librosa/valid dir before re-initiating '
                 'LibrosaMusdbValidSet ')

            for i, track in enumerate(tqdm(musdb_loader.musdb_valid)):
                for target in ['linear_mixture', 'vocals', 'drums', 'bass', 'other']:
                    soundfile.write(file='{}/{}_{}.wav'.format(self.valid_root, i, target),
                                    data=track.targets[target].audio.astype(np.float32),
                                    samplerate=track.rate
                                    )

        super().__init__(musdb_loader.musdb_valid, n_fft, hop_length, num_frame, target_names, cache_mode, dev_mode)

        # is_wav mode's lengths != not_is_wav mode's length.
        self.lengths = [self.get_audio(i, 'vocals').shape[0] for i in range(self.num_tracks)]

    def cache_dataset(self):
        warn('Librosa Musdbset does not need to be cached.')
        pass

    def get_audio(self, idx, target_name, pos=0, length=None):

        arg_dicts = {
            'file': self.valid_root.joinpath('{}_{}.wav'.format(idx, target_name)),
            'start': pos,
            'dtype': 'float32'
        }

        if length is not None:
            arg_dicts['stop'] = pos + length

        return soundfile.read(**arg_dicts)[0]
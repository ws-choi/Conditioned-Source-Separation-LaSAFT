from argparse import ArgumentParser

from torch.utils.data import DataLoader

from lasaft.data.musdb_wrapper.datasets import MusdbLoader, MusdbTrainSet, MusdbTestSet, \
    MusdbValidSet
from lasaft.data.musdb_wrapper.preprocessed_datasets import FiledMusdbTrainSet, FiledMusdbTestSet, FiledMusdbValidSet


class DataProvider():
    @staticmethod
    def add_data_provider_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--musdb_root', type=str, default='data/musdb18_wav/')
        parser.add_argument('--musdb_is_wav', type=bool, default=False)
        parser.add_argument('--filed_mode', type=bool, default=False)

        parser.add_argument('--target_names', type=str, default=None)

        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--pin_memory', type=bool, default=False)

        parser.add_argument('--dev_mode', type=bool, default=False)

        parser.add_argument('--one_hot_mode', type=bool, default=False)
        return parser

    def __init__(self, musdb_root, musdb_is_wav, filed_mode, target_names,
                 batch_size, pin_memory, num_workers, dev_mode, one_hot_mode, **kwargs):
        self.dev_mode = dev_mode
        self.musdb_root = musdb_root
        self.filed_mode = filed_mode
        self.target_names = target_names
        self.cache_mode = not filed_mode
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.is_wav = musdb_is_wav
        self.one_hot_mode = one_hot_mode

    def get_train_dataloader(self, n_fft, hop_length, num_frame):
        return get_train_dataloader(self.musdb_root, self.is_wav, self.filed_mode,
                                    n_fft, hop_length, num_frame,
                                    self.batch_size, self.num_workers, self.pin_memory,
                                    self.target_names,
                                    self.cache_mode, self.dev_mode)

    def get_valid_dataloader(self, n_fft, hop_length, num_frame):
        return get_valid_dataloader(self.musdb_root, self.is_wav, self.filed_mode,
                                    n_fft, hop_length, num_frame,
                                    self.batch_size, self.num_workers, self.pin_memory,
                                    self.target_names,
                                    self.cache_mode, self.dev_mode)

    def get_test_dataloader(self, n_fft, hop_length, num_frame):
        return get_test_dataloader(self.musdb_root, self.is_wav, self.filed_mode,
                                   n_fft, hop_length, num_frame,
                                   self.batch_size, self.num_workers, self.pin_memory,
                                   self.target_names,
                                   self.cache_mode, self.dev_mode)


def get_train_dataloader(musdb_root, is_wav, filed_mode,
                         n_fft, hop_length, num_frame,
                         batch_size=4, num_workers=1, pin_memory=True,
                         target_names=None,
                         cache_mode=True, dev_mode=False, one_hot_mode=False
                         ) -> DataLoader:
    if filed_mode:
        musdb_train = FiledMusdbTrainSet(musdb_root,
                                         is_wav,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         num_frame=num_frame,
                                         target_names=target_names,
                                         cache_mode=cache_mode,
                                         dev_mode=dev_mode,
                                         one_hot_mode=one_hot_mode)
    else:
        musdb_loader = MusdbLoader(musdb_root, is_wav)
        musdb_train = MusdbTrainSet(musdb_loader.musdb_train,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    num_frame=num_frame,
                                    target_names=target_names,
                                    cache_mode=cache_mode,
                                    dev_mode=dev_mode,
                                    one_hot_mode=one_hot_mode)

    return DataLoader(musdb_train, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


def get_valid_dataloader(musdb_root, is_wav, filed_mode,
                         n_fft, hop_length, num_frame,
                         batch_size=4, num_workers=1, pin_memory=True,
                         target_names=None,
                         cache_mode=True, dev_mode=False
                         ) -> DataLoader:
    if filed_mode:
        musdb_valid = FiledMusdbValidSet(musdb_root,
                                         is_wav,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         num_frame=num_frame,
                                         target_names=target_names,
                                         cache_mode=cache_mode,
                                         dev_mode=dev_mode)
    else:

        musdb_loader = MusdbLoader(musdb_root, is_wav)
        musdb_valid = MusdbValidSet(musdb_loader.musdb_valid,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    num_frame=num_frame,
                                    target_names=target_names,
                                    cache_mode=cache_mode,
                                    dev_mode=dev_mode)

    return DataLoader(musdb_valid, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


def get_test_dataloader(musdb_root, is_wav, filed_mode,
                        n_fft, hop_length, num_frame,
                        batch_size=4, num_workers=1, pin_memory=True,
                        target_names=None,
                        cache_mode=True, dev_mode=False
                        ) -> DataLoader:
    if filed_mode:
        musdb_test = FiledMusdbTestSet(musdb_root,
                                       is_wav,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       num_frame=num_frame,
                                       target_names=target_names,
                                       cache_mode=cache_mode,
                                       dev_mode=dev_mode)
    else:
        musdb_loader = MusdbLoader(musdb_root, is_wav)
        musdb_test = MusdbTestSet(musdb_loader.musdb_test,
                                  n_fft=n_fft,
                                  hop_length=hop_length,
                                  num_frame=num_frame,
                                  target_names=target_names,
                                  cache_mode=cache_mode,
                                  dev_mode=dev_mode)

    return DataLoader(musdb_test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
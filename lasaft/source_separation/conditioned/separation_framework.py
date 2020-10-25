from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from typing import List, Union

import numpy as np
import pydub
import pytorch_lightning as pl
import soundfile
import torch
import wandb
from pytorch_lightning import EvalResult
from pytorch_lightning.loggers import WandbLogger

from lasaft.source_separation.conditioned import loss_functions
from lasaft.utils import fourier
from lasaft.utils.fourier import get_trim_length
from lasaft.utils.functions import get_optimizer_by_name, get_estimation
from lasaft.utils.weight_initialization import init_weights_functional


class Conditional_Source_Separation(pl.LightningModule, metaclass=ABCMeta):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--optimizer', type=str, default='adam')

        return loss_functions.add_model_specific_args(parser)

    def __init__(self, n_fft, hop_length, num_frame, optimizer, lr, dev_mode):
        super(Conditional_Source_Separation, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.trim_length = get_trim_length(self.hop_length)
        self.n_trim_frames = self.trim_length // self.hop_length
        self.num_frame = num_frame

        self.lr = lr
        self.optimizer = optimizer

        self.target_names = ['vocals', 'drums', 'bass', 'other']
        self.dev_mode = dev_mode

    def configure_optimizers(self):
        optimizer = get_optimizer_by_name(self.optimizer)
        return optimizer(self.parameters(), lr=float(self.lr))

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def on_test_epoch_start(self):

        import os
        output_folder = 'museval_output'
        if os.path.exists(output_folder):
            os.rmdir(output_folder)
        os.mkdir(output_folder)

        self.valid_estimation_dict = None
        self.test_estimation_dict = {}

        self.musdb_test = self.test_dataloader().dataset
        num_tracks = self.musdb_test.num_tracks
        for target_name in self.target_names:
            self.test_estimation_dict[target_name] = {mixture_idx: {}
                                                      for mixture_idx
                                                      in range(num_tracks)}

    def test_step(self, batch, batch_idx):
        mixtures, mixture_ids, window_offsets, input_conditions, target_names = batch

        estimated_targets = self.separate(mixtures, input_conditions)[:, self.trim_length:-self.trim_length]

        for mixture, mixture_idx, window_offset, input_condition, target_name, estimated_target \
                in zip(mixtures, mixture_ids, window_offsets, input_conditions, target_names, estimated_targets):
            self.test_estimation_dict[target_name][mixture_idx.item()][
                window_offset.item()] = estimated_target.detach().cpu().numpy()

        return torch.zeros(0)

    def on_test_epoch_end(self):

        import museval
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')
        idx_list = [1] if self.dev_mode else range(self.musdb_test.num_tracks)

        for idx in idx_list:
            estimation = {}
            for target_name in self.target_names:
                estimation[target_name] = get_estimation(idx, target_name, self.test_estimation_dict)
                if estimation[target_name] is not None:
                    estimation[target_name] = estimation[target_name].astype(np.float32)

            # Real SDR
            if len(estimation) == len(self.target_names):
                track_length = self.musdb_test.musdb_test[idx].samples
                estimated_targets = [estimation[target_name][:track_length] for target_name in self.target_names]

                if track_length > estimated_targets[0].shape[0]:
                    raise NotImplementedError
                else:
                    estimated_targets_dict = {target_name: estimation[target_name][:track_length] for target_name in
                                              self.target_names}
                    track_score = museval.eval_mus_track(
                        self.musdb_test.musdb_test[idx],
                        estimated_targets_dict
                    )

                    score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                        ['target', 'metric'])['score'] \
                        .median().to_dict()

                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log(
                            {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})

                    else:
                        print(track_score)

                    results.add_track(track_score)

            if idx == 1 and isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({'result_sample_{}_{}'.format(self.current_epoch, target_name): [
                    wandb.Audio(estimation[target_name], caption='{}_{}'.format(idx, target_name), sample_rate=44100)]})

        if isinstance(self.logger, WandbLogger):

            result_dict = results.df.groupby(
                ['track', 'target', 'metric']
            )['score'].median().reset_index().groupby(
                ['target', 'metric']
            )['score'].median().to_dict()

            self.logger.experiment.log(
                {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
            )
        else:
            print(results)

    def export_mp3(self, idx, target_name):
        estimated = self.test_estimation_dict[target_name][idx]
        estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
        soundfile.write('tmp_output.wav', estimated, samplerate=44100)
        audio = pydub.AudioSegment.from_wav('tmp_output.wav')
        audio.export('{}_estimated/output_{}.mp3'.format(idx, target_name))

    @abstractmethod
    def forward(self, input_signal, input_condition) -> torch.Tensor:
        pass

    @abstractmethod
    def separate(self, input_signal, input_condition) -> torch.Tensor:
        pass

    @abstractmethod
    def init_weights(self):
        pass


class Spectrogram_based(Conditional_Source_Separation, metaclass=ABCMeta):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_fft', type=int, default=1024)
        parser.add_argument('--hop_length', type=int, default=256)
        parser.add_argument('--num_frame', type=int, default=128)
        parser.add_argument('--spec_type', type=str, default='magnitude')
        parser.add_argument('--spec_est_mode', type=str, default='masking')

        parser.add_argument('--train_loss', type=str, default='spec_mse')
        parser.add_argument('--val_loss', type=str, default='spec_mse')
        parser.add_argument('--unfreeze_stft_from', type=int, default=-1)  # -1 means never.

        return Conditional_Source_Separation.add_model_specific_args(parser)

    def __init__(self, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 conditional_spec2spec,
                 optimizer, lr, dev_mode,
                 train_loss, val_loss
                 ):
        super(Spectrogram_based, self).__init__(n_fft, hop_length, num_frame,
                                                optimizer, lr, dev_mode)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        assert spec_type in ['magnitude', 'complex']
        assert spec_est_mode in ['masking', 'mapping']
        self.magnitude_based = spec_type == 'magnitude'
        self.masking_based = spec_est_mode == 'masking'
        self.stft = fourier.multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)
        self.stft.freeze()

        self.conditional_spec2spec = conditional_spec2spec
        self.valid_estimation_dict = {}
        self.val_loss = val_loss
        self.train_loss = train_loss

        self.init_weights()

    def init_weights(self):
        init_weights_functional(self.conditional_spec2spec,
                                self.conditional_spec2spec.activation)

    def training_step(self, batch, batch_idx):
        mixture_signal, target_signal, condition = batch
        loss = self.train_loss(self, mixture_signal, condition, target_signal)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                   reduce_fx=torch.mean)
        return result

    # Validation Process
    def on_validation_epoch_start(self):
        for target_name in self.target_names:
            self.valid_estimation_dict[target_name] = {mixture_idx: {}
                                                       for mixture_idx
                                                       in range(14)}

    def validation_step(self, batch, batch_idx):

        mixtures, mixture_ids, window_offsets, input_conditions, target_names, targets = batch

        loss = self.val_loss(self, mixtures, input_conditions, targets)
        result = pl.EvalResult()
        result.log('raw_val_loss', loss, prog_bar=False, logger=False, reduce_fx=torch.mean)

        # Result Cache
        if 0 in mixture_ids.view(-1):
            estimated_targets = self.separate(mixtures, input_conditions)[:, self.trim_length:-self.trim_length]
            targets = targets[:, self.trim_length:-self.trim_length]

            for mixture, mixture_idx, window_offset, input_condition, target_name, estimated_target \
                    in zip(mixtures, mixture_ids, window_offsets, input_conditions, target_names, estimated_targets):

                if mixture_idx == 0:
                    self.valid_estimation_dict[target_name][mixture_idx.item()][
                        window_offset.item()] = estimated_target.detach().cpu().numpy()

        return result

    def validation_epoch_end(self, outputs: Union[EvalResult, List[EvalResult]]) -> EvalResult:

        for idx in [0]:
            estimation = {}
            for target_name in self.target_names:
                estimation[target_name] = get_estimation(idx, target_name, self.valid_estimation_dict)
                if estimation[target_name] is None:
                    continue
                if estimation[target_name] is not None:
                    estimation[target_name] = estimation[target_name].astype(np.float32)

                    if self.current_epoch > 10 and isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log({'result_sample_{}_{}'.format(self.current_epoch, target_name): [
                            wandb.Audio(estimation[target_name][44100 * 20:44100 * 40],
                                        caption='{}_{}'.format(idx, target_name),
                                        sample_rate=44100)]})

        reduced_loss = sum(outputs['raw_val_loss'] / len(outputs['raw_val_loss']))
        result = pl.EvalResult(early_stop_on=reduced_loss, checkpoint_on=reduced_loss)
        result.log('val_loss', reduced_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return result

    @abstractmethod
    def to_spec(self, input_signal) -> torch.Tensor:
        pass

    @abstractmethod
    def separate(self, input_signal, input_condition) -> torch.Tensor:
        pass

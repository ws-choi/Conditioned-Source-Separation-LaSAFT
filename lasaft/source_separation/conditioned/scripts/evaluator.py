from pathlib import Path

import hydra
import museval
import torch.cuda
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from lasaft.utils.functions import wandb_login
from lasaft.utils.instantiator import HydraInstantiator as HI


def eval(cfg: DictConfig):

    # MODEL
    if cfg['pretrained']['model']['spec_type'] != 'magnitude':
        cfg['pretrained']['model']['input_channels'] = 4

    model = HI.pretrained(cfg)

    if cfg['pretrained']['ckpt'] is None:
        raise ValueError("Required eval.ckpt is missing.")
    else:
        ckpt = Path(cfg['pretrained']['ckpt'])
        if not ckpt.exists():
            if 'ckpt_callback' in cfg['pretrained']:
                print('no cached checkpoint found.\nautomatic download via callback!')
                hydra.utils.call(cfg['pretrained']['ckpt_callback'])
                print('successfully downloaded the pretrained model.')
            else:
                raise FileNotFoundError

    # -- logger setting
    if 'logger' in cfg:
        logger = hydra.utils.instantiate(cfg['logger'])
        if len(logger) > 0:
            logger = logger['logger']
            if isinstance(logger, WandbLogger):
                wandb_login(key=cfg['wandb_api_key'])
                logger.watch(model, log='all')
                hparams= {}
                hparams["pretrained"] = cfg['pretrained']
                hparams['overlap_ratio'] = cfg['overlap_ratio']
                # save number of model parameters
                hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
                hparams["model/params_trainable"] = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                hparams["model/params_not_trainable"] = sum(
                    p.numel() for p in model.parameters() if not p.requires_grad
                )

                # send hparams to all loggers
                logger.log_hyperparams(hparams)

    else:
        logger = None

    # DATASET
    dp = HI.data_provider(cfg)

    _, test_data_loader = dp.get_test_dataset_and_loader()

    device = 'cpu'
    if int(cfg['gpus']) > 0:
        if torch.cuda.is_available():
            device = 'cuda'
    model = model.load_from_checkpoint(ckpt).to(device)

    ##
    overlap_ratio = cfg['overlap_ratio']
    batch_size = cfg['batch_size']

    dataset = test_data_loader.dataset.musdb_reference
    sources = ['vocals', 'drums', 'bass', 'other']

    results = museval.EvalStore(frames_agg='median', tracks_agg='median')

    for idx in range(len(dataset)):

        track = dataset[idx]
        estimation = {source: model.separate_track(track.audio, source, overlap_ratio, batch_size)
                      for source in sources}

        # Real SDR
        if len(estimation) == len(sources):
            track_length = dataset[idx].samples
            estimated_targets = [estimation[target_name][:track_length] for target_name in sources]
            if track_length > estimated_targets[0].shape[0]:
                raise NotImplementedError
            else:
                estimated_targets_dict = {target_name: estimation[target_name][:track_length] for target_name in
                                          sources}
                track_score = museval.eval_mus_track(
                    dataset[idx],
                    estimated_targets_dict
                )
                score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                    ['target', 'metric'])['score'] \
                    .median().to_dict()
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(
                        {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})
                else:
                    print(track_score)
                results.add_track(track_score)

    if isinstance(logger, WandbLogger):
        result_dict = results.df.groupby(
            ['track', 'target', 'metric']
        )['score'].median().reset_index().groupby(
            ['target', 'metric']
        )['score'].median().to_dict()
        logger.experiment.log(
            {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
        )
        logger.close()
    else:
        print(results)

    return None
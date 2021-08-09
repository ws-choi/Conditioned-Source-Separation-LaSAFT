from warnings import warn

import hydra
from packaging import version

import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from pytorch_lightning.loggers import WandbLogger

from lasaft.data.data_provider import DataProvider
from lasaft.source_separation.model_definition import get_class_by_name
from lasaft.utils.functions import mkdir_if_not_exists, wandb_login
from pathlib import Path
from pytorch_lightning import Trainer

from lasaft.utils.instantiator import HydraInstantiator as HI


def eval(cfg: DictConfig):
    if cfg['eval']['ckpt'] is None:
        raise ValueError("Required eval.ckpt is missing.")

    # MODEL
    if cfg['model']['spec_type'] != 'magnitude':
        cfg['model']['input_channels'] = 4

    model = HI.model(cfg)

    # Trainer Definition

    # -- checkpoint
    ckpt_path = Path(to_absolute_path(cfg['eval']['ckpt']))

    # -- logger setting
    if 'logger' in cfg:

        logger = hydra.utils.instantiate(cfg['logger'])
        model_name = model.spec2spec.__class__.__name__
        if len(logger) > 0:
            logger = logger['logger']
            if isinstance(logger, WandbLogger):
                wandb_login(key=cfg['wandb_api_key'])
                logger.watch(model, log='all')

    # Trainer
    if isinstance(cfg['trainer']['gpus'], int):
        if cfg['trainer']['gpus'] > 1:
            warn('# gpu and num_workers should be 1, Not implemented: museval for distributed parallel')
            cfg['trainer']['gpus'] = 1
            cfg['trainer']['distributed_backend'] = None

    # DATASET
    dp = HI.data_provider(cfg)

    cfg['trainer']['precision'] = 32
    trainer = HI.trainer(cfg, logger=logger, _convert_="partial")
    _, test_data_loader = dp.get_test_dataset_and_loader()
    model = model.load_from_checkpoint(ckpt_path)

    trainer.test(model, test_data_loader)

    return None
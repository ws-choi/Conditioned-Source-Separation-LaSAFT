from warnings import warn
from packaging import version

import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from pytorch_lightning.loggers import WandbLogger

from lasaft.data.data_provider import DataProvider
from lasaft.source_separation.model_definition import get_class_by_name
from lasaft.utils.functions import mkdir_if_not_exists
from pathlib import Path
from pytorch_lightning import Trainer

from lasaft.utils.instantiator import HydraInstantiator as HI


def eval(cfg: DictConfig):
    # requires
    if(cfg['eval']['run_id'] is None):
        raise ValueError("Required eval.run_id is missing.")
    if(cfg['eval']['epoch'] is None):
        raise ValueError("Required eval.epoch is missing.")

    # MODEL
    if cfg['model']['spec_type'] != 'magnitude':
        cfg['model']['input_channels'] = 4

    model = HI.model(cfg)

    # Trainer Definition

    # -- checkpoint
    ckpt_path = Path(to_absolute_path(cfg['eval']['ckpt_root_path']))
    ckpt_path = ckpt_path.joinpath(cfg['model']['_target_'].split('.')[-1])
    ckpt_path = ckpt_path.joinpath(cfg['eval']['run_id'])
    ckpt_path = '{}/{}'.format(str(ckpt_path), cfg['eval']['epoch'])

    # -- logger setting
    trainer_kwargs = {}
    model_name = model.spec2spec.__class__.__name__
    log = cfg['training']['log']
    if log == 'False':
        trainer_kwargs['logger'] = False
        trainer_kwargs['checkpoint_callback'] = False
        trainer_kwargs['early_stop_callback'] = False
    elif log == 'wandb':
        trainer_kwargs['logger'] = WandbLogger(project='lasaft_exp', tags=[model_name], offline=False,
                                     name=cfg['eval']['run_id'] + '_eval_' + cfg['eval']['epoch'].replace('=','_'))
        trainer_kwargs['logger'].log_hyperparams(model.hparams)
        trainer_kwargs['logger'].watch(model, log='all')
    elif log == 'tensorboard':
        raise NotImplementedError
    else:
        trainer_kwargs['logger'] = True  # default
        default_save_path = 'etc/lightning_logs'
        mkdir_if_not_exists(default_save_path)
    
    if(version.parse(pl.__version__) > version.parse('1.3.0')):
        if log == 'False':
            trainer_kwargs['callbacks'] = [trainer_kwargs['checkpoint_callback'], trainer_kwargs['early_stop_callback']]
            del(trainer_kwargs['checkpoint_callback'])
            del(trainer_kwargs['early_stop_callback'])

    # Trainer
    if isinstance(cfg['trainer']['gpus'], int):
        if cfg['trainer']['gpus'] > 1:
            warn('# gpu and num_workers should be 1, Not implemented: museval for distributed parallel')
            cfg['trainer']['gpus'] = 1
            cfg['trainer']['distributed_backend'] = None

    # DATASET
    dp = HI.data_provider(cfg)

    cfg['trainer']['precision'] = 32
    trainer = HI.trainer(cfg, **trainer_kwargs)
    _, test_data_loader = dp.get_test_dataset_and_loader()
    model = model.load_from_checkpoint(ckpt_path)

    trainer.test(model, test_data_loader)

    return None
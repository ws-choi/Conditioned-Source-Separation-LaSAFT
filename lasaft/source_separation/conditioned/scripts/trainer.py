from warnings import warn

import hydra.utils
from packaging import version

from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path
from pytorch_lightning import seed_everything

from lasaft.utils.functions import mkdir_if_not_exists, wandb_login, log_hyperparameters

from lasaft.utils.instantiator import HydraInstantiator as HI


def train(cfg: DictConfig):
    if cfg['model']['spec_type'] != 'magnitude':
        cfg['model']['input_channels'] = 4

    if cfg['trainer']['resume_from_checkpoint'] is None:
        if cfg['training']['seed'] is not None:
            seed_everything(cfg['training']['seed'])

    # model = framework(**args)
    model = HI.model(cfg)

    if cfg['model']['last_activation'] != 'identity' and cfg['model']['spec_est_mode'] != 'masking':
        warn('Please check if you really want to use a mapping-based spectrogram estimation method '
            'with a final activation function. ')
    ##########################################################

    # -- checkpoint
    ckpt_path = Path(to_absolute_path(cfg['training']['ckpt_root_path']))
    mkdir_if_not_exists(ckpt_path)
    ckpt_path = ckpt_path.joinpath(cfg['model']['_target_'].split('.')[-1])
    mkdir_if_not_exists(ckpt_path)
    run_id = cfg['training']['run_id']
    ckpt_path = ckpt_path.joinpath(run_id)
    mkdir_if_not_exists(ckpt_path)
    save_top_k = cfg['training']['save_top_k']

    checkpoint_kwargs = {
        'filepath': ckpt_path,
        'save_top_k': save_top_k,
        'verbose': False,
        'monitor': 'val_loss',
        'save_last': False,
        'save_weights_only': cfg['training']['save_weights_only']
    }

    if version.parse(pl.__version__) > version.parse('1.3.0'):
        checkpoint_kwargs['dirpath'] = ckpt_path
        del(checkpoint_kwargs['filepath'])

    checkpoint_callback = ModelCheckpoint(
        **checkpoint_kwargs
    )

    
    # -- early stop
    patience = cfg['training']['patience']
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=patience,
        verbose=False
    )

    trainer_kwargs = {
        'checkpoint_callback': checkpoint_callback,
        'callbacks': [early_stop_callback]
    }

    if version.parse(pl.__version__) > version.parse('1.3.0'):
        trainer_kwargs['callbacks'] = [checkpoint_callback, early_stop_callback]
        if 'checkpoint_callback' in trainer_kwargs.keys():
            del(trainer_kwargs['checkpoint_callback'])
        if 'early_stop_callback' in trainer_kwargs.keys():
            del(trainer_kwargs['early_stop_callback'])

    if cfg['trainer']['resume_from_checkpoint'] is not None:
        run_id = run_id + "_resume_" + cfg['trainer']['resume_from_checkpoint']
        cfg['trainer']['resume_from_checkpoint'] = Path(
            cfg['training']['ckpt_root_path']).joinpath(
            cfg['model']['_target_'].split('.')[-1]).joinpath(
            cfg['training']['run_id']).joinpath(
            cfg['trainer']['resume_from_checkpoint']
        )
        cfg['trainer']['resume_from_checkpoint'] = str(cfg['trainer']['resume_from_checkpoint'])

    model_name = model.spec2spec.__class__.__name__

    # -- logger setting
    if 'logger' in cfg:

        logger = hydra.utils.instantiate(cfg['logger'])
        if len(logger) > 0:
            logger = logger['logger']
            trainer_kwargs['logger'] = logger
            if isinstance(logger, WandbLogger):
                wandb_login(key=cfg['wandb_api_key'])
                logger.watch(model, log='all')

    # Trainer
    trainer = HI.trainer(cfg, **trainer_kwargs)
    dp = HI.data_provider(cfg)

    train_dataset, training_dataloader = dp.get_training_dataset_and_loader()
    valid_dataset, validation_dataloader = dp.get_validation_dataset_and_loader()

    if cfg['trainer']['auto_lr_find']:
        lr_find = trainer.tuner.lr_find(model,
                                        training_dataloader,
                                        validation_dataloader,
                                        early_stop_threshold=None,
                                        min_lr=1e-5)

        print(f"Found lr: {lr_find.suggestion()}")
        return None

    if cfg['trainer']['resume_from_checkpoint'] is not None:
        print('resume from the checkpoint')

    log_hyperparameters(
        config=cfg,
        model=model,
        trainer=trainer
    )

    trainer.fit(model, training_dataloader, validation_dataloader)

    return None

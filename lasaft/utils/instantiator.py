from pytorch_lightning import Trainer
from lasaft.data.data_provider import DataProvider
import hydra
from omegaconf import DictConfig


class HydraInstantiator():
    @classmethod
    def model(cls, cfg: DictConfig):
        if 'training' in cfg.keys():
            kwargs = {
                'train_loss': cls.train_loss(cfg),
                'val_loss': cls.val_loss(cfg)
            }
        model_cfg = {**cfg['model'], **cls.remove_target(cfg['dataset']), **cls.remove_target(cfg['training'])}
        return cls.instantiate(model_cfg, **kwargs)

    @classmethod
    def train_loss(cls, cfg: DictConfig):
        loss_cfg = cfg['training']['train_loss']
        if ('window_length' in loss_cfg.keys()):
            loss_cfg['window_length'] = cfg['dataset']['hop_length'] * (cfg['dataset']['num_frame'] - 1)
        return cls.instantiate(loss_cfg)

    @classmethod
    def val_loss(cls, cfg: DictConfig):
        loss_cfg = cfg['training']['val_loss']
        if ('window_length' in loss_cfg.keys()):
            loss_cfg['window_length'] = cfg['dataset']['hop_length'] * (cfg['dataset']['num_frame'] - 1)
        return cls.instantiate(loss_cfg)

    @classmethod
    def trainer(cls, cfg: DictConfig, **kwargs) -> Trainer:
        return cls.instantiate(cfg['trainer'], **kwargs)

    @classmethod
    def data_provider(cls, cfg: DictConfig) -> DataProvider:
        return cls.instantiate(cfg['dataset'])

    @classmethod
    def instantiate(cls, *args, **kwargs):
        return hydra.utils.instantiate(*args, **kwargs)

    @classmethod
    def remove_target(cls, cfg: DictConfig):
        if '_target_' in cfg.keys():
            _cfg = dict(cfg)
            del (_cfg['_target_'])
            return _cfg
        return cfg

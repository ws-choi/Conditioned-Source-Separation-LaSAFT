import os
from typing import Sequence
from warnings import warn
import numpy as np
import rich
import rich.tree
import rich.syntax
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_activation_by_name(activation):
    if activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "softmax":
        return nn.Softmax
    elif activation == "identity":
        return nn.Identity
    else:
        return None


def get_optimizer_by_name(optimizer):
    if optimizer == "adam":
        return torch.optim.Adam
    elif optimizer == "adagrad":
        return torch.optim.Adagrad
    elif optimizer == "sgd":
        return torch.optim.SGD
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop
    else:
        return torch.optim.Adam


def string_to_tuple(kernel_size):
    kernel_size_ = kernel_size.strip().replace('(', '').replace(')', '').split(',')
    kernel_size_ = [int(kernel) for kernel in kernel_size_]
    return kernel_size_


def string_to_list(int_list):
    int_list_ = int_list.strip().replace('[', '').replace(']', '').split(',')
    int_list_ = [int(v) for v in int_list_]
    return int_list_


def mkdir_if_not_exists(default_save_path):
    if not os.path.exists(default_save_path):
        os.makedirs(default_save_path)


def get_estimation(idx, target_name, estimation_dict):
    estimated = estimation_dict[target_name][idx]
    if len(estimated) == 0:
        warn('TODO: zero estimation, caused by ddp')
        return None
    estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
    return estimated


def flat_word_set(word_set):
    return [subword for word in word_set for subword in word.split(' ')]


def wandb_login(key):
    wandb.login(key=key)


def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "model",
                "datamodule",
                "callbacks",
                "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["dataset"] = config["dataset"]
    hparams["training"] = config["training"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above

    def empty(*args, **kwargs):
        pass

    trainer.logger.log_hyperparams = empty

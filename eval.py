from argparse import ArgumentParser
from datetime import datetime

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.utilities.distributed import rank_zero_info

from lasaft.source_separation.conditioned.scripts import evaluator as evaluator
from lasaft.utils.functions import mkdir_if_not_exists

dotenv.load_dotenv(override=True)


def main(cfg: DictConfig):
    # Load config
    rank_zero_info(OmegaConf.to_yaml(cfg))

    evaluator.eval(cfg)


@hydra.main(config_path="conf", config_name="eval")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == '__main__':
    hydra_entry()

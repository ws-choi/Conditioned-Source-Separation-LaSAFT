import os
from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import Trainer

from lasaft.data.musdb_wrapper import DataProvider
from lasaft.source_separation.conditioned.scripts import evaluator, trainer
from lasaft.source_separation.model_definition import get_class_by_name
from lasaft.utils.functions import mkdir_if_not_exists


def main(args):
    pass


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    temp_args, _ = parser.parse_known_args()

    # Params
    # Model
    model = get_class_by_name('conditioned_separation', temp_args.model)
    parser = model.add_model_specific_args(parser)

    # Dataset
    parser = DataProvider.add_data_provider_args(parser)

    # Environment Setup
    mkdir_if_not_exists('etc')
    mkdir_if_not_exists('etc/checkpoints')
    parser.add_argument('--ckpt_root_path', type=str, default='etc/checkpoints')
    parser.add_argument('--log', type=str, default=True)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--save_weights_only', type=bool, default=False)

    # Training
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser = Trainer.add_argparse_args(parser)
    ######################################################################

    args = vars(parser.parse_args())




    ###########
    # Trainer #
    ###########

    args['gpus'] = str(args['gpus'])
    args['batch_size'] = 8
    #args['auto_select_gpus'] = True
    args['pin_memory'] = True

    args['save_top_k'] = 3
    args['save_weights_only'] = True
    if args['run_id'] is None:
        args['run_id'] = 'exp_' + str(datetime.today().strftime("%Y%m%d_%H%M"))

    run_id = args['run_id']
    args['kernel_size_t'] = 3
    args['kernel_size_f'] = 3
    args['num_workers'] = 8
    args['bn_factor'] = 16
    args['min_epochs'] = 80

    if args['dev_mode'] is not None:
        if args['dev_mode'] == True or args['dev_mode'] == 'True' or args['dev_mode'] == 'true':
            args['num_workers'] = 0

    # Training
    args['patience'] = 50
    trainer.train(args)

    # Evaluation
    if args['auto_lr_find'] == 'lr':
        exit()

    args['precision'] = 32
    dir = os.path.join(args['ckpt_root_path'], args['model'], run_id)
    for _, _, checkpoints in os.walk(dir):
        for filename in checkpoints:
            args['epoch'] = filename
            evaluator.eval(args)

import inspect
from warnings import warn

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything

from lasaft.data.musdb_wrapper import DataProvider
from lasaft.source_separation.model_definition import get_class_by_name
from lasaft.utils.functions import mkdir_if_not_exists


def train(param):
    if not isinstance(param, dict):
        args = vars(param)
    else:
        args = param

    framework = get_class_by_name('conditioned_separation', args['model'])
    if args['spec_type'] != 'magnitude':
        args['input_channels'] = 4

    if args['resume_from_checkpoint'] is None:
        if args['seed'] is not None:
            seed_everything(args['seed'])

    model = framework(**args)

    if args['last_activation'] != 'identity' and args['spec_est_mode'] != 'masking':
        warn('Please check if you really want to use a mapping-based spectrogram estimation method '
             'with a final activation function. ')
    ##########################################################

    # -- checkpoint
    ckpt_path = Path(args['ckpt_root_path'])
    mkdir_if_not_exists(ckpt_path)
    ckpt_path = ckpt_path.joinpath(args['model'])
    mkdir_if_not_exists(ckpt_path)
    run_id = args['run_id']
    ckpt_path = ckpt_path.joinpath(run_id)
    mkdir_if_not_exists(ckpt_path)
    save_top_k = args['save_top_k']

    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=save_top_k,
        verbose=False,
        monitor='val_loss',
        save_last=False,
        save_weights_only=args['save_weights_only']
    )
    args['checkpoint_callback'] = checkpoint_callback

    # -- early stop
    patience = args['patience']
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=patience,
        verbose=False
    )
    args['early_stop_callback'] = early_stop_callback

    if args['resume_from_checkpoint'] is not None:
        run_id = run_id + "_resume_" + args['resume_from_checkpoint']
        args['resume_from_checkpoint'] = Path(
            args['ckpt_root_path']).joinpath(
            args['model']).joinpath(
            args['run_id']).joinpath(
            args['resume_from_checkpoint']
        )
        args['resume_from_checkpoint'] = str(args['resume_from_checkpoint'])

    # -- logger setting
    log = args['log']
    if log == 'False':
        args['logger'] = False
    elif log == 'wandb':
        args['logger'] = WandbLogger(project='lasaft', tags=args['model'], offline=False, id=run_id)
        args['logger'].log_hyperparams(model.hparams)
        args['logger'].watch(model, log='all')
    elif log == 'tensorboard':
        raise NotImplementedError
    else:
        args['logger'] = True  # default
        default_save_path = 'etc/lightning_logs'
        mkdir_if_not_exists(default_save_path)

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)

    # DATASET
    ##########################################################
    data_provider = DataProvider(**args)
    ##########################################################
    # Trainer Definition

    # Trainer
    trainer = Trainer(**trainer_kwargs)
    n_fft, hop_length, num_frame = args['n_fft'], args['hop_length'], args['num_frame']
    train_data_loader = data_provider.get_train_dataloader(n_fft, hop_length, num_frame)
    valid_data_loader = data_provider.get_valid_dataloader(n_fft, hop_length, num_frame)

    for key in sorted(args.keys()):
        print('{}:{}'.format(key, args[key]))

    if args['auto_lr_find']:
        lr_finder = trainer.lr_find(model, train_data_loader, valid_data_loader, early_stop_threshold=None)
        print(lr_finder.results)
        # torch.save(lr_finder.results, 'lr_result.cache')
        new_lr = lr_finder.suggestion()
        print('new_lr_suggestion:', new_lr)
        return 0

    print(model)

    trainer.fit(model, train_data_loader, valid_data_loader)

    return None

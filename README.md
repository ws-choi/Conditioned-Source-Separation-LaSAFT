# LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lasaft-latent-source-attentive-frequency/music-source-separation-on-musdb18)](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=lasaft-latent-source-attentive-frequency)

Check separated samples on this [demo page](https://lasaft.github.io/)!

An official Pytorch Implementation of the paper "LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation" (accepted to ICASSP 2021. ([slide](https://ws-choi.github.io/Conditioned-Source-Separation-LaSAFT/)))

## Demonstration: A Pretrained Model

[![demo](https://i.imgur.com/8hPZJIY.png)](https://youtu.be/buWnt89kVzs?t=8) 

Interactive Demonstration - [Colab Link](https://colab.research.google.com/github/ws-choi/Conditioned-Source-Separation-LaSAFT/blob/main/colab_demo/LaSAFT_with_GPoCM_(large)_Stella_Jang_Example.ipynb)
  - including how to download and use the pretrained model

## Quickstart: How to use Pretrained Models

### 1. [Install](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT#1-installation) LaSAFT.
### 2. Load a Pretrained Model.
```python
from lasaft.pretrained import PreTrainedLaSAFTNet
model = PreTrainedLaSAFTNet(model_name='lasaft_large_2020')
```
### 3. call ```model.separate_track``` !
```python
# audio should be an np(numpy) array of an stereo audio track
# with dtype of float32
# shape must be (T, 2)

vocals = model.separate_track(audio, 'vocals') 
drums = model.separate_track(audio, 'drums') 
bass = model.separate_track(audio, 'bass') 
other = model.separate_track(audio, 'other')
```


## Step-by-Step Tutorials

### 1. Installation

We highly recommend you to **install environments using scripts below**, even if we uploaded the [pip-requirements.txt](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/blob/main/requirements.txt)


```
conda env create -f lasaft_env_gpu.yaml -n lasaft
conda activate lasaft
pip install -r requirements.txt
```

### 2. Dataset: Musdb18

LaSAFT was trained/evaluated on the [Musdb18](https://sigsep.github.io/datasets/musdb.html) dataset.

We provide [wrapper packages](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/tree/main/lasaft/data/musdb_wrapper.py) to efficiently load musdb18 tracks as pytorch tensors.

You can also find [useful scripts](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/tree/main/lasaft/data) for downloading and preprocessing Musdb18 (or its 7s-samples).

### 3. Training

- Below is an example to train a U-Net with LaSAFT+GPoCM, whose hyper-parameters are set as default.
    ```shell script
    python train.py trainer.gpus=1 dataset.batch_size=6
    ```
- train.py includes training scripts for several models described in the paper [1].
    - It provides several options, including pytorch-lightning parameters
    - `model/conditioned_separation: CUNET_TFC_FiLM, CUNET_TFC_FiLM_LaSAFT, CUNET_TFC_FiLM_TDF, CUNET_TFC_GPoCM, CUNET_TFC_GPoCM_LaSAFT, CUNET_TFC_GPoCM_LightSAFT, CUNET_TFC_GPoCM_TDF, default, lasaft_net, lightsaft_net`

- An example of Training/Validation loss (see [wandb report](https://wandb.ai/wschoi/lasaft/reports/LaSAFT-larger-model-training-eval-example--Vmlldzo1NDczMTM) for more details)
![](https://raw.githubusercontent.com/ws-choi/Conditioned-Source-Separation-LaSAFT/main/assets/training_example.png)

#### Examples

- Table 1 in [1]

    - FiLM CUNet
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_FiLM dataset.batch_size=8 trainer.precision=16 trainer.gpus=1 training.patience=10 training.lr=0.001 logger=wandb
        ```
    - FiLM CUNet + TDF
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_FiLM_TDF dataset.batch_size=8 trainer.precision=16 trainer.gpus=1 training.patience=10 training.lr=0.001 logger=wandb
        ```
    - FiLM CUNet + LaSAFT
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_FiLM_LaSAFT dataset.batch_size=8 trainer.precision=16 trainer.gpus=1 training.patience=10 training.lr=0.001 logger=wandb
        ```
    
    - GPoCM CUNet
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_GPoCM dataset.batch_size=8 trainer.precision=16 trainer.gpus=1 training.patience=10 training.lr=0.001 logger=wandb
        ```
    - GPoCM CUNet + TDF
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_GPoCM_TDF dataset.batch_size=8 trainer.precision=16 trainer.gpus=1 training.patience=10 training.lr=0.001 logger=wandb
        ```
    - GPoCM CUNet + LaSAFT (* proposed model) 
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_GPoCM_LaSAFT dataset.batch_size=8 trainer.precision=16 trainer.gpus=1 training.patience=10 training.lr=0.001 logger=wandb
        ```
    - GPoCM CUNet + LightSAFT
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_GPoCM_LightSAFT dataset.batch_size=8 trainer.precision=16 trainer.gpus=1 training.patience=10 training.lr=0.001 logger=wandb
        ```

- Table 2 in [1] (Multi-GPUs Version)

    - GPoCM CUNet + LaSAFT (* proposed model) 
        ```shell script
        python train.py model=conditioned_separation/CUNET_TFC_GPoCM_LaSAFT trainer=four_2080tis model.n_blocks=9 model.num_tdfs=6 model.embedding_dim=64 dataset.n_fft=4096 dataset.hop_length=1024 trainer.deterministic=True training.patience=10 training.lr=0.001 training.auto_lr_schedule=True logger=wandb training.run_id=lasaft-2020
        ```

#### tunable hyperparameters

```
train is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

dataset: default
eval: default
model/conditioned_separation: CUNET_TFC_FiLM, CUNET_TFC_FiLM_LaSAFT, CUNET_TFC_FiLM_TDF, CUNET_TFC_GPoCM, CUNET_TFC_GPoCM_LaSAFT, CUNET_TFC_GPoCM_LightSAFT, CUNET_TFC_GPoCM_TDF, base, film, gpocm, lasaft_net, lightsaft_net, tfc
trainer: default
training: default
training/train_loss: distortion, dsr, ldsr, ncs, ncs_44100, ndsr, ndsr_44100, nlcs, raw_and_spec, raw_l1, raw_l2, raw_mse, sdr, sdr_like, spec_l1, spec_l2, spec_mse
training/val_loss: distortion, dsr, ldsr, ncs, ncs_44100, ndsr, ndsr_44100, nlcs, raw_and_spec, raw_l1, raw_l2, raw_mse, sdr, sdr_like, spec_l1, spec_l2, spec_mse


== Config ==
Override anything in the config (foo.bar=value)

trainer:
  _target_: pytorch_lightning.Trainer
  checkpoint_callback: true
  callbacks: null
  default_root_dir: null
  gradient_clip_val: 0.0
  process_position: 0
  num_nodes: 1
  num_processes: 1
  gpus: null
  auto_select_gpus: false
  tpu_cores: null
  log_gpu_memory: null
  progress_bar_refresh_rate: 1
  overfit_batches: 0.0
  track_grad_norm: -1
  check_val_every_n_epoch: 1
  fast_dev_run: false
  accumulate_grad_batches: 1
  max_epochs: 1
  min_epochs: 1
  max_steps: null
  min_steps: null
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  limit_test_batches: 1.0
  val_check_interval: 1.0
  flush_logs_every_n_steps: 100
  log_every_n_steps: 50
  accelerator: ddp
  sync_batchnorm: false
  precision: 16
  weights_summary: top
  weights_save_path: null
  num_sanity_val_steps: 2
  truncated_bptt_steps: null
  resume_from_checkpoint: null
  profiler: null
  benchmark: false
  deterministic: false
  reload_dataloaders_every_epoch: false
  auto_lr_find: false
  replace_sampler_ddp: true
  terminate_on_nan: false
  auto_scale_batch_size: false
  prepare_data_per_node: true
  amp_backend: native
  amp_level: O2
  move_metrics_to_cpu: false
dataset:
  _target_: lasaft.data.data_provider.DataProvider
  musdb_root: etc/musdb18_dev_wav
  batch_size: 8
  num_workers: 0
  pin_memory: true
  num_frame: 128
  hop_length: 1024
  n_fft: 2048
model:
  spec_type: complex
  spec_est_mode: mapping
  n_blocks: 7
  input_channels: 4
  internal_channels: 24
  first_conv_activation: relu
  last_activation: identity
  t_down_layers: null
  f_down_layers: null
  control_vector_type: embedding
  control_input_dim: 4
  embedding_dim: 32
  condition_to: decoder
  unfreeze_stft_from: -1
  control_n_layer: 4
  control_type: dense
  pocm_type: matmul
  pocm_norm: batch_norm
  _target_: lasaft.source_separation.conditioned.cunet.models.dcun_tfc_gpocm_lasaft.DCUN_TFC_GPoCM_LaSAFT_Framework
  n_internal_layers: 5
  kernel_size_t: 3
  kernel_size_f: 3
  bn_factor: 16
  min_bn_units: 16
  tfc_tdf_bias: false
  tfc_tdf_activation: relu
  num_tdfs: 6
  dk: 32
training:
  train_loss:
    _target_: lasaft.source_separation.conditioned.loss_functions.Conditional_Spectrogram_Loss
    mode: mse
  val_loss:
    _target_: lasaft.source_separation.conditioned.loss_functions.Conditional_RAW_Loss
    mode: l1
  ckpt_root_path: etc/checkpoints
  log: true
  run_id: ${now:%Y-%m-%d}/${now:%H-%M-%S}
  save_weights_only: false
  optimizer: adam
  lr: 0.001
  auto_lr_schedule: true
  save_top_k: 5
  patience: 10
  seed: 2020
```

### 4. Evaluation

```
python eval.py eval.run_id=[RUN_ID] 'eval.epoch=[EPOCH_CKPT]'
```

- example
    - if you trained you model with the following script
        ```
        python train.py model=conditioned_separation/CUNET_TFC_FiLM dataset.batch_size=8 trainer=single_gpu training.lr=0.001 logger=wandb
        ```
    - suppose that its run_id = 2021-08-09/17-16-26
    - then run eval scripts like below.
        ```
        python eval.py model=conditioned_separation/CUNET_TFC_FiLM eval.ckpt='outputs/2021-08-09/17-16-26/checkpoints/epoch\=00.ckpt' trainer.gpus=1
        ```

## You can cite this paper as follows:

```bibtex
@INPROCEEDINGS{9413896,
  author={Choi, Woosung and Kim, Minseok and Chung, Jaehwa and Jung, Soonyoung},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Lasaft: Latent Source Attentive Frequency Transformation For Conditioned Source Separation}, 
  year={2021},
  volume={},
  number={},
  pages={171-175},
  doi={10.1109/ICASSP39728.2021.9413896}}
```


### LaSAFT: Latent Source Attentive Frequency Transformation

![](https://imgur.com/vQNgttJ.png)

### GPoCM: Gated Point-wise Convolutional Modulation

![](https://imgur.com/9A4otVA.png)


## Reference

[1]  Woosung Choi,  Minseok Kim,  Jaehwa Chung, and Soonyoung Jung, “LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation.,” arXiv preprint arXiv:2010.11631 (2020).  

## Other Links

- This code borrows heavily from [ISMIR2020_U_Nets_SVS](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS/) repository. Many thanks.

- The original Conditional U-Net 
    - [tensorflow version](https://github.com/gabolsgabs/cunet) (official)
    - [pytorch version](https://github.com/ws-choi/Conditioned-U-Net-pytorch)
    - [run over CPU in a Docker container - by loretoparisi](https://github.com/loretoparisi/Conditioned-Source-Separation-LaSAFT)

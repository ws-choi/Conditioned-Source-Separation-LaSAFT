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

(Optional)
```
conda create -n lasaft
conda activate lasaft
```

(Install)
```
conda install pytorch=1.7.1 cudatoolkit=11.0 -c pytorch
conda install -c conda-forge ffmpeg librosa=0.6
conda install -c anaconda jupyter
pip install musdb==0.3.1 museval==0.3.0 pytorch_lightning==1.1.6 wandb==0.10.15 pydub==0.24.1 wget
```

### 2. Dataset: Musdb18

LaSAFT was trained/evaluated on the [Musdb18](https://sigsep.github.io/datasets/musdb.html) dataset.

We provide [wrapper packages](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/tree/main/lasaft/data/musdb_wrapper.py) to efficiently load musdb18 tracks as pytorch tensors.

You can also find [useful scripts](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/tree/main/lasaft/data) for downloading and preprocessing Musdb18 (or its 7s-samples).

### 3. Training

- Below is an example to train a U-Net with LaSAFT+GPoCM, whose hyper-parameters are set as default.
    ```shell script
    python main.py --problem_name conditioned_separation --mode train --run_id lasaft_net --musdb_root etc/musdb18_dev_wav --gpus 1 --precision 16 --batch_size 6 --num_workers 0 --pin_memory True --save_top_k 3 --save_weights_only True --patience 10 --lr 0.001 --model CUNET_TFC_GPoCM_LaSAFT
    ```
- main.py includes training scripts for several models described in the paper [1].
    - It provides several options, including pytorch-lightning parameters

#### Examples

- Table 1 in [1]

    - FiLM CUNet
        ```shell script
        python main.py --problem_name conditioned_separation --mode train --musdb_root etc/musdb18_dev_wav --gpus 1 --precision 16 --batch_size 8 --num_workers 0 --pin_memory True --save_top_k 3 --save_weights_only True --patience 10 --lr 0.001 --deterministic --model CUNET_TFC_FiLM --log False
        ```
    - FiLM CUNet + TDF
        ```shell script
        python main.py --problem_name conditioned_separation --mode train --musdb_root etc/musdb18_dev_wav --gpus 1 --precision 16 --batch_size 8 --num_workers 0 --pin_memory True --save_top_k 3 --save_weights_only True --patience 10 --lr 0.001 --deterministic --model CUNET_TFC_FiLM_TDF --log False
        ```
    - FiLM CUNet + LaSAFT
        ```shell script
        python main.py --problem_name conditioned_separation --mode train --musdb_root etc/musdb18_dev_wav --gpus 1 --precision 16 --batch_size 8 --num_workers 0 --pin_memory True --save_top_k 3 --save_weights_only True --patience 10 --lr 0.001 --deterministic --model CUNET_TFC_FiLM_LaSAFT --log False
        ```
    
    - GPoCM CUNet
        ```shell script
        python main.py --problem_name conditioned_separation --mode train --musdb_root etc/musdb18_dev_wav --gpus 1 --precision 16 --batch_size 8 --num_workers 0 --pin_memory True --save_top_k 3 --save_weights_only True --patience 10 --lr 0.001 --deterministic --model CUNET_TFC_GPoCM --log False
        ```
    - GPoCM CUNet + TDF
        ```shell script
        python main.py --problem_name conditioned_separation --mode train --musdb_root etc/musdb18_dev_wav --gpus 1 --precision 16 --batch_size 8 --num_workers 0 --pin_memory True --save_top_k 3 --save_weights_only True --patience 10 --lr 0.001 --deterministic --model CUNET_TFC_GPoCM_TDF --log False
        ```
    - GPoCM CUNet + LaSAFT (* proposed model) 
        ```shell script
        python main.py --problem_name conditioned_separation --mode train --musdb_root etc/musdb18_dev_wav --gpus 1 --precision 16 --batch_size 8 --num_workers 0 --pin_memory True --save_top_k 3 --save_weights_only True --patience 10 --lr 0.001 --deterministic --model CUNET_TFC_GPoCM_LaSAFT --log False
        ```
- Table 2 in [1] (Multi-GPUs Version)

    - GPoCM CUNet + LaSAFT (* proposed model) 
        ```shell script
        python main.py --problem_name conditioned_separation --mode train --musdb_root ../repos/musdb18_wav --n_blocks 9 --num_tdfs 6 --n_fft 4096 --hop_length 1024 --precision 16 --embedding_dim 64 --pin_memory True --save_top_k 3 --patience 10 --deterministic --model CUNET_TFC_GPoCM_LaSAFT --gpus 4 --distributed_backend ddp --sync_batchnorm True --run_id lasaft_2020 --batch_size 4 --seed 2020 --log False --lr 0.0001 --auto_lr_schedule True 

        ```
### 4. [Evaluation](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/issues/15#issuecomment-807883207)

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

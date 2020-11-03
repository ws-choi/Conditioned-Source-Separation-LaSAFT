# LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lasaft-latent-source-attentive-frequency/music-source-separation-on-musdb18)](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=lasaft-latent-source-attentive-frequency)

Check separated samples on this [demo page](https://lasaft.github.io/)!

## Caution! 
> This page is under construction. It might contain wrong information.

> We will upload our code and README after refactoring our code for better readability.

A Pytorch Implementation of the paper "LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation"
## Installation

```
conda install pytorch=1.6 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge ffmpeg librosa
conda install -c anaconda jupyter
pip install musdb museval pytorch_lightning effortless_config wandb pydub nltk spacy 
```

## Dataset

1. Download [Musdb18](https://sigsep.github.io/datasets/musdb.html)
2. Unzip files
3. We recommend you to use the wav file mode for the fast data preparation. 
    - please see the [musdb instruction](https://pypi.org/project/musdb/) for the wave file mode.
    ```shell script
   musdbconvert path/to/musdb-stems-root path/to/new/musdb-wav-root
   ```

## Demonstration: A Pretrained Model

[![demo](https://i.imgur.com/8hPZJIY.png)](https://youtu.be/buWnt89kVzs?t=8) 

Interactive Demonstration - [Colab Link](https://colab.research.google.com/github/ws-choi/Conditioned-Source-Separation-LaSAFT/blob/main/colab_demo/LaSAFT_with_GPoCM_Stella_Jang_Example.ipynb)

## Tutorial

> TBA

### 1. activate your conda

```shell script
conda activate yourcondaname
```

### 2. Training a proposed architecture

link: [Proposed architecture](https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/blob/main/lasaft/source_separation/conditioned/cunet/models/dcun_tfc_gpocm_lasaft.py) (CUNET TFC + GPoCM + LaSAFT)

```shell script
python main.py 
    --problem_name conditioned_separation
    --model CUNET_TFC_GPoCM_LaSAFT
    --mode train

    --run_id cunet_tfc_gpocm_lasaft
    --log False
    
    --musdb_root ../repos/musdb18_wav
    --musdb_is_wav True
    --filed_mode True
    
    --gpus 1
    --precision 16
    --batch_size 8
    --num_workers 8
    --pin_memory True
    --save_top_k 3
    --save_weights_only True
    --patience 50
    --min_epochs 100

    --train_loss spec_mse
    --val_loss raw_l1
    
    --optimizer adam
    --lr 0.001
    
    --n_fft 2048
    --hop_length 1024
    --num_frame 128
    --spec_type complex
    --spec_est_mode mapping
    --first_conv_activation relu
    --last_activation identity
    --input_channels 4
    
    --control_vector_type embedding
    --control_input_dim 4
    --embedding_dim 32
    --condition_to decoder
    
    --n_blocks 7
    --internal_channels 24
    --n_internal_layers 5
    --kernel_size_t 3 
    --kernel_size_f 3 
    --tfc_tdf_bias True
    --num_tdfs 6
    --dk 32
    
    --control_n_layer 4
    --control_type dense
    --pocm_type matmul
    --pocm_norm batch_norm
    
    --seed 2020
```

### 3. Evaluation

After training is done, checkpoints are saved in the following directory. 

```etc/modelname/run_id/*.ckpt```

For evaluation, 

> TBA

Below is the result.

> TBA


### 4. Interactive Report (wandb)

> TBA

## Indermediate Blocks

Please see this [document](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS/blob/master/paper_with_code/Paper%20with%20Code%20-%203.%20INTERMEDIATE%20BLOCKS.ipynb).

## How to use

### 1. Training

#### 1.1. Intermediate Block independent Parameters

##### 1.1.A. General Parameters

- ```--musdb_root``` musdb path
- ```--musdb_is_wav``` whether the path contains wav files or not
- ```--filed_mode``` whether you want to use filed mode or not. recommend to use it for the fast data preparation.
- ```--target_name``` one of vocals, drum, bass, other

##### 1.1.B. Training Environment

- ```--mode``` train or eval
- ```--gpus``` number of gpus
    - ***(WARN)*** gpus > 1 might be problematic when evaluating models.
- ```distributed_backend``` use this option only when you are using multi-gpus. distributed backend, one of ddp, dp, ... we recommend you to use ddp.
- ```--sync_batchnorm``` True only when you are using ddp
- ```--pin_memory```
- ```--num_workers```
- ```--precision``` 16 or 32
- ```--dev_mode``` whether you want a developement mode or not. dev mode is much faster because it uses only a small subset of the dataset.
- ```--run_id``` (optional) directory path where you want to store logs and etc. if none then the timestamp.
- ```--log``` True for default pytorch lightning log. ```wandb``` is also available.
- ```--seed``` random seed for a deterministic result. 

##### 1.1.C. Training hyperparmeters
- ```--batch_size``` trivial :)
- ```--optimizer``` adam, rmsprop, etc
- ```--lr``` learning rate
- ```--save_top_k``` how many top-k epochs you want to save the training state (criterion: validation loss)
- ```--patience``` early stop control parameter. see pytorch lightning docs.
- ```--min_epochs``` trivial :)
- ```--max_epochs``` trivial :)
- ```--model```
    - Dense_CUNet_TFC_FiLM
    - Dense_CUNet_TFC_GPoCM

##### 1.1.D. Fourier parameters
- ```--n_fft```
- ```--hop_length```
- ```num_frame``` number of frames (time slices) 
##### 1.1.F. criterion
- ```--train_loss```: spec_mse, raw_l1, etc...
- ```--val_loss```: spec_mse, raw_l1, etc...

#### 1.2. CU-net Parameters

- ```--n_blocks```: number of intermediate blocks. must be an odd integer. (default=7)
- ```--input_channels```: 
    - if you use two-channeled complex-valued spectrogram, then 4
    - if you use two-channeled manginutde spectrogram, then 2 
- ```--internal_channels```:  number of internal chennels (default=24)
- ```--first_conv_activation```: (default='relu')
- ```--last_activation```: (default='sigmoid')
- ```--t_down_layers```: list of layer where you want to doubles/halves the time resolution. if None, ds/us applied to every single layer. (default=None)
- ```--f_down_layers```: list of layer where you want to doubles/halves the frequency resolution. if None, ds/us applied to every single layer. (default=None)

#### 1.3. Separation Framework
- ```--spec_type```: type of a spectrogram. ['complex', 'magnitude']
- ```--spec_est_mode```: spectrogram estimation method. ['mapping', 'masking']

- **CaC Framework**
    - you can use cac framework [1] by setting
        - ```--spec_type complex --spec_est_mode mapping --last_activation identity```
- **Mag-only Framework**
    - if you want to use the traditional magnitude-only estimation with sigmoid, then try
        - ```--spec_type magnitude --spec_est_mode masking --last_activation sigmoid```
    - you can also change the last activation as follows
        - ```--spec_type magnitude --spec_est_mode masking --last_activation relu```
- Alternatives
    - you can build an svs framework with any combination of these parameters
    - e.g. ```--spec_type complex --spec_est_mode masking --last_activation tanh```


        
#### 1.4. Block-dependent Parameters (TBA)

##### 1.4.A. TDF Net

- ```--bn_factor```: bottleneck factor $bn$ (default=16)
- ```--min_bn_units```: when target frequency domain size is too small, we just use this value instead of $\frac{f}{bn}$. (default=16)
- ```--bias```: (default=False) 
- ```--tdf_activation```: activation function of each block (default=relu)       

---

##### 1.4.B. TDC Net

- ```--n_internal_layers```: number of 1-d CNNs in a block (default=5)
- ```--kernel_size_f```: size of kernel of frequency-dimension (default=3)
- ```--tdc_activation```: activation function of each block (default=relu)
        
---
        
##### 1.4.C. TFC Net
- ```--n_internal_layers```: number of 1-d CNNs in a block (default=5)
- ```--kernel_size_t```: size of kernel of time-dimension (default=3)
- ```--kernel_size_f```: size of kernel of frequency-dimension (default=3)
- ```--tfc_activation```: activation function of each block (default=relu)

---
        
##### 1.4.D. TFC_TDF Net
- ```--n_internal_layers```: number of 1-d CNNs in a block (default=5)
- ```--kernel_size_t```: size of kernel of time-dimension (default=3)
- ```--kernel_size_f```: size of kernel of frequency-dimension (default=3)
- ```--tfc_tdf_activation```: activation function of each block (default=relu)       
- ```--bn_factor```: bottleneck factor $bn$ (default=16)
- ```--min_bn_units```: when target frequency domain size is too small, we just use this value instead of $\frac{f}{bn}$. (default=16)
- ```--tfc_tdf_bias```: (default=False)

---

## Reproducible Experimental Results

> TBA

### Interactive Report (wandb)

> TBA
>
## You can cite this paper as follows:

> @misc{choi2020lasaft,
      title={LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation}, 
      author={Woosung Choi and Minseok Kim and Jaehwa Chung and Soonyoung Jung},
      year={2020},
      eprint={2010.11631},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
## Reference

[1]  Woosung Choi,  Minseok Kim,  Jaehwa Chung, and Soonyoung Jung, “LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation.,” arXiv preprint arXiv:2010.11631 (2020).  

## Other Links

- This code borrows heavily from [ISMIR2020_U_Nets_SVS](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS/) repository. Many thanks.

- The original Conditional U-Net 
    - [tensorflow version](https://github.com/gabolsgabs/cunet) (official)
    - [pytorch version](https://github.com/ws-choi/Conditioned-U-Net-pytorch)

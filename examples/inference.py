from lasaft.pretrained import PreTrainedLaSAFTNet
import librosa
import sys
import torch
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_path = sys.argv[1]
audio = librosa.load(audio_path, 44100, False)[0].T
model = PreTrainedLaSAFTNet(model_name='lasaft_large_2020').to(device)

vocals = model.separate_track(audio, 'vocals', overlap_ratio=0.5)
sf.write('vocals.wav', vocals, 44100, format='WAV')
# drums = model.separate_track(audio, 'drums')
# bass = model.separate_track(audio, 'bass')
# other = model.separate_track(audio, 'other')

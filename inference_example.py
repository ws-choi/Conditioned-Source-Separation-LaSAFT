from lasaft.pretrained import PreTrainedLaSAFTNet
import librosa
import sys
import torch
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_path = sys.argv[1]
audio = librosa.load(audio_path, 44100, False)[0].T
model = PreTrainedLaSAFTNet(model_name='lasaft_large_2020').to(device)

vocals = model.separate_track(audio, 'vocals', overlap_ratio=0.5, batch_size=8)
sf.write('vocals.wav', vocals, 44100, format='WAV')
print('vocals.wav created')
drums = model.separate_track(audio, 'drums', overlap_ratio=0.5, batch_size=8)
sf.write('drums.wav', drums, 44100, format='WAV')
print('drums.wav created')
bass = model.separate_track(audio, 'bass', overlap_ratio=0.5, batch_size=8)
sf.write('bass.wav', bass, 44100, format='WAV')
print('bass.wav created')
other = model.separate_track(audio, 'other', overlap_ratio=0.5, batch_size=8)
sf.write('other.wav', other, 44100, format='WAV')
print('other.wav created')

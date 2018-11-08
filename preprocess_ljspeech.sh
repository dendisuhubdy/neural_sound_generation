source activate torch
python preprocess.py --preset=./presets/ljspeech_mixture.json ljspeech ../data/LJSpeech ../data/LJSpeech_Npz_tacotron_80mels

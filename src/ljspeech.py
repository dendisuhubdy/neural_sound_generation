from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import librosa

from nnmnkwii import preprocessing as P
from os.path import exists
from hparams_tacotron import hparams

import audio_tacotron as audio
from audio_tacotron import is_mulaw_quantize, is_mulaw, is_raw


def build_from_path(in_dir, out_dir, num_workers=1, sample_rate=22050, fft_size=1024, hop_size=256, n_mels=80, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index, wav_path, text, sample_rate, fft_size, hop_size, n_mels)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, sample_rate, fft_size, hop_size, n_mels):
    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Mu-law quantize
    # this really gets called if input_type in hparams
    # is changed from raw to mulaw
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = P.mulaw_quantize(wav, hparams.quantize_channels)

        # Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start:end]
        out = out[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = P.mulaw(wav, hparams.quantize_channels)
        constant_values = P.mulaw(0.0, hparams.quantize_channels)
        out_dtype = np.float32
    else:
        # [-1, 1]
        out = wav
        constant_values = 0.0
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    # mel_spectrogram = audio.melspectrogram(wav, 22050, 1024, 40).astype(np.float32).T
    # change this line to adjust hyperparams
    mel_spectrogram = audio.melspectrogram(wav, sample_rate, fft_size, hop_size, n_mels).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, fft_size, hop_size)

    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    # assert len(out) >= N * audio.get_hop_size()
    assert len(out) >= N * hop_size

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    # out = out[:N * audio.get_hop_size()]
    # assert len(out) % audio.get_hop_size() == 0
    out = out[:N * hop_size]
    assert len(out) % hop_size == 0

    timesteps = len(out)
    # compute example reconstruction
    # change this line to adjust hparams
    # signal = audio.inv_mel_spectrogram(mel_spectrogram, sample_rate, fft_size, n_mels)
    
    # mel_spectrogram = mel_spectrogram.T

    # Write the spectrograms to disk:
    audio_filename = 'ljspeech-audio-%05d.npy' % index
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    # recon_audio_filename = 'ljspeech-audio-%05d.wav' % index
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)
    # audio.save_wav(signal, os.path.join(out_dir, recon_audio_filename))

    # Return a tuple describing this training example:
    return (audio_filename, mel_filename, timesteps, text)

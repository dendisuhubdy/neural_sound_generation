import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile

from hparams_tacotron import hparams

import lws


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]

def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))

def save_wavenet_wav(wav, path):
    librosa.output.write_wav(path, wav, sr=hparams.sample_rate)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end

def trim_silence(wav):
    '''Trim leading and trailing silence
    Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
    '''
    #Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
    return librosa.effects.trim(wav, top_db= hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]

def get_hop_size():
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
        print("Predefined hop_size is ", hop_size)
    return hop_size

def linearspectrogram(wav, fft_size, hop_size):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize),
              fft_size, hop_size)
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav, sample_rate, fft_size, hop_size, n_mels):
    D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize),
              fft_size, hop_size)
    S = _amp_to_db(_linear_to_mel(np.abs(D), sample_rate, fft_size, n_mels)) - hparams.ref_level_db
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.signal_normalization:
        return _normalize(S)
    return S

def inv_linear_spectrogram(linear_spectrogram, fft_size, hop_size):
    '''Converts linear spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hparams.ref_level_db) #Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(fft_size, hop_size)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, fft_size, hop_size),
                               hparams.preemphasis, hparams.preemphasize)


def inv_mel_spectrogram(mel_spectrogram, sample_rate, fft_size, hop_size, n_mel):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db),
                       sample_rate, fft_size, n_mel)  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(fft_size, hop_size)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S ** hparams.power, fft_size, hop_size),
                               hparams.preemphasis, hparams.preemphasize)

def _lws_processor(fft_size, hop_size):
    return lws.lws(fft_size, hop_size, mode="speech")


def lws_num_frames(length, fsize, fshift):
    """Compute number of time frames of lws spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def lws_pad_lr(x, fsize, fshift):
    """Compute left and right padding lws internally uses
    """
    M = lws_num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r

def _griffin_lim(S, fft_size, hop_size):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hop_size)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, fft_size, hop_size)))
        y = _istft(S_complex * angles, hop_size)
    return y

def _stft(y, fft_size, hop_size):
    if hparams.use_lws:
        return _lws_processor(fft_size, hop_size).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=fft_size, hop_length=hop_size)

def _istft(y, hop_size):
    return librosa.istft(y, hop_length=hop_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    '''compute right padding (final frame)
    '''
    return int(fsize // 2)


# Conversions
# _mel_basis = None
# _inv_mel_basis = None

def _linear_to_mel(spectogram, sample_rate, fft_size, n_mels, _mel_basis = None):
    # global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(sample_rate, fft_size, n_mels)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, sample_rate, fft_size, n_mels, _inv_mel_basis=None):
    # global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(sample_rate, fft_size, n_mels))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(sample_rate, fft_size, n_mels):
    assert hparams.fmax <= hparams.sample_rate // 2
    # return librosa.filters.mel(hparams.sample_rate,
                               # hparams.fft_size,
                               # n_mels=hparams.num_mels,
                               # fmin=hparams.fmin,
                               # fmax=hparams.fmax)
    return librosa.filters.mel(sample_rate,
                               fft_size,
                               n_mels=n_mels,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)

def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
                 -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value,
                    hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)


def _assert_valid_input_type(s):
    assert s == "mulaw-quantize" or s == "mulaw" or s == "raw"


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == "mulaw-quantize"


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == "mulaw"


def is_raw(s):
    _assert_valid_input_type(s)
    return s == "raw"


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)

if __name__ == "__main__":
    import sys
    import os
    out_dir = str(sys.argv[1])
    recon_sample_rate = int(sys.argv[2]) 
    recon_fft_size = int(sys.argv[3])
    recon_hop_size = int(sys.argv[4])
    recon_n_mels = int(sys.argv[5])
    
    # audio load
    # src_wav_filename = "ljspeech-audio-00001.npy"
    # orig_wav = np.load(os.path.join(out_dir, src_wav_filename))
    # save_wav(orig_wav, "./orig-ljspeech-audio-00001.wav")
    
    # compute the melspectrogram
    # assign filename
    mel_filename = "ljspeech-mel-00001.npy"
    # load the numpy dumped file
    melspectrogram = np.load(os.path.join(out_dir, mel_filename))
    melspectrogram = melspectrogram.T
    # print it's values
    assert np.array(melspectrogram).shape[0] == recon_n_mels

    signal = inv_mel_spectrogram(melspectrogram,
                                 recon_sample_rate,
                                 recon_fft_size,
                                 recon_hop_size,
                                 recon_n_mels)

    save_wav(signal, "./recon-ljspeech-audio-00001-{}-{}-{}-{}.wav".format(recon_sample_rate,
                                                               recon_fft_size,
                                                               recon_hop_size,
                                                               recon_n_mels))

"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""

import numpy as np
from scipy import signal
from scipy.io import wavfile

# librosa as sound processors
import librosa
import librosa.filters
# Fast spectrogram phase recovery
# using Local Weighted Sums (LWS)
import lws


def load_wav(path, sample_rate):
    return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sample_rate, wav.astype(np.int16))


def preemphasis(x, preemphasis):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, preemphasis)


def inv_preemphasis(x, preemphasis):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x, preemphasis)


def spectrogram(y, ref_level_db):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram, ref_level_db, power):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) +
                   ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** power)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)


def melspectrogram(y, ref_level_db, allow_clipping_in_normalization,
                   min_level_db):
    D = _lws_processor().stft(preemphasis(y)).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db
    if not allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - min_level_db >= 0
    return _normalize(S)


def _lws_processor(fft_size, hop_size):
    return lws.lws(fft_size, hop_size, mode="speech")


def _linear_to_mel(spectrogram):
    _mel_basis = None
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis(fmin, fmax, sample_rate, fft_size, num_mels):
    if fmax is not None:
        assert fmax <= sample_rate // 2
    return librosa.filters.mel(sample_rate, fft_size,
                               fmin=fmin,
                               fmax=fmax,
                               n_mels=num_mels)


def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def _denormalize(S, min_level_db):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
# coding: utf-8
from __future__ import with_statement, print_function, absolute_import
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.io.wavfile import read
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.preprocessing.data import _handle_zeros_in_scale

# librosa as sound processors
import librosa
import librosa.filters
# Fast spectrogram phase recovery
# using Local Weighted Sums (LWS)
import lws
import umap

import torch
import torchaudio
from torchvision import datasets, transforms
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def round_down(num, divisor):
    return num - (num%divisor)


def round_up(x):
    return int(math.ceil(x / 10.0)) * 10

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len).long().cuda()
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path, sr):
    sampling_rate, data = read(full_path)
    assert sr == sampling_rate, "{} SR doesn't match {} on path {}".format(
        sr, sampling_rate, full_path)
    return torch.FloatTensor(data.astype(np.float32))


def load_filepaths_and_text(filename, sort_by_length, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]

    if sort_by_length:
        filepaths_and_text.sort(key=lambda x: len(x[1]))

    return filepaths_and_text


def to_device(x, device):
    # unsqueeze might be the wrong
    # approach here 
    x = x.unsqueeze(1)
    x = x.contiguous().to(device) #.cuda(async=True)
    return torch.autograd.Variable(x)


def load_wav(path, sample_rate):
    return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate):
    wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sample_rate, wav.astype(np.int16))


def trim(quantized, silence_threshold):
    start, end = start_and_end_indices(quantized, silence_threshold)
    return quantized[start:end]


def adjust_time_resolution(quantized, mel, silence_threshold):
    """Adjust time resolution by repeating features
    Args:
        quantized (ndarray): (T,)
        mel (ndarray): (N, D)
    Returns:
        tuple: Tuple of (T,) and (T, D)
    """
    assert len(quantized.shape) == 1
    assert len(mel.shape) == 2

    upsample_factor = quantized.size // mel.shape[0]
    mel = np.repeat(mel, upsample_factor, axis=0)
    n_pad = quantized.size - mel.shape[0]
    if n_pad != 0:
        assert n_pad > 0
        mel = np.pad(mel, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)

    # trim
    start, end = start_and_end_indices(quantized, silence_threshold)

    return quantized[start:end], mel[start:end, :]


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


def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


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


def _normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def _denormalize(S, min_level_db):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def visualize_embedding(model):
    proj = umap.UMAP(n_neighbors=3,
                     min_dist=0.1,
                     metric='euclidean').fit_transform(model._embedding.weight.data.cpu())
    # plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
    return proj


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


def get_audio_length(path):
    output = subprocess.check_output(
        ['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(
            tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio


def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=True)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(
            noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        windows = {'hamming': scipy.signal.hamming,
                   'hann': scipy.signal.hann,
                   'blackman': scipy.signal.blackman,
                   'bartlett': scipy.signal.bartlett}

        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(
            audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(
            filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size]
                     for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(
            math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        # Get every Nth bin, starting from rank
        samples = bins[offset::self.num_replicas]
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]


def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else x.sign()


def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else x.log1p()


def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else x.abs()


def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()


def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()


def mulaw(x, mu=256):
    """Mu-Law companding
    Method described in paper [1]_.
    .. math::
        f(x) = sign(x) \ln (1 + \mu |x|) / \ln (1 + \mu)
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Compressed signal ([-1, 1])
    See also:
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    .. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
        implementations using the tms320c54x." SPRA163 (1997).
    """
    return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)


def inv_mulaw(y, mu=256):
    """Inverse of mu-law companding (mu-law expansion)
    .. math::
        f^{-1}(x) = sign(y) (1 / \mu) (1 + \mu)^{|y|} - 1)
    Args:
        y (array-like): Compressed signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncomprresed signal (-1 <= x <= 1)
    See also:
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    """
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def mulaw_quantize(x, mu=256):
    """Mu-Law companding + quantize
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)
    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).
    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> y = P.mulaw_quantize(x)
        >>> print(y.min(), y.max(), y.dtype)
        15 246 int64
    See also:
        :func:`nnmnkwii.preprocessing.mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw_quantize`
    """
    y = mulaw(x, mu)
    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)


def inv_mulaw_quantize(y, mu=256):
    """Inverse of mu-law companding + quantize
    Args:
        y (array-like): Quantized signal (∈ [0, mu]).
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncompressed signal ([-1, 1])
    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> from nnmnkwii import preprocessing as P
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> x_hat = P.inv_mulaw_quantize(P.mulaw_quantize(x))
        >>> x_hat = (x_hat * 32768).astype(np.int16)
    See also:
        :func:`nnmnkwii.preprocessing.mulaw`
        :func:`nnmnkwii.preprocessing.inv_mulaw`
        :func:`nnmnkwii.preprocessing.mulaw_quantize`
    """
    # [0, m) to [-1, 1]
    y = 2 * _asfloat(y) / mu - 1
    return inv_mulaw(y, mu)


def preemphasis(x, coef=0.97):
    """Pre-emphasis
    Args:
        x (1d-array): Input signal.
        coef (float): Pre-emphasis coefficient.
    Returns:
        array: Output filtered signal.
    See also:
        :func:`inv_preemphasis`
    Examples:
        >>> from nnmnkwii.util import example_audio_file
        >>> from scipy.io import wavfile
        >>> fs, x = wavfile.read(example_audio_file())
        >>> x = x.astype(np.float64)
        >>> from nnmnkwii import preprocessing as P
        >>> y = P.preemphasis(x, coef=0.97)
        >>> assert x.shape == y.shape
    """
    b = np.array([1., -coef], x.dtype)
    a = np.array([1.], x.dtype)
    return signal.lfilter(b, a, x)


def inv_preemphasis(x, coef=0.97):
    """Inverse operation of pre-emphasis
    Args:
        x (1d-array): Input signal.
        coef (float): Pre-emphasis coefficient.
    Returns:
        array: Output filtered signal.
    See also:
        :func:`preemphasis`
    Examples:
        >>> from nnmnkwii.util import example_audio_file
        >>> from scipy.io import wavfile
        >>> fs, x = wavfile.read(example_audio_file())
        >>> x = x.astype(np.float64)
        >>> from nnmnkwii import preprocessing as P
        >>> x_hat = P.inv_preemphasis(P.preemphasis(x, coef=0.97), coef=0.97)
        >>> assert np.allclose(x, x_hat)
    """
    b = np.array([1.], x.dtype)
    a = np.array([1., -coef], x.dtype)
    return signal.lfilter(b, a, x)


def _delta(x, window):
    return np.correlate(x, window, mode="same")


def _apply_delta_window(x, window):
    """Returns delta features given a static features and a window.
    Args:
        x (numpy.ndarray): Input static features, of shape (``T x D``).
        window (numpy.ndarray): Window coefficients.
    Returns:
        (ndarray): Delta features, shape　(``T x D``).
    """
    T, D = x.shape
    y = np.zeros_like(x)
    for d in range(D):
        y[:, d] = _delta(x[:, d], window)
    return y


def delta_features(x, windows):
    """Compute delta features and combine them.
    This function computes delta features given delta windows, and then
    returns combined features (e.g., static + delta + delta-delta).
    Note that if you want to keep static features, you need to give
    static window as well as delta windows.
    Args:
        x (numpy.ndarray): Input static features, of shape (``T x D``).
        y (list): List of windows. See :func:`nnmnkwii.paramgen.mlpg` for what
            the delta window means.
    Returns:
        numpy.ndarray: static + delta features (``T x (D * len(windows)``).
    Examples:
        >>> from nnmnkwii.preprocessing import delta_features
        >>> windows = [
        ...         (0, 0, np.array([1.0])),            # static
        ...         (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
        ...         (1, 1, np.array([1.0, -2.0, 1.0])), # delta-delta
        ...     ]
        >>> T, static_dim = 10, 24
        >>> x = np.random.rand(T, static_dim)
        >>> y = delta_features(x, windows)
        >>> assert y.shape == (T, static_dim * len(windows))
    """
    T, D = x.shape
    assert len(windows) > 0
    combined_features = np.empty((T, D * len(windows)), dtype=x.dtype)
    for idx, (_, _, window) in enumerate(windows):
        combined_features[:, D * idx:D * idx +
                          D] = _apply_delta_window(x, window)
    return combined_features


def trim_zeros_frames(x, eps=1e-7):
    """Remove trailling zeros frames.
    Similar to :func:`numpy.trim_zeros`, trimming trailing zeros features.
    Args:
        x (numpy.ndarray): Feature matrix, shape (``T x D``)
        eps (float): Values smaller than ``eps`` considered as zeros.
    Returns:
        numpy.ndarray: Trimmed 2d feature matrix, shape (``T' x D``)
    Examples:
        >>> import numpy as np
        >>> from nnmnkwii.preprocessing import trim_zeros_frames
        >>> x = np.random.rand(100,10)
        >>> y = trim_zeros_frames(x)
    """

    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.
    return x[: len(np.trim_zeros(s))]


def remove_zeros_frames(x, eps=1e-7):
    """Remove zeros frames.
    Given a feature matrix, remove all zeros frames as well as trailing ones.
    Args:
        x (numpy.ndarray): 2d feature matrix, shape (``T x D``)
        eps (float): Values smaller than ``eps`` considered as zeros.
    Returns:
        numpy.ndarray: Zeros-removed 2d feature matrix, shape (``T' x D``).
    Examples:
        >>> import numpy as np
        >>> from nnmnkwii.preprocessing import remove_zeros_frames
        >>> x = np.random.rand(100,10)
        >>> y = remove_zeros_frames(x)
    """
    T, D = x.shape
    s = np.sum(np.abs(x), axis=1)
    s[s < eps] = 0.
    return x[s > eps]


def adjust_frame_length(x, pad=True, divisible_by=1, **kwargs):
    """Adjust frame length given a feature vector or matrix.
    This adjust the number of frames of a given feature vector or matrix to be
    divisible by ``divisible_by`` by padding to the end or removing the last
    few frames. Default uses zero-padding.
    Args:
        x (numpy.ndarray): Input 1d or 2d array, shape (``T,`` or ``T x D``).
        pad (bool) : If True, pads values to the end, otherwise removes last few
          frames to ensure same frame lengths.
        divisible_by (int) : If ``divisible_by`` > 0, number of frames will be
          adjusted to be divisible by ``divisible_by``.
        kwargs (dict): Keyword argments passed to :func:`numpy.pad`. Default is
          mode = ``constant``, which means zero padding.
    Returns:
        numpy.ndarray: adjusted array, of each shape (``T`` or ``T' x D``).
    Examples:
        >>> from nnmnkwii.preprocessing import adjust_frame_length
        >>> import numpy as np
        >>> x = np.zeros((10, 1))
        >>> x = adjust_frame_length(x, pad=True, divisible_by=3)
        >>> assert x.shape[0] == 12
    See also:
        :func:`nnmnkwii.preprocessing.adjust_frame_lengths`
    """
    kwargs.setdefault("mode", "constant")

    assert x.ndim == 2 or x.ndim == 1
    Tx = x.shape[0]

    if divisible_by > 1:
        rem = Tx % divisible_by
        if rem == 0:
            T = Tx
        else:
            if pad:
                T = Tx + divisible_by - rem
            else:
                T = Tx - rem
    else:
        T = Tx

    if Tx != T:
        if T > Tx:
            if x.ndim == 1:
                x = np.pad(x, (0, T - Tx), **kwargs)
            elif x.ndim == 2:
                x = np.pad(x, [(0, T - Tx), (0, 0)], **kwargs)
        else:
            x = x[:T]

    return x


def adjust_frame_lengths(x, y, pad=True, ensure_even=False, divisible_by=1,
                         **kwargs):
    """Adjust frame lengths given two feature vectors or matrices.
    This ensures that two feature vectors or matrices have same number of
    frames, by padding to the end or removing the last few frames.
    Default uses zero-padding.
    .. warning::
        ``ensure_even`` is deprecated and will be removed in v0.1.0.
        Use ``divisible_by=2`` instead.
    Args:
        x (numpy.ndarray): Input 2d feature matrix, shape (``T^1 x D``).
        y (numpy.ndarray): Input 2d feature matrix, shape (``T^2 x D``).
        pad (bool) : If True, pads values to the end, otherwise removes last few
          frames to ensure same frame lengths.
        divisible_by (int) : If ``divisible_by`` > 0, number of frames will be
          adjusted to be divisible by ``divisible_by``.
        kwargs (dict): Keyword argments passed to :func:`numpy.pad`. Default is
          mode = ``constant``, which means zero padding.
    Returns:
        Tuple: Pair of adjusted feature matrices, of each shape (``T x D``).
    Examples:
        >>> from nnmnkwii.preprocessing import adjust_frame_lengths
        >>> import numpy as np
        >>> x = np.zeros((10, 1))
        >>> y = np.zeros((11, 1))
        >>> x, y = adjust_frame_lengths(x, y)
        >>> assert len(x) == len(y)
    See also:
        :func:`nnmnkwii.preprocessing.adjust_frame_length`
    """
    assert x.ndim in [1, 2] and y.ndim in [1, 2]
    kwargs.setdefault("mode", "constant")
    Tx = x.shape[0]
    Ty = y.shape[0]
    if x.ndim == 2:
        assert x.shape[-1] == y.shape[-1]

    if ensure_even:
        divisible_by = 2

    if pad:
        T = max(Tx, Ty)
        if divisible_by > 1:
            rem = T % divisible_by
            if rem != 0:
                T = T + divisible_by - rem
    else:
        T = min(Tx, Ty)
        if divisible_by > 1:
            rem = T % divisible_by
            T = T - rem

    if Tx != T:
        if Tx < T:
            if x.ndim == 1:
                x = np.pad(x, (0, T - Tx), **kwargs)
            elif x.ndim == 2:
                x = np.pad(x, [(0, T - Tx), (0, 0)], **kwargs)
        else:
            x = x[:T]

    if Ty != T:
        if Ty < T:
            if y.ndim == 1:
                y = np.pad(y, (0, T - Ty), **kwargs)
            elif y.ndim == 2:
                y = np.pad(y, [(0, T - Ty), (0, 0)], **kwargs)
        else:
            y = y[:T]

    return x, y


def meanvar(dataset, lengths=None, mean_=0., var_=0.,
            last_sample_count=0, return_last_sample_count=False):
    """Mean/variance computation given a iterable dataset
    Dataset can have variable length samples. In that cases, you need to
    explicitly specify lengths for all the samples.
    Args:
        dataset (nnmnkwii.datasets.Dataset): Dataset
        lengths: (list): Frame lengths for each dataset sample.
        mean\_ (array or scalar): Initial value for mean vector.
        var\_ (array or scaler): Initial value for variance vector.
        last_sample_count (int): Last sample count. Default is 0. If you set
          non-default ``mean_`` and ``var_``, you need to set
          ``last_sample_count`` property. Typically this will be the number of
          time frames ever seen.
        return_last_sample_count (bool): Return ``last_sample_count`` if True.
    Returns:
        tuple: Mean and variance for each dimention. If
          ``return_last_sample_count`` is True, returns ``last_sample_count``
          as well.
    See also:
        :func:`nnmnkwii.preprocessing.meanstd`, :func:`nnmnkwii.preprocessing.scale`
    Examples:
        >>> from nnmnkwii.preprocessing import meanvar
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(y) for y in Y]
        >>> data_mean, data_var = meanvar(Y, lengths)
    """
    dtype = dataset[0].dtype

    for idx, x in enumerate(dataset):
        if lengths is not None:
            x = x[:lengths[idx]]
        mean_, var_, _ = _incremental_mean_and_var(
            x, mean_, var_, last_sample_count)
        last_sample_count += len(x)
    mean_, var_ = mean_.astype(dtype), var_.astype(dtype)

    if return_last_sample_count:
        return mean_, var_, last_sample_count
    else:
        return mean_, var_


def meanstd(dataset, lengths=None, mean_=0., var_=0.,
            last_sample_count=0, return_last_sample_count=False):
    """Mean/std-deviation computation given a iterable dataset
    Dataset can have variable length samples. In that cases, you need to
    explicitly specify lengths for all the samples.
    Args:
        dataset (nnmnkwii.datasets.Dataset): Dataset
        lengths: (list): Frame lengths for each dataset sample.
        mean\_ (array or scalar): Initial value for mean vector.
        var\_ (array or scaler): Initial value for variance vector.
        last_sample_count (int): Last sample count. Default is 0. If you set
          non-default ``mean_`` and ``var_``, you need to set
          ``last_sample_count`` property. Typically this will be the number of
          time frames ever seen.
        return_last_sample_count (bool): Return ``last_sample_count`` if True.
    Returns:
        tuple: Mean and variance for each dimention. If
          ``return_last_sample_count`` is True, returns ``last_sample_count``
          as well.
    See also:
        :func:`nnmnkwii.preprocessing.meanvar`, :func:`nnmnkwii.preprocessing.scale`
    Examples:
        >>> from nnmnkwii.preprocessing import meanstd
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(y) for y in Y]
        >>> data_mean, data_std = meanstd(Y, lengths)
    """
    ret = meanvar(dataset, lengths, mean_, var_,
                  last_sample_count, return_last_sample_count)
    m, v = ret[0], ret[1]
    v = _handle_zeros_in_scale(np.sqrt(v))
    if return_last_sample_count:
        assert len(ret) == 3
        return m, v, ret[2]
    else:
        return m, v


def minmax(dataset, lengths=None):
    """Min/max computation given a iterable dataset
    Dataset can have variable length samples. In that cases, you need to
    explicitly specify lengths for all the samples.
    Args:
        dataset (nnmnkwii.datasets.Dataset): Dataset
        lengths: (list): Frame lengths for each dataset sample.
    See also:
        :func:`nnmnkwii.preprocessing.minmax_scale`
    Examples:
        >>> from nnmnkwii.preprocessing import minmax
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(x) for x in X]
        >>> data_min, data_max = minmax(X, lengths)
    """
    max_ = -np.inf
    min_ = np.inf

    for idx, x in enumerate(dataset):
        if lengths is not None:
            x = x[:lengths[idx]]
        min_ = np.minimum(min_, np.min(x, axis=(0,)))
        max_ = np.maximum(max_, np.max(x, axis=(0,)))

    return min_, max_


def scale(x, data_mean, data_std):
    """Mean/variance scaling.
    Given mean and variances, apply mean-variance normalization to data.
    Args:
        x (array): Input data
        data_mean (array): Means for each feature dimention.
        data_std (array): Standard deviation for each feature dimention.
    Returns:
        array: Scaled data.
    Examples:
        >>> from nnmnkwii.preprocessing import meanstd, scale
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> lengths = [len(y) for y in Y]
        >>> data_mean, data_std = meanstd(Y, lengths)
        >>> scaled_y = scale(Y[0], data_mean, data_std)
    See also:
        :func:`nnmnkwii.preprocessing.inv_scale`
    """
    return (x - data_mean) / _handle_zeros_in_scale(data_std, copy=False)


def inv_scale(x, data_mean, data_std):
    """Inverse tranform of mean/variance scaling.
    Given mean and variances, apply mean-variance denormalization to data.
    Args:
        x (array): Input data
        data_mean (array): Means for each feature dimention.
        data_std (array): Standard deviation for each feature dimention.
    Returns:
        array: Denormalized data.
    See also:
        :func:`nnmnkwii.preprocessing.scale`
    """
    return data_std * x + data_mean


def __minmax_scale_factor(data_min, data_max, feature_range):
    data_range = data_max - data_min
    scale = (feature_range[1] - feature_range[0]) / \
        _handle_zeros_in_scale(data_range, copy=False)
    return scale


def minmax_scale_params(data_min, data_max, feature_range=(0, 1)):
    """Compute parameters required to perform min/max scaling.
    Given data min, max and feature range, computes scalining factor and
    minimum value. Min/max scaling can be done as follows:
    .. code-block:: python
        x_scaled = x * scale_ + min_
    Args:
        x (array): Input data
        data_min (array): Data min for each feature dimention.
        data_max (array): Data max for each feature dimention.
        feature_range (array like): Feature range.
    Returns:
        tuple: Minimum value and scaling factor for scaled data.
    Examples:
        >>> from nnmnkwii.preprocessing import minmax, minmax_scale
        >>> from nnmnkwii.preprocessing import minmax_scale_params
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> data_min, data_max = minmax(X)
        >>> min_, scale_ = minmax_scale_params(data_min, data_max)
        >>> scaled_x = minmax_scale(X[0], min_=min_, scale_=scale_)
    See also:
        :func:`nnmnkwii.preprocessing.minmax_scale`,
        :func:`nnmnkwii.preprocessing.inv_minmax_scale`
    """
    scale_ = __minmax_scale_factor(data_min, data_max, feature_range)
    min_ = feature_range[0] - data_min * scale_
    return min_, scale_


def minmax_scale(x, data_min=None, data_max=None, feature_range=(0, 1),
                 scale_=None, min_=None):
    """Min/max scaling for given a single data.
    Given data min, max and feature range, apply min/max normalization to data.
    Optionally, you can get a little performance improvement to give scaling
    factor (``scale_``) and minimum value (``min_``) used in scaling explicitly.
    Those values can be computed by
    :func:`nnmnkwii.preprocessing.minmax_scale_params`.
    .. note::
        If ``scale_`` and ``min_`` are given, ``feature_range`` will be ignored.
    Args:
        x (array): Input data
        data_min (array): Data min for each feature dimention.
        data_max (array): Data max for each feature dimention.
        feature_range (array like): Feature range.
        scale\_ ([optional]array): Scaling factor.
        min\_ ([optional]array): Minimum value for scaling.
    Returns:
        array: Scaled data.
    Raises:
        ValueError: If (``data_min``, ``data_max``) or
          (``scale_`` and ``min_``) are not specified.
    See also:
        :func:`nnmnkwii.preprocessing.inv_minmax_scale`,
        :func:`nnmnkwii.preprocessing.minmax_scale_params`
    Examples:
        >>> from nnmnkwii.preprocessing import minmax, minmax_scale
        >>> from nnmnkwii.util import example_file_data_sources_for_acoustic_model
        >>> from nnmnkwii.datasets import FileSourceDataset
        >>> X, Y = example_file_data_sources_for_acoustic_model()
        >>> X, Y = FileSourceDataset(X), FileSourceDataset(Y)
        >>> data_min, data_max = minmax(X)
        >>> scaled_x = minmax_scale(X[0], data_min, data_max)
    """
    if (scale_ is None or min_ is None) and (data_min is None or data_max is None):
        raise ValueError("""
`data_min` and `data_max` or `scale_` and `min_` must be specified to perform minmax scale""")
    if scale_ is None:
        scale_ = __minmax_scale_factor(data_min, data_max, feature_range)
    if min_ is None:
        min_ = feature_range[0] - data_min * scale_
    return x * scale_ + min_


def inv_minmax_scale(x, data_min=None, data_max=None, feature_range=(0, 1),
                     scale_=None, min_=None):
    """Inverse transform of min/max scaling for given a single data.
    Given data min, max and feature range, apply min/max denormalization to data.
    .. note::
        If ``scale_`` and ``min_`` are given, ``feature_range`` will be ignored.
    Args:
        x (array): Input data
        data_min (array): Data min for each feature dimention.
        data_max (array): Data max for each feature dimention.
        feature_range (array like): Feature range.
        scale\_ ([optional]array): Scaling factor.
        min\_ ([optional]array): Minimum value for scaling.
    Returns:
        array: Scaled data.
    Raises:
        ValueError: If (``data_min``, ``data_max``) or
          (``scale_`` and ``min_``) are not specified.
    See also:
        :func:`nnmnkwii.preprocessing.minmax_scale`,
        :func:`nnmnkwii.preprocessing.minmax_scale_params`
    """
    if (scale_ is None or min_ is None) and (data_min is None or data_max is None):
        raise ValueError("""
`data_min` and `data_max` or `scale_` and `min_` must be specified to perform inverse of minmax scale""")
    if scale_ is None:
        scale_ = __minmax_scale_factor(data_min, data_max, feature_range)
    if min_ is None:
        min_ = feature_range[0] - data_min * scale_
    return (x - min_) / scale_

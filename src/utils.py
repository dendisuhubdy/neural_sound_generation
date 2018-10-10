"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
# coding: utf-8
from __future__ import with_statement, print_function, absolute_import
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.io.wavfile import read

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


# def to_gpu(x):
    # x = x.contiguous().cuda(async=True)
    # return torch.autograd.Variable(x)


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

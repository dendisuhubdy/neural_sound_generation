"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.distributed import DistributedSampler

import random
import numpy as np
import math
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from utils import text_to_sequence

from features import dynamic_range_compression
from features import window_sumsquare, dynamic_range_compression
from utils import (load_wav_to_torch, load_filepaths_and_text,
                   round_down, round_up)


def load_training_data(args, kwargs):
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)('./{}/{}/'.format(args.datadir, args.dataset),
                                       train=True, download=True,
                                       transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    return train_loader


def load_test_data(args, kwargs):
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)('./{}/{}/'.format(args.datadir, args.dataset),
                                       train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    return test_loader


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_sampler = DistributedSampler(trainset) \
        if hparams.distributed_run else None
    
    val_sampler = DistributedSampler(valset) \
        if hparams.distributed_run else None

    train_loader = DataLoader(trainset,
                              num_workers=1,
                              shuffle=False,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size,
                              pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    # here you can just output the valset
    # or you can load it as a data loader
    val_loader = DataLoader(valset,
                            num_workers=1,
                            shuffle=False,
                            sampler=val_sampler,
                            batch_size=hparams.batch_size,
                            pin_memory=False,
                            drop_last=True, collate_fn=collate_fn)
    return train_loader, val_loader, collate_fn


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024,
                 hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050,
                 mel_fmin=0.0, mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, batch):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(batch.data) >= -1)
        assert(torch.max(batch.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(batch)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, shuffle=True):
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, hparams.sort_by_length)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio = load_wav_to_torch(filename, self.sampling_rate)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            # here melspec should be 4 dimensions
            # but for tacotron2 they need it to be 3 dimensions
            # we need it to be 4 dimensions for VQ VAE and VAE
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        # obtain the maximum input length
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec with extra single zero vector to mark the end
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch]) + 1
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            # print("==== Mel size ========")
            # print(mel.size())
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1):] = 1
            output_lengths[i] = mel.size(1)
        
        # print(mel_padded)
        # print(mel_padded.size())

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths


"""
BSD 3-Clause License

Copyright (c) 2017, Prem Seetharaman
All rights reserved.

* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""



class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(win_length >= filter_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

def sanity_check(model, c, g):
    if model.has_speaker_embedding():
        if g is None:
            raise RuntimeError(
                "WaveNet expects speaker embedding, but speaker-id is not provided")
    else:
        if g is not None:
            raise RuntimeError(
                "WaveNet expects no speaker embedding, but speaker-id is provided")

    if model.local_conditioning_enabled():
        if c is None:
            raise RuntimeError("WaveNet expects conditional features, but not given")
    else:
        if c is not None:
            raise RuntimeError("WaveNet expects no conditional features, but given")


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0, constant_values=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=constant_values)
    return x


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root, col, speaker_id=None,
                 train=True, test_size=0.05, test_num_samples=None, random_state=1234):
        self.data_root = data_root
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state

    def interest_indices(self, paths):
        indices = np.arange(len(paths))
        if self.test_size is None:
            test_size = self.test_num_samples / len(paths)
        else:
            test_size = self.test_size
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=self.random_state)
        return train_indices if self.train else test_indices

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4 or len(l) == 5
        self.multi_speaker = len(l) == 5
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))

        paths_relative = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.data_root, f), paths_relative))

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # Filter by train/tset
                indices = self.interest_indices(paths)
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])

                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

                return paths

        # Filter by train/test
        indices = self.interest_indices(paths)
        paths = list(np.array(paths)[indices])
        lengths_np = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, lengths_np))

        if self.multi_speaker:
            speaker_ids_np = list(np.array(self.speaker_ids)[indices])
            self.speaker_ids = list(map(int, speaker_ids_np))
            assert len(paths) == len(self.speaker_ids)

        return paths

    def collect_features(self, path):
        return np.load(path)


class RawAudioDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(RawAudioDataSource, self).__init__(data_root, 0, **kwargs)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(MelSpecDataSource, self).__init__(data_root, 1, **kwargs)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler
    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class PyTorchDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.Mel is None:
            mel = None
        else:
            mel = self.Mel[idx]

        raw_audio = self.X[idx]
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        # (x,c,g)
        return raw_audio, mel, speaker_id

    def __len__(self):
        return len(self.X)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def clone_as_averaged_model(device, model, ema):
    assert ema is not None
    averaged_model = build_model().to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(target)
        losses = self.criterion(input, target)
        return ((losses * mask_).sum()) / mask_.sum()


class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self):
        super(DiscretizedMixturelogisticLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, 1)
        mask_ = mask.expand_as(target)

        losses = discretized_mix_logistic_loss(
            input, target, num_classes=hparams.quantize_channels,
            log_scale_min=hparams.log_scale_min, reduce=False)
        assert losses.size() == target.size()
        return ((losses * mask_).sum()) / mask_.sum()


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


def collate_fn(batch):
    """Create batch
    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T,)
            - x[1] (ndarray,int) : list of (T, D)
            - x[2] (ndarray,int) : list of (1,), speaker id
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2 and hparams.cin_channels > 0
    global_conditioning = len(batch[0]) >= 3 and hparams.gin_channels > 0

    if hparams.max_time_sec is not None:
        max_time_steps = int(hparams.max_time_sec * hparams.sample_rate)
    elif hparams.max_time_steps is not None:
        max_time_steps = hparams.max_time_steps
    else:
        max_time_steps = None

    # Time resolution adjustment
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            if hparams.upsample_conditional_features:
                assert_ready_for_upsampling(x, c)
                if max_time_steps is not None:
                    max_steps = ensure_divisible(max_time_steps, audio.get_hop_size(), True)
                    if len(x) > max_steps:
                        max_time_frames = max_steps // audio.get_hop_size()
                        s = np.random.randint(0, len(c) - max_time_frames)
                        ts = s * audio.get_hop_size()
                        x = x[ts:ts + audio.get_hop_size() * max_time_frames]
                        c = c[s:s + max_time_frames, :]
                        assert_ready_for_upsampling(x, c)
            else:
                x, c = audio.adjust_time_resolution(x, c)
                if max_time_steps is not None and len(x) > max_time_steps:
                    s = np.random.randint(0, len(x) - max_time_steps)
                    x, c = x[s:s + max_time_steps], c[s:s + max_time_steps, :]
                assert len(x) == len(c)
            new_batch.append((x, c, g))
        batch = new_batch
    else:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            x = audio.trim(x)
            if max_time_steps is not None and len(x) > max_time_steps:
                s = np.random.randint(0, len(x) - max_time_steps)
                if local_conditioning:
                    x, c = x[s:s + max_time_steps], c[s:s + max_time_steps, :]
                else:
                    x = x[s:s + max_time_steps]
            new_batch.append((x, c, g))
        batch = new_batch

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # (B, T, C)
    # pad for time-axis
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels)
        x_batch = np.array([_pad_2d(np_utils.to_categorical(
            x[0], num_classes=hparams.quantize_channels),
            max_input_len, padding_value) for x in batch], dtype=np.float32)
    else:
        x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len)
                            for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels)
        y_batch = np.array([_pad(x[0], max_input_len, constant_values=padding_value) for x in batch], dtype=np.int)
    else:
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    # (B, T, D)
    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T)
        c_batch = torch.FloatTensor(c_batch).transpose(1, 2).contiguous()
    else:
        c_batch = None

    if global_conditioning:
        g_batch = torch.LongTensor([x[2] for x in batch])
    else:
        g_batch = None

    # Covnert to channel first i.e., (B, C, T)
    x_batch = torch.FloatTensor(x_batch).transpose(1, 2).contiguous()
    # Add extra axis
    if is_mulaw_quantize(hparams.input_type):
        y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()
    else:
        y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, y_batch, c_batch, g_batch, input_lengths


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def get_audio_data_loaders(data_root, speaker_id, test_shuffle=True):
    data_loaders = {}
    local_conditioning = hparams.cin_channels > 0
    for phase in ["train", "test"]:
        train = phase == "train"
        X = FileSourceDataset(RawAudioDataSource(data_root, speaker_id=speaker_id,
                                                 train=train,
                                                 test_size=hparams.test_size,
                                                 test_num_samples=hparams.test_num_samples,
                                                 random_state=hparams.random_state))
        if local_conditioning:
            Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id=speaker_id,
                                                      train=train,
                                                      test_size=hparams.test_size,
                                                      test_num_samples=hparams.test_num_samples,
                                                      random_state=hparams.random_state))
            assert len(X) == len(Mel)
            print("Local conditioning enabled. Shape of a sample: {}.".format(
                Mel[0].shape))
        else:
            Mel = None
        print("[{}]: length of the dataset is {}".format(phase, len(X)))

        if train:
            lengths = np.array(X.file_data_source.lengths)
            # Prepare sampler
            sampler = PartialyRandomizedSimilarTimeLengthSampler(
                lengths, batch_size=hparams.batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = test_shuffle

        dataset = PyTorchDataset(X, Mel)
        data_loader = data_utils.DataLoader(
            dataset, batch_size=hparams.batch_size,
            num_workers=hparams.num_workers, sampler=sampler, shuffle=shuffle,
            collate_fn=collate_fn, pin_memory=hparams.pin_memory)

        speaker_ids = {}
        for idx, (x, c, g) in enumerate(dataset):
            if g is not None:
                try:
                    speaker_ids[g] += 1
                except KeyError:
                    speaker_ids[g] = 1
        if len(speaker_ids) > 0:
            print("Speaker stats:", speaker_ids)

        data_loaders[phase] = data_loader

    return data_loaders

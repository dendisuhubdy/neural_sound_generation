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
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.distributed import DistributedSampler

import random
import numpy as np
import math
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os.path import dirname, join, expanduser
from scipy.signal import get_window
from sklearn.model_selection import train_test_split
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

import audio_tacotron as audio
from audio_tacotron import is_mulaw_quantize, is_mulaw, is_raw
from features import dynamic_range_compression
from features import window_sumsquare, dynamic_range_compression
from hparams_tacotron import hparams


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


def get_audio_data_loaders(data_root, speaker_id, batch_size, test_shuffle=True):
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
                lengths, batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = test_shuffle

        dataset = PyTorchDataset(X, Mel)
        data_loader = data_utils.DataLoader(
            dataset, batch_size,
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


def test_load_data():
    speaker_id = None
    data_root = str(sys.argv[1])
    batch_size = 12
    data_loaders = get_audio_data_loaders(data_root, speaker_id, batch_size, test_shuffle=True)
    return data_loaders


if __name__ == "__main__":
    test_load_data()

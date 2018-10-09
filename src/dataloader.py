"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import os
import subprocess
import math
import librosa
import numpy as np
import scipy.signal
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os.path import exists


import torch
import torchaudio
from torchvision import datasets, transforms
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .datasets import cmu_arctic, ljspeech
from .io import hts
from .preprocessing import *

from utils import (load_wav, melspectrogramm, get_hop_size,
                   lws_pad_lr, start_and_end_indices,
                   is_mulaw_quantize, is_mulaw, is_raw)


def load_training_data(args, kwargs):
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)('./data/{}/'.format(args.dataset),
                                       train=True, download=True,
                                       transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    return train_loader


def load_test_data(args, kwargs):
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)('./data/{}/'.format(args.dataset),
                                       train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    return test_loader


class ljpseech():
    def __init__(self):
        pass


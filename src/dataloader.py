"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from data_utils import TextMelLoader, TextMelCollate


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

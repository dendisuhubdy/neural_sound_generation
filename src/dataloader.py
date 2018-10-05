"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import torch
from torchvision import datasets, transforms

 
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

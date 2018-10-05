"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import argparse

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataloader import load_training_data, load_test_data
from models import VAE
from train import train
from test import test


def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # load training data
    train_loader = load_training_data(args, kwargs)
    # load test data
    test_loader = load_test_data(args, kwargs)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(args, model, optimizer, train_loader, device, epoch)
        test(args, model, test_loader, device, epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       '../results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()

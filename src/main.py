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
from models import VAE, VQVAE
from train import train_vae, train_vqvae
from test import test_vae, test_vqvae


def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr-rate', type=float, default=1e-3, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='dataset for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='vae', metavar='N',
                        help='model for training')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
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

    # here we can swap models to VAE, VQ-VAE, PixelCNN, PixelRNN
    if args.dataset == 'MNIST':
        input_dim = 1
        dim = 256
        z_dim = 128

    elif args.dataset == 'CIFAR10':
        input_dim = 3
        dim = 256
        z_dim = 512

    if args.model == 'vae':
        model = VAE(input_dim, dim, z_dim).to(device)
    elif args.model == 'vqvae':
        model = VQVAE(input_dim, dim, z_dim).to(device)
    print(model)

    # setup optimizer as Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)

    for epoch in range(1, args.epochs + 1):
        if args.model == 'vae':
            train_vae(args, model, optimizer, train_loader, device, epoch)
            test_vae(args, model, test_loader, device, epoch)
        elif args.model == 'vqvae':
            train_vqvae(args, model, optimizer, train_loader, device, epoch)
            test_vqvae(args, model, test_loader, device, epoch)

        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample, _ = model(sample)
            save_image(sample.view(64, 1, 28, 28),
                       './results/sample_' + str(epoch) + '.png')


if __name__ == "__main__":
    main()

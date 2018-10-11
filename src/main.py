"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import argparse

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from dataloader import (load_training_data, load_test_data,
                       prepare_dataloaders)
from models import DefaultVAE, VAE, VQVAE
from train import train, train_vae, train_vqvae
from test import test, test_vae, test_vqvae
from hparams import create_hparams
from utils import to_device


def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr-rate', type=float, default=1e-3, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='dataset for training')
    parser.add_argument('--datadir', type=str, default='./data/', metavar='N',
                        help='dataset directory for training')
    # remember to execute
    # `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`
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
                        help='contribution of commitment loss,\
                              between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def save_checkpoint(args, state):
    filename='./models/{}/checkpoint.pth.tar'.format(args.model)
    torch.save(state, filename)
    # if is_best:
        # shutil.copyfile(filename, './models/{}/model_best.pth.tar'.format(args.model))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'ljspeech':
        train_loader, test_loader, collate_fn = prepare_dataloaders(hparams)
        # train_loader, test_loader = prepare_dataloaders(hparams)

    else:
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
    elif args.dataset == 'ljspeech':
        input_dim = 1
        dim = 256
        z_dim = 128
    else:
        input_dim = 1
        dim = 256
        z_dim = 128

    if args.model == 'vae':
        model = VAE(input_dim, dim, z_dim).to(device)
        # model = DefaultVAE().to(device)
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
            # sample = torch.randn(64, 1, 28, 28).to(device)
            # sample, _ = next(iter(test_loader))
            text_padded, input_lengths, mel_padded, \
                    gate_padded, output_lengths  = next(iter(test_loader))
            # here we input mel padded into the model
            sample = to_device(mel_padded, device).float()
            if args.model == 'vae':
                reconstruction, _ = model(sample)
            elif args.model == 'vqvae':
                reconstruction, _, _ = model(sample)
            grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
            save_image(grid, './results/sample_' + str(args.model)\
                        + '_' + str(args._dataset) + '_' + str(epoch) + '.png')
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()})

if __name__ == "__main__":
    main()

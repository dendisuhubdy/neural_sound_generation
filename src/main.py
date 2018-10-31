"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import argparse
import os
import sys

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from dataloader import (load_training_data, load_test_data,
                        get_audio_data_loaders)
from models import DefaultVAE, VAE, VQVAE
from train import train, train_vae, train_vqvae
from test import test, test_vae, test_vqvae


def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr-rate', type=float, default=1e-3, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='dataset for training')
    parser.add_argument('--datadir', type=str,
                        default='./data/', metavar='N',
                        help='dataset directory for training')
    parser.add_argument('--sampledir', type=str,
                        default='./vae_samples/', metavar='N',
                        help='sample directories')
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
    parser.add_argument('--dim', type=int, default=256, metavar='S',
                        help='hidden layer width')
    parser.add_argument('--z-dim', type=int, default=128, metavar='S',
                        help='hidden layer size')
    # parser.add_argument('--hparams', type=str,
                        # required=False, help='comma separated name=value pairs')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def save_checkpoint(args, state):
    filename = './models/{}/checkpoint_{}.pth.tar'.format(args.model, args.dataset)
    torch.save(state, filename)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # hparams = create_hparams(args.hparams)
    # torch.backends.cudnn.enabled = hparams.cudnn_enabled
    # torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'ljspeech':
        # Dataloader setup
        speaker_id = None
        data_root = os.path.join(args.datadir, 'ljs_1024_256_80')
        audio_data_loaders = get_audio_data_loaders(data_root, speaker_id, test_shuffle=True)
        train_loader = audio_data_loaders["train"]
        test_loader = audio_data_loaders["test"]
        print("LJSpeech data loaded")

    else:
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        # load training data
        train_loader = load_training_data(args, kwargs)
        # load test data
        test_loader = load_test_data(args, kwargs)

    # here we can swap models to VAE, VQ-VAE, PixelCNN, PixelRNN
    if args.dataset == 'MNIST':
        input_dim = 1
        # dim = 256
        # z_dim = 128

    elif args.dataset == 'CIFAR10':
        input_dim = 3
        # dim = 256
        # z_dim = 512
    elif args.dataset == 'ljspeech':
        input_dim = 1
        # dim = 256
        # z_dim = 128
    else:
        input_dim = 1
        # dim = 256
        # z_dim = 128

    if args.model == 'vae':
        model = VAE(input_dim, args.dim, args.z_dim).to(device)
        # model = DefaultVAE().to(device)
    elif args.model == 'vqvae':
        model = VQVAE(input_dim, args.dim, args.z_dim).to(device)
    print(model)

    # setup optimizer as Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr_rate)
    
    last_epoch = 0
    try:
        # Train!
        for epoch in range(1, args.epochs + 1):
            if args.model == 'vae':
                train_vae(args, model, optimizer, train_loader, device, epoch)
                test_vae(args, model, test_loader, device, epoch)
            elif args.model == 'vqvae':
                train_vqvae(args, model, optimizer, train_loader, device, epoch)
                test_vqvae(args, model, test_loader, device, epoch)

            with torch.no_grad():
                # sample = torch.randn(64, 1, 28, 28).to(device)
                sample, _ = next(iter(test_loader))
                sample = sample.to(device)
                sample = sample.unsqueeze(1)
                # text_padded, input_lengths, mel_padded, \
                        # gate_padded, output_lengths  = next(iter(test_loader))
                # # here we input mel padded into the model
                # sample = unsqueeze_to_device(mel_padded, device).float()
                print("Evaluating samples")
                if args.model == 'vae':
                    reconstruction, _ = model(sample)
                elif args.model == 'vqvae':
                    reconstruction, _, _ = model(sample)
                np.save(os.path.join(args.sampledir, format(args.dataset),\
                            'reconstruction_' + str(args.model)\
                            + '_data_' + str(args.dataset)\
                            + '_dim_' + str(args.dim)\
                            + '_z_dim_' + str(args.z_dim)\
                            + '_epoch_' + str(epoch) + '.npy'),
                        reconstruction.cpu(),
                        allow_pickle=True)
                # grid_samples = make_grid(sample.cpu(), nrow=8, range=(-1, 1), normalize=True)
                # save_image(grid_samples,
                            # os.path.join(args.sampledir, format(args.dataset),\
                            # 'samples_' + str(args.model)\
                            # + '_data_' + str(args.dataset)\
                            # + '_dim_' + str(args.dim)\
                            # + '_z_dim_' + str(args.z_dim)\
                            # + '_epoch_' + str(epoch) + '.png'))
                # grid_reconstruction = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
                # save_image(grid_reconstruction, 
                            # os.path.join(args.sampledir, format(args.dataset),\
                            # 'reconstruction_' + str(args.model)\
                            # + '_data_' + str(args.dataset)\
                            # + '_dim_' + str(args.dim)\
                            # + '_z_dim_' + str(args.z_dim)\
                            # + '_epoch_' + str(epoch) + '.png'))
                last_epoch = epoch

                save_checkpoint(args, {
                    'epoch': last_epoch,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()})
        except KeyboardInterrupt:
            print("Interrupted!")
            pass
        finally:
            save_checkpoint(args, {
                'epoch': last_epoch,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()})


if __name__ == "__main__":
    main()

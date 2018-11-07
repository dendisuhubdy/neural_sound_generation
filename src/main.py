"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import argparse
import os
import sys
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from dataloader import (load_training_data, load_test_data,
                        get_audio_data_loaders)
from models import DefaultVAE, VAE, VQVAE
from train import train, train_vae, train_vqvae
from test import test, test_vae, test_vqvae
from audio_tacotron import inv_mel_spectrogram, save_wav


def parse_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=36, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--lr-rate', type=float, default=1e-3, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='dataset for training')
    parser.add_argument('--datadir', type=str,
                        default='./data/', metavar='N',
                        help='dataset directory for training')
    parser.add_argument('--sampledir', type=str,
                        default='./results/', metavar='N',
                        help='sample directories')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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
    parser.add_argument('--dim', type=int, default=1, metavar='S',
                        help='hidden layer width')
    parser.add_argument('--z-dim', type=int, default=512, metavar='S',
                        help='hidden layer size')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def save_checkpoint(args, state):
    filename = './models/{}/checkpoint_{}_{}_{}.pth.tar'.format(args.model,
                                                             args.dataset,
                                                             args.dim,
                                                             args.z_dim)
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
        audio_data_loaders = get_audio_data_loaders(data_root,
                                                    speaker_id,
                                                    args.batch_size,
                                                    test_shuffle=True)
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
                test_data_iterator = iter(test_loader)
                x, y, c, g, input_lengths = next(test_data_iterator)
                # Prepare data
                x, y = x.to(device), y.to(device)
                input_lengths = input_lengths.to(device)
                c = c.to(device) if c is not None else None
                g = g.to(device) if g is not None else None
                c = c.unsqueeze(1)
                print("Evaluating samples")
                if args.model == 'vae':
                    reconstruction, _ = model(c)
                elif args.model == 'vqvae':
                    reconstruction, _, _ = model(c)
                reconstruction = reconstruction.squeeze(1)
                reconstruction = reconstruction.cpu()
                reconstruction = reconstruction.numpy()
                np.save(os.path.join(args.sampledir, format(args.dataset),\
                            'reconstruction_' + str(args.model)\
                            + '_data_' + str(args.dataset)\
                            + '_dim_' + str(args.dim)\
                            + '_z_dim_' + str(args.z_dim)\
                            + '_epoch_' + str(epoch) + '.npy'),
                            reconstruction,
                            allow_pickle=True)
                
                print("Trying audio reconstruction on test set..")

                mel_concat = None

                for batch_idx, mel in enumerate(reconstruction):
                    if batch_idx == 0:
                        mel_concat = mel
                    else:
                        mel_concat = np.concatenate((mel_concat, mel), axis=1)
                
                print(mel_concat.shape)

                sampling_rate = 22050
                fft_size = 1024
                hop_size = 256 # overlap window
                n_mels = 80 # number of melcepstrum coefficients (log scale)

                assert mel_concat.shape[0] == n_mels
                    
                signal = inv_mel_spectrogram(mel,
                                             sampling_rate,
                                             fft_size,
                                             hop_size,
                                             n_mels)

                save_wav(signal, os.path.join(args.sampledir,format(args.dataset),\
                            'audio_recon_' + str(args.model)\
                            + '_data_' + str(args.dataset)\
                            + '_dim_' + str(args.dim)\
                            + '_z_dim_' + str(args.z_dim)\
                            + '_epoch_' + str(epoch)\
                            + '_fftsize_' + str(fft_size)\
                            + '_hopsize_' + str(hop_size)\
                            + '.wav'))
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

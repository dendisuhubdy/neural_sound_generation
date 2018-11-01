"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
# import tqdm
from tqdm import tqdm  # , trange
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from loss import binary_cross_entropy, mse_loss
from models import to_scalar
from util import unsqueeze_to_device


def train(args, model, optimizer, train_loader, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        text_padded, input_lengths, mel_padded, \
                gate_padded, output_lengths = batch
        # here we input mel padded into the model
        data = unsqueeze_to_device(mel_padded, device).float()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = binary_cross_entropy(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def train_vae(args, model, optimizer, train_loader, device, epoch):
    model.train()
    train_loss = 0
    if args.dataset == 'ljspeech':
        # for batch_idx, batch in enumerate(train_loader):
        for batch_idx, (x, y, c, g, input_lengths) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            # Prepare data
            # x : (B, C, T) raw audio
            # y : (B, T, 1) text
            # c : (B, C, T) melspectrogram
            # g : (B,) speaker ID
            x, y = x.to(device), y.to(device)
            input_lengths = input_lengths.to(device)
            c = c.to(device) if c is not None else None
            g = g.to(device) if g is not None else None
            # fetch melspectrogram into the autoencoder
            c = c.unsqueeze(1)
            x_tilde, kl_d = model(c)
            # hackish operation
            target = torch.zeros(c.size(0), c.size(1), c.size(2), c.size(3))
            target[:, :, :, :x_tilde.size(3)] = x_tilde
            target = target.to(device)
            # print("new target size")
            # print(target.size())
            # the reason we do this hack is this error
            # RuntimeError: input and target shapes do not match: input [32 x 1 x 80 x 408]
            # target [32 x 1 x 80 x 411]
            # so we basically pad the output or truncate it so it has the same size as
            # the input data
            loss = mse_loss(target, c, kl_d)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(c), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(c)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

    else:
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, kl_d = model(data)
            loss = mse_loss(recon_batch, data, kl_d)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


def train_vqvae(args, model, optimizer, train_loader, device, epoch):
    model.train()
    train_loss = 0
    if args.dataset == 'ljspeech':
        for batch_idx, (x, y, c, g, input_lengths) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            # Prepare data
            x, y = x.to(device), y.to(device)
            input_lengths = input_lengths.to(device)
            c = c.to(device) if c is not None else None
            g = g.to(device) if g is not None else None
            c = c.unsqueeze(1)
            x_tilde, z_e_x, z_q_x = model(c)
            # hackish operation
            target = torch.zeros(c.size(0), c.size(1), c.size(2), c.size(3))
            target[:, :, :, :x_tilde.size(3)] = x_tilde
            target = target.to(device)
            # print("new target size")
            # print(target.size())
            # the reason we do this hack is this error
            # RuntimeError: input and target shapes do not match: input [32 x 1 x 80 x 408]
            # target [32 x 1 x 80 x 411]
            # Reconstruction loss
            # so we basically pad the output or truncate it so it has the same size as
            # the input data
            loss_recons = F.mse_loss(target, c)
            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            loss = loss_recons + loss_vq + args.beta * loss_commit
            loss.backward()
            optimizer.step()

            train_loss = loss_recons.item() + loss_vq.item()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(c), len(train_loader.dataset),
                    args.log_interval * batch_idx / len(train_loader),
                    train_loss
                    ))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

    else:
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            # print("============ Input data size =========")
            # print(data.size())
            optimizer.zero_grad()
            x_tilde, z_e_x, z_q_x = model(data)
            # print("============ Output data size =========")
            # print(x_tilde.size())
            # Reconstruction loss
            loss_recons = F.mse_loss(x_tilde, data)
            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            loss = loss_recons + loss_vq + args.beta * loss_commit
            loss.backward()
            optimizer.step()

            train_loss = loss_recons.item() + loss_vq.item()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    args.log_interval * batch_idx / len(train_loader),
                    train_loss
                    ))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

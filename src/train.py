"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from loss import binary_cross_entropy, mse_loss
from models import to_scalar


def train_vae(args, model, optimizer, train_loader, device, epoch):
    model.train()
    train_loss = 0
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
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(data)
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

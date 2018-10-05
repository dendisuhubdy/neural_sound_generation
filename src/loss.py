"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""
import torch
from torch.nn import functional as F


def binary_cross_entropy(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def mse_loss(x_tilde, x, kl_d):
    loss_recons = F.mse_loss(x_tilde, x, size_average=False) / x.size(0)
    loss = loss_recons + kl_d
    # nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
    # log_px = nll.mean().item() - np.log(128) + kl_d.item()
    # log_px /= np.log(2)
    return loss

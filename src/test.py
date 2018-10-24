import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from loss import binary_cross_entropy, mse_loss
from models import to_scalar
from utils import unsqueeze_to_device


def test(args, model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            text_padded, input_lengths, mel_padded, \
                    gate_padded, output_lengths = batch
            data = unsqueeze_to_device(mel_padded, device).float()
            recon_batch, mu, logvar = model(data)
            test_loss += binary_cross_entropy(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def test_vae(args, model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        if args.dataset == 'ljspeech':
            for batch_idx, batch in enumerate(test_loader):
                text_padded, input_lengths, mel_padded, \
                        gate_padded, output_lengths = batch
                data = unsqueeze_to_device(mel_padded, device).float()
                data = data.to(device)
                recon_batch, kl_d = model(data)
                target = torch.zeros(data.size(0), data.size(1), data.size(2), data.size(3))
                target[:, :, :, :recon_batch.size(3)] = recon_batch
                target = target.to(device)
                loss = mse_loss(target, data, kl_d)
                test_loss += loss.item()
                # if batch_idx == 0:
                    # n = min(data.size(0), 8)
                    # comparison = torch.cat([data[:n],
                                          # recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                    # save_image(.cpu(),
                             # './results/reconstruction_' + str(epoch) + '.png', nrow=n)
        else:
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, kl_d = model(data)
                loss = mse_loss(recon_batch, data, kl_d)
                test_loss += loss.item()
                # if i == 0:
                    # n = min(data.size(0), 8)
                    # comparison = torch.cat([data[:n],
                                          # recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                    # save_image(comparison.cpu(),
                             # './results/reconstruction_' + str(args.model)  + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


def test_vqvae(args, model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        if args.dataset == 'ljspeech':
            loss_recons, loss_vq = 0., 0.
            for batch_idx, batch in enumerate(test_loader):
                text_padded, input_lengths, mel_padded, \
                        gate_padded, output_lengths = batch
                data = unsqueeze_to_device(mel_padded, device).float()
                data = data.to(device)
                x_tilde, z_e_x, z_q_x = model(data)
                target = torch.zeros(data.size(0), data.size(1), data.size(2), data.size(3))
                target[:, :, :, :x_tilde.size(3)] = x_tilde
                target = target.to(device)
                loss_recons += F.mse_loss(target, data)
                loss_vq += F.mse_loss(z_q_x, z_e_x)
            loss_recons /= len(test_loader)
            loss_vq /= len(test_loader)
        else:
            loss_recons, loss_vq = 0., 0.
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                x_tilde, z_e_x, z_q_x = model(data)
                loss_recons += F.mse_loss(x_tilde, data)
                loss_vq += F.mse_loss(z_q_x, z_e_x)
            loss_recons /= len(test_loader)
            loss_vq /= len(test_loader)
        
        test_loss = loss_recons + loss_vq
        print('====> Test set loss: {:.4f}'.format(test_loss))

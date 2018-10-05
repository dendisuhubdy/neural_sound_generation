import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from loss import binary_cross_entropy, mse_loss
from models import to_scalar

def test_vae(args, model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, kl_d = model(data)
            loss = mse_loss(recon_batch, data, kl_d)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         './results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def test_vqvae(args, model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
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

from typing import Literal

import torch


class TotalVariationLoss(torch.nn.Module):
    """
    Total Variation Loss that matches TV between real and reconstructed images.
    Option 2: Match TV between real and reconstruction (preserve edge structure)
    """

    def __init__(self,
                 reconstruction_loss: Literal['mse', 'huber'] = 'mse',
                 tv_weight: float = 1.0):
        super().__init__()
        self.tv_weight = tv_weight

        if reconstruction_loss == 'mse':
            self.loss_fn = torch.nn.functional.mse_loss
        elif reconstruction_loss == 'huber':
            self.loss_fn = torch.nn.functional.huber_loss
        else:
            raise ValueError(f"Reconstruction loss {reconstruction_loss} not supported")

    def total_variation(self, x):
        """
        Compute total variation for arbitrary spatial dimensions.
        x: [batch, channels, *spatial_dims]
        Returns: [batch] - TV value for each sample in the batch
        """
        tv = 0
        # Start from dimension 2 (after batch and channel)
        for dim in range(2, x.dim()):
            # Create slices for neighboring differences along this dimension
            slice_1 = [slice(None)] * x.dim()
            slice_2 = [slice(None)] * x.dim()

            slice_1[dim] = slice(1, None)    # [1:]
            slice_2[dim] = slice(None, -1)   # [:-1]

            # Compute absolute difference along this dimension
            diff = torch.abs(x[tuple(slice_1)] - x[tuple(slice_2)])
            # Sum over channels and spatial dimensions, keep batch dimension
            tv += torch.sum(diff, dim=tuple(range(1, diff.dim())))

        return tv

    def forward(self, x_real, x_recon):
        """
        Compute TV loss between real and reconstructed images.

        Args:
            x_real: Real images [batch, channels, *spatial_dims]
            x_recon: Reconstructed images [batch, channels, *spatial_dims]

        Returns:
            total_loss: Weighted TV loss
            logs: Dictionary with loss components
        """
        # Compute TV for both real and reconstructed
        # Each returns [batch_size] tensor with TV value per sample
        tv_real = self.total_variation(x_real)
        tv_recon = self.total_variation(x_recon)

        # Match TV between real and reconstruction per sample
        tv_loss = self.loss_fn(tv_recon, tv_real, reduction='mean')

        # Scale by weight
        total_loss = self.tv_weight * tv_loss

        logs = {
            'tv_loss': tv_loss.item(),
            'tv_real_mean': tv_real.mean().item(),
            'tv_recon_mean': tv_recon.mean().item(),
            'tv_real_std': tv_real.std().item(),
            'tv_recon_std': tv_recon.std().item(),
            'total_tv_loss': total_loss.item()
        }

        return total_loss, logs

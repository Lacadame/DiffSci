import torch.nn as nn
import torch

class GaussianWeightedMSELoss(nn.Module):
    """
    A custom MSE loss that applies a Gaussian weight mask.
    This version has reduction='none' implicitly to match the KarrasModule framework.
    """
    def __init__(self, W, H, focus_radius, device='cpu'):
        super(GaussianWeightedMSELoss, self).__init__()
        self.W = W
        self.H = H
        self.focus_radius = focus_radius
        weight_mask = self._create_gaussian_window(W, H, focus_radius)
        self.register_buffer('weight_mask', weight_mask.to(device))

    def _create_gaussian_window(self, W, H, radius):
        sigma = self.focus_radius + 1e-8
        coords_x = torch.linspace(-1, 1, W)
        coords_y = torch.linspace(-1, 1, H)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        distance_squared = grid_x**2 + grid_y**2
        gaussian_weights = torch.exp(-distance_squared / (2 * sigma**2))
        return gaussian_weights.unsqueeze(0).unsqueeze(0)

    def forward(self, input, target):
        """
        Calculates the weighted squared error without reduction.
        """
        if self.weight_mask.device != input.device:
            self.weight_mask = self.weight_mask.to(input.device)
        
        squared_error = (input - target) ** 2
        
        # Apply the weights and return the resulting tensor.
        # The final .mean() will be called inside the KarrasModule.loss_fn
        return squared_error * self.weight_mask

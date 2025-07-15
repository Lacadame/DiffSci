import torch
import torch.nn as nn
from typing import Tuple

class GaussianWeightedMSELoss(nn.Module):
    """
    A general-purpose Mean Squared Error (MSE) loss that applies a Gaussian
    weight mask to N-dimensional spatial data. This allows the model to focus
    more on the center of the spatial dimensions (e.g., D, H, W).

    """
    def __init__(self, shape: Tuple[int, ...], focus_radius: float, device: str = 'cpu'):
        """
        Initializes the GaussianWeightedMSELoss module.

        Args:
            shape (Tuple[int, ...]): A tuple representing the spatial dimensions
                of the input tensors (e.g., (H, W) for 2D, or (D, H, W) for 3D).
            focus_radius (float): A parameter controlling the standard deviation
                of the Gaussian. A smaller value creates a sharper focus.
            device (str): The device to store the weight mask on ('cpu' or 'cuda').
        """
        super(GaussianWeightedMSELoss, self).__init__()
        self.shape = shape
        self.focus_radius = focus_radius
        
        # Create the N-dimensional Gaussian weight mask and register it as a buffer.
        weight_mask = self._create_gaussian_window(shape, focus_radius)
        self.register_buffer('weight_mask', weight_mask.to(device))

    def _create_gaussian_window(self, shape: Tuple[int, ...], radius: float) -> torch.Tensor:
        """
        Creates an N-dimensional Gaussian distribution to be used as a weight mask.
        
        Args:
            shape (Tuple[int, ...]): The spatial dimensions of the window.
            radius (float): The radius which controls the sigma of the Gaussian.

        Returns:
            torch.Tensor: A tensor of shape [1, 1, *shape] containing the
                          Gaussian weights, with values between 0 and 1.
        """
        # The standard deviation (sigma) of the Gaussian.
        # CORRECTED: Now uses the 'radius' parameter passed to the method.
        sigma = radius + 1e-8

        # Create a 1D coordinate tensor for each spatial dimension
        coords = [torch.linspace(-1, 1, s) for s in shape]
        
        # Create an N-dimensional coordinate grid
        # The 'ij' indexing ensures the grid axes match the tensor dimensions order
        grids = torch.meshgrid(*coords, indexing='ij')
        
        # Calculate the squared Euclidean distance from the center (0,0,...,0)
        distance_squared = torch.zeros_like(grids[0])
        for grid in grids:
            distance_squared += grid**2
        
        # Calculate the N-dimensional Gaussian
        gaussian_weights = torch.exp(-distance_squared / (2 * sigma**2))
        
        # Reshape to [1, 1, *shape] to be broadcastable with input tensors [B, C, *shape]
        # e.g., for 3D data, this will be [1, 1, D, H, W]
        final_shape = (1, 1) + shape
        return gaussian_weights.view(final_shape)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted squared error without reduction. This makes it
        compatible with frameworks like the KarrasModule that apply other
        weights before the final reduction.

        Args:
            input (torch.Tensor): The predicted tensor. Shape: [B, C, *shape].
            target (torch.Tensor): The ground truth tensor. Shape: [B, C, *shape].

        Returns:
            torch.Tensor: A tensor of shape [B, C, *shape] containing the
                          weighted squared errors.
        """
        # Ensure the weight mask is on the same device as the input tensor
        if self.weight_mask.device != input.device:
            self.weight_mask = self.weight_mask.to(input.device)

        # 1. Calculate the standard element-wise squared error
        squared_error = (input - target) ** 2
        
        # 2. Apply the Gaussian weights by element-wise multiplication.
        # The weight_mask [1, 1, *shape] will be broadcast across the
        # batch and channel dimensions of the squared_error [B, C, *shape].
        weighted_squared_error = squared_error * self.weight_mask
        
        # 3. Return the weighted error tensor without reduction
        return weighted_squared_error


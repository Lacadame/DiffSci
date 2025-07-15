import torch.nn as nn
import torch


class GaussianWeightedMSELoss(nn.Module):
    """
    A custom MSE loss that applies a Gaussian weight mask.
    This version has reduction='none' implicitly to match the KarrasModule framework.

    We also assume that the tensors will have two types of shape dimensions. The ones we call spatial shapes and
    non-spatial shapes. It is assumes that the tensors will have a certain ordering in such shapes in the form

    tensor.shape = (*non_spatial_shapes, *spatial_shapes)
    """
    def __init__(self, spatial_shape_array, focus_radius, num_non_spatial_shapes: int = 2, device='cpu'):
        super(GaussianWeightedMSELoss, self).__init__()
        self.spatial_shape_array = spatial_shape_array
        self.focus_radius = focus_radius
        self.num_non_spatial_shapes = num_non_spatial_shapes
        weight_mask = self._create_gaussian_window(spatial_shape_array, focus_radius)
        self.register_buffer('weight_mask', weight_mask.to(device))

    def _create_gaussian_window(self, shape_array, radius):
        sigma = radius + 1e-8
        coords = []
        for shape in shape_array:
            coords.append(torch.linspace(-1, 1, shape))
        grid = torch.meshgrid(coords, indexing='ij')
        distance_squared = torch.zeros(grid[0].shape)
        for i in range(shape_array):
            distance_squared += grid[i]**2
        gaussian_weights = torch.exp(-distance_squared / (2 * sigma**2))
        for _ in range(self.num_non_spatial_shapes):
            gaussian_weights.unsqueeze_(0)

        return gaussian_weights

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

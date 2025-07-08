import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict


def smoothstep_window(length, border, device=None):
    """
    Returns a 1D window of size `length` that is 1 in the interior and tapers smoothly to 0 at the border of width `border`.
    Uses a cubic smoothstep: 3x^2 - 2x^3.
    """
    if border == 0:
        return torch.ones(length, device=device)
    idx = torch.arange(length, device=device, dtype=torch.float32)
    win = torch.ones(length, device=device, dtype=torch.float32)
    # Left border
    left = idx < border
    x = idx[left] / border
    win[left] = 3 * x**2 - 2 * x**3
    # Right border
    right = idx >= (length - border)
    x = (length - idx[right] - 1) / border
    win[right] = 3 * x**2 - 2 * x**3
    return win


class EdgeDetectionPreprocessor(nn.Module):
    """
    Differentiable edge detection preprocessing for distance transform discriminators.
    Designed to be easily adaptable from 2D to 3D.
    """

    def __init__(
        self,
        dim=2,
        processors='all',
        feature_weights: Dict[str, float] = None,
        border_width=8,
    ):
        super().__init__()
        self.dim = dim
        self.border_width = border_width  # Number of pixels for border window
        valid_processors = ['original', 'sobel', 'laplacian', 'gradient', 'morph']
        if processors == 'all':
            self.processors = valid_processors
        elif isinstance(processors, str):
            if processors not in valid_processors:
                raise ValueError(f"Unknown processor: {processors}")
            self.processors = [processors]
        else:
            for p in processors:
                if p not in valid_processors:
                    raise ValueError(f"Unknown processor: {p}")
            self.processors = list(processors)

        # Handle feature weights and precompute normalized weights for selected processors
        if feature_weights is None:
            feature_weights = {k: 1.0 for k in valid_processors}
        selected_weights = [float(feature_weights.get(p, 1.0)) for p in self.processors]
        total_weight = sum(selected_weights)
        if total_weight == 0:
            self.normalized_weights = {p: 0.0 for p in self.processors}
        else:
            self.normalized_weights = {p: w / total_weight for p, w in zip(self.processors, selected_weights)}

        # Register edge detection kernels as buffers (non-trainable parameters)
        self._register_kernels()

    def _register_kernels(self):
        """Register differentiable edge detection kernels"""

        if self.dim == 2:
            # 2D Sobel kernels
            sobel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=torch.float32)

            # 2D Laplacian kernel
            laplacian = torch.tensor([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]], dtype=torch.float32)

            # 2D Structure tensor kernels (for coherence detection)
            struct_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32)
            struct_y = torch.tensor([[-1], [0], [1]], dtype=torch.float32)

        else:  # 3D
            # 3D Sobel kernels (6 directions)
            sobel_x = torch.tensor([[[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]],
                                  [[-2, 0, 2],
                                   [-4, 0, 4],
                                   [-2, 0, 2]],
                                  [[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]], dtype=torch.float32)

            sobel_y = torch.tensor([[[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]],
                                  [[-2, -4, -2],
                                   [0, 0, 0],
                                   [2, 4, 2]],
                                  [[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]]], dtype=torch.float32)

            sobel_z = torch.tensor([[[-1, -2, -1],
                                   [-2, -4, -2],
                                   [-1, -2, -1]],
                                  [[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]],
                                  [[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]], dtype=torch.float32)

            # 3D Laplacian kernel
            laplacian = torch.zeros(3, 3, 3, dtype=torch.float32)
            laplacian[1, 1, 1] = -6
            laplacian[0, 1, 1] = laplacian[2, 1, 1] = 1
            laplacian[1, 0, 1] = laplacian[1, 2, 1] = 1
            laplacian[1, 1, 0] = laplacian[1, 1, 2] = 1

            # 3D gradient kernels
            struct_x = torch.tensor([[[-1, 0, 1]]], dtype=torch.float32)
            struct_y = torch.tensor([[[-1], [0], [1]]], dtype=torch.float32)
            struct_z = torch.tensor([[[-1]], [[0]], [[1]]], dtype=torch.float32)

        # Register kernels as buffers
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))
        self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))
        self.register_buffer('struct_x', struct_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('struct_y', struct_y.unsqueeze(0).unsqueeze(0))

        if self.dim == 3:
            self.register_buffer('sobel_z', sobel_z.unsqueeze(0).unsqueeze(0))
            self.register_buffer('struct_z', struct_z.unsqueeze(0).unsqueeze(0))

    def _conv_nd(self, x, kernel, padding='same'):
        """N-dimensional convolution wrapper"""
        if self.dim == 2:
            return F.conv2d(x, kernel, padding=padding)
        else:
            return F.conv3d(x, kernel, padding=padding)

    def sobel_edges(self, x):
        """
        Sobel edge detection on distance fields.
        Returns gradient magnitude and direction information.
        """
        # Apply Sobel kernels
        grad_x = self._conv_nd(x, self.sobel_x)
        grad_y = self._conv_nd(x, self.sobel_y)

        if self.dim == 3:
            grad_z = self._conv_nd(x, self.sobel_z)
            magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
            # return torch.cat([magnitude, grad_x, grad_y, grad_z], dim=1)
        else:
            magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            # return torch.cat([magnitude, grad_x, grad_y], dim=1)
        return torch.cat([magnitude], dim=1)

    def laplacian_edges(self, x):
        """
        Laplacian edge detection - excellent for detecting ridges/valleys in distance fields.
        """
        laplacian_response = self._conv_nd(x, self.laplacian)

        # Return both raw response and absolute response
        # return torch.cat([laplacian_response, torch.abs(laplacian_response)], dim=1)
        return torch.cat([laplacian_response], dim=1)

    def gradient_magnitude(self, x):
        """
        Direct gradient magnitude - most natural for distance fields.
        This detects areas where distance changes rapidly.
        """
        grad_x = self._conv_nd(x, self.struct_x)
        grad_y = self._conv_nd(x, self.struct_y)

        if self.dim == 3:
            grad_z = self._conv_nd(x, self.struct_z)
            magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)

            # Also compute gradient coherence (how aligned gradients are)
            # coherence = self._gradient_coherence_3d(grad_x, grad_y, grad_z)
        else:
            magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

            # Compute gradient coherence in 2D
            # coherence = self._gradient_coherence_2d(grad_x, grad_y)
        # return torch.cat([magnitude, coherence], dim=1)
        return torch.cat([magnitude], dim=1)

    def morphological_gradient(self, x, kernel_size=3):
        """
        Morphological gradient - difference between dilation and erosion.
        Particularly good for detecting boundaries in distance fields.
        """
        # Create morphological kernel
        if self.dim == 2:
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)
            dilated = F.max_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)
            eroded = -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size//2)
        else:
            kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=x.device)
            dilated = F.max_pool3d(x, kernel_size, stride=1, padding=kernel_size//2)
            eroded = -F.max_pool3d(-x, kernel_size, stride=1, padding=kernel_size//2)

        morph_grad = dilated - eroded
        return morph_grad

    def _gradient_coherence_2d(self, grad_x, grad_y):
        """Compute gradient coherence in 2D"""
        # Structure tensor components
        Jxx = grad_x * grad_x
        Jxy = grad_x * grad_y
        Jyy = grad_y * grad_y

        # Gaussian smoothing of structure tensor
        sigma = 1.0
        kernel_size = 5
        gaussian_kernel = self._gaussian_kernel_2d(kernel_size, sigma).to(grad_x.device)

        Jxx_smooth = self._conv_nd(Jxx, gaussian_kernel)
        Jxy_smooth = self._conv_nd(Jxy, gaussian_kernel)
        Jyy_smooth = self._conv_nd(Jyy, gaussian_kernel)

        # Coherence measure
        trace = Jxx_smooth + Jyy_smooth
        det = Jxx_smooth * Jyy_smooth - Jxy_smooth * Jxy_smooth
        coherence = (trace - 2*torch.sqrt(det + 1e-8)) / (trace + 1e-8)

        return coherence

    def _gradient_coherence_3d(self, grad_x, grad_y, grad_z):
        """Compute gradient coherence in 3D"""
        # Simplified 3D coherence - can be made more sophisticated
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)

        # Compute local variance of gradient directions
        gx_norm = grad_x / (magnitude + 1e-8)
        gy_norm = grad_y / (magnitude + 1e-8)
        gz_norm = grad_z / (magnitude + 1e-8)

        # Simple coherence measure
        coherence = magnitude * (torch.abs(gx_norm) + torch.abs(gy_norm) + torch.abs(gz_norm))
        return coherence

    def _gaussian_kernel_2d(self, kernel_size, sigma):
        """Create 2D Gaussian kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2

        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()

        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        return kernel.unsqueeze(0).unsqueeze(0)

    def _gaussian_blur(self, x, sigma):
        """Apply Gaussian blur to the whole image (for border smoothing)."""
        if sigma is None or sigma <= 0:
            return x
        # Use torch.nn.functional.gaussian_blur if available (PyTorch >= 1.10)
        # Otherwise, use a custom implementation
        if self.dim == 2:
            # [B, C, H, W]
            kernel_size = int(4 * sigma + 1) | 1  # ensure odd
            return F.gaussian_blur(x, (kernel_size, kernel_size), (sigma, sigma))
        else:
            # [B, C, D, H, W]
            kernel_size = int(4 * sigma + 1) | 1
            return F.gaussian_blur(x, (kernel_size, kernel_size, kernel_size), (sigma, sigma, sigma))

    def _make_window(self, shape, device):
        """
        Create a N-dimensional smooth window: 1 in the interior, smoothstep at the border.
        shape: tuple of N dimensions (e.g. (H, W) for 2D or (D, H, W) for 3D)
        """
        # Create window for each dimension
        windows = []
        for size in shape:
            windows.append(smoothstep_window(size, self.border_width, device))

        # Start with first window
        window = windows[0]

        # Multiply with remaining windows, adding dimensions as needed
        for i, win in enumerate(windows[1:], 1):
            # Add singleton dimensions before and after current axis
            shape = [1] * len(shape)
            shape[i] = -1
            win = win.view(*shape)
            window = window.unsqueeze(i) * win

        return window

    def _apply_border_window(self, x):
        """
        Apply a window to the borders of x to reduce edge artifacts.
        x: [B, C, H, W] or [B, C, D, H, W]
        """
        if self.border_width is None or self.border_width <= 0:
            return x
        shape = x.shape[-2:] if self.dim == 2 else x.shape[-3:]
        window = self._make_window(shape, x.device)
        # Expand to [1, 1, ...] for broadcasting
        while window.dim() < x.dim():
            window = window.unsqueeze(0)
        return x * window

    def forward(self, x):
        """
        Apply selected edge detection methods and concatenate results.

        Args:
            x: Input distance transform [B, C, H, W] or [B, C, D, H, W]

        Returns:
            Edge features concatenated along channel dimension
        """
        features = []

        # Apply border window to a copy of x for edge features
        x_windowed = self._apply_border_window(x)

        # TODO: Remove the tanh that I will pass
        # x_windowed = torch.tanh(x_windowed)

        # Apply selected processors with precomputed normalized weights
        for p in self.processors:
            if p == 'original':
                features.append(x * self.normalized_weights[p])
            elif p == 'sobel':
                features.append(self.sobel_edges(x_windowed) * self.normalized_weights[p])
            elif p == 'laplacian':
                features.append(self.laplacian_edges(x_windowed) * self.normalized_weights[p])
            elif p == 'gradient':
                features.append(self.gradient_magnitude(x_windowed) * self.normalized_weights[p])
            elif p == 'morph':
                features.append(self.morphological_gradient(x_windowed) * self.normalized_weights[p])
        # Concatenate all features
        return torch.cat(features, dim=1)
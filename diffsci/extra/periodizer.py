"""
DiffusionPeriodizer: Wrapper for enforcing periodicity in diffusion model outputs.

This module implements the expand-crop-blend approach for generating periodic volumes
from a diffusion model trained on non-periodic data.

Strategy:
    1. Pre-processing: Expand input periodically by `pad` pixels on each side
    2. Forward pass: Run the network on the expanded input
    3. Post-processing: Crop back to original size, then blend boundaries with cosine weights

The cosine blending ensures smooth periodicity at boundaries even when the effective
receptive field extends beyond the padding.
"""

from typing import Tuple, Union, Optional, Callable
import torch
import torch.nn as nn

from diffsci.torchutils import periodic_getitem_extended


class DiffusionPeriodizer(nn.Module):
    """
    Wrapper that adds periodic boundary handling to any network.

    The network should have a forward method: [B, C, *spatial] -> [B, C, *spatial]
    where *spatial can be (H, W) for 2D or (H, W, D) for 3D.

    Parameters
    ----------
    net : nn.Module
        The network to wrap. Must have a forward method that preserves spatial dimensions.
    pad : int or tuple
        Padding size for periodic expansion. If int, same for all dimensions.
        For 3D: (pad_H, pad_W, pad_D). For 2D: (pad_H, pad_W).
    blend_width : int or tuple
        Width of the cosine blending zone at each boundary.
        If int, same for all dimensions.
    dimension : int
        Spatial dimension (2 or 3).

    Example
    -------
    >>> model = YourDiffusionModel()
    >>> periodizer = DiffusionPeriodizer(model, pad=32, blend_width=8, dimension=3)
    >>> # During sampling, wrap your denoising step:
    >>> x_periodic = periodizer(x, t=timestep)
    """

    def __init__(
        self,
        net: nn.Module,
        pad: Union[int, Tuple[int, ...]],
        blend_width: Union[int, Tuple[int, ...]] = 8,
        dimension: int = 3,
    ):
        super().__init__()
        self.net = net
        self.dimension = dimension

        # Normalize pad to tuple
        if isinstance(pad, int):
            self.pad = tuple([pad] * dimension)
        else:
            assert len(pad) == dimension, f"pad must have {dimension} elements"
            self.pad = tuple(pad)

        # Normalize blend_width to tuple
        if isinstance(blend_width, int):
            self.blend_width = tuple([blend_width] * dimension)
        else:
            assert len(blend_width) == dimension, f"blend_width must have {dimension} elements"
            self.blend_width = tuple(blend_width)

    def expand_periodic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand input tensor periodically by self.pad on each side.

        For a 3D tensor [B, C, H, W, D] with pad=(pH, pW, pD):
        - Output shape: [B, C, H+2*pH, W+2*pW, D+2*pD]
        - The center contains the original data
        - The borders contain periodic copies

        Uses periodic_getitem_extended which handles multi-period tiling.
        """
        # Get spatial dimensions (after batch and channel)
        spatial_shape = x.shape[2:]
        assert len(spatial_shape) == self.dimension, \
            f"Expected {self.dimension}D spatial input, got {len(spatial_shape)}D"

        # Build slices for periodic expansion
        # For each dim: slice(-pad, size + pad) extracts pad from end, full tensor, pad from start
        slices = [slice(None), slice(None)]
        for dim_idx, (size, p) in enumerate(zip(spatial_shape, self.pad)):
            # slice(-p, size + p) gives us: [last p elements] + [full tensor] + [first p elements]
            slices.append(slice(-p, size + p))

        x_expanded = periodic_getitem_extended(x, *slices)

        return x_expanded

    def crop_center(self, x: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Crop the center of the expanded tensor back to original spatial dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Expanded tensor [B, C, *expanded_spatial]
        original_shape : tuple
            Original spatial shape (H, W, D) or (H, W)

        Returns
        -------
        torch.Tensor
            Cropped tensor [B, C, *original_shape]
        """
        # Build slice for center crop
        slices = [slice(None), slice(None)]  # B, C dimensions
        for p, orig_size in zip(self.pad, original_shape):
            slices.append(slice(p, p + orig_size))

        return x[tuple(slices)]

    def cosine_blend_boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cosine blending at boundaries to ensure smooth periodicity.

        For each spatial dimension, we blend a strip at the start with the
        corresponding strip at the end, using cosine-weighted interpolation.

        The idea: for the output to be periodic, x[..., 0, :, :] should equal x[..., -1, :, :].
        We create a smooth transition by blending near-boundary values.

        Blending formula for position i in [0, blend_width):
            weight = 0.5 * (1 - cos(π * (i + 0.5) / blend_width))  # 0 at i=0, 1 at i=blend_width
            x[..., i, :, :] = weight * x[..., i, :, :] + (1-weight) * x[..., -(blend_width-i), :, :]

        This ensures:
            - At i=0: mostly takes from the opposite boundary
            - At i=blend_width-1: mostly keeps original value
            - The blend is symmetric: same treatment on both ends
        """
        x = x.clone()  # Don't modify in place

        for dim_idx, bw in enumerate(self.blend_width):
            if bw <= 0:
                continue

            spatial_dim = dim_idx + 2  # Account for B, C dimensions
            size = x.shape[spatial_dim]

            if bw * 2 >= size:
                # Blend width too large; skip this dimension
                continue

            # Create cosine weights
            # At position i: weight goes from ~0 (i=0) to ~1 (i=bw-1)
            positions = torch.arange(bw, device=x.device, dtype=x.dtype)
            weights = 0.5 * (1 - torch.cos(torch.pi * (positions + 0.5) / bw))

            # Reshape weights for broadcasting
            # For dim 2 (H in 3D): shape [1, 1, bw, 1, 1]
            # For dim 3 (W in 3D): shape [1, 1, 1, bw, 1]
            # For dim 4 (D in 3D): shape [1, 1, 1, 1, bw]
            weight_shape = [1] * x.dim()
            weight_shape[spatial_dim] = bw
            weights = weights.reshape(weight_shape)

            # Get slices for the start and end regions
            # Start region: indices [0, bw)
            # End region: indices [size-bw, size)
            start_slice = [slice(None)] * x.dim()
            start_slice[spatial_dim] = slice(0, bw)

            end_slice = [slice(None)] * x.dim()
            end_slice[spatial_dim] = slice(size - bw, size)

            # Get the strips
            start_strip = x[tuple(start_slice)]  # Shape: [B, C, ..., bw, ...]
            end_strip = x[tuple(end_slice)]      # Shape: [B, C, ..., bw, ...]

            # For periodicity: the start should match the end
            # Blend start with flipped end, and end with flipped start
            end_flipped = end_strip.flip(dims=[spatial_dim])
            start_flipped = start_strip.flip(dims=[spatial_dim])

            # Blend: at i=0 (weight~0): take from opposite side
            #        at i=bw-1 (weight~1): keep original
            new_start = weights * start_strip + (1 - weights) * end_flipped
            new_end = weights.flip(dims=[spatial_dim]) * end_strip + \
                      (1 - weights.flip(dims=[spatial_dim])) * start_flipped

            # Apply blended values
            x[tuple(start_slice)] = new_start
            x[tuple(end_slice)] = new_end

        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with periodic boundary handling.

        1. Expand input periodically
        2. Run network
        3. Crop to original size
        4. Apply cosine boundary blending

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [B, C, H, W, D] or [B, C, H, W]
        **kwargs
            Additional arguments passed to the network (e.g., t for timestep)

        Returns
        -------
        torch.Tensor
            Output with enforced periodicity, same shape as input
        """
        original_spatial_shape = x.shape[2:]

        # 1. Expand periodically
        x_expanded = self.expand_periodic(x)

        # 2. Run network
        y_expanded = self.net(x_expanded, *args, **kwargs)

        # 3. Crop to original size
        y_cropped = self.crop_center(y_expanded, original_spatial_shape)

        # 4. Apply cosine blending at boundaries
        y_periodic = self.cosine_blend_boundaries(y_cropped)

        return y_periodic

    def forward_no_blend(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass without the boundary blending step.
        Useful for comparison or when blending is not desired.
        """
        original_spatial_shape = x.shape[2:]
        x_expanded = self.expand_periodic(x)
        y_expanded = self.net(x_expanded, *args, **kwargs)
        y_cropped = self.crop_center(y_expanded, original_spatial_shape)
        return y_cropped

    def forward_expand_only(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass that returns the expanded output without cropping.
        Useful for debugging to see what the network produces on tiled input.
        """
        x_expanded = self.expand_periodic(x)
        y_expanded = self.net(x_expanded, *args, **kwargs)
        return y_expanded


class PeriodicSamplerWrapper:
    """
    Wrapper for diffusion samplers that applies periodization at each step.

    This wraps a sampler's denoising step to use DiffusionPeriodizer,
    ensuring periodicity is enforced throughout the sampling process.

    Parameters
    ----------
    sampler : object
        A diffusion sampler with a step() method
    periodizer : DiffusionPeriodizer
        The periodizer to apply at each step
    apply_every_n_steps : int
        Apply periodization every N steps (default 1 = every step)
        Using N > 1 can speed up sampling while still encouraging periodicity

    Example
    -------
    >>> periodizer = DiffusionPeriodizer(model, pad=32, blend_width=8)
    >>> # Manual sampling loop:
    >>> for t in timesteps:
    ...     x = periodizer(x, t=t)  # Each step enforces periodicity
    """

    def __init__(
        self,
        sampler,
        periodizer: DiffusionPeriodizer,
        apply_every_n_steps: int = 1,
    ):
        self.sampler = sampler
        self.periodizer = periodizer
        self.apply_every_n_steps = apply_every_n_steps
        self._step_count = 0

    def step(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform one sampling step with optional periodization.
        """
        self._step_count += 1

        if self._step_count % self.apply_every_n_steps == 0:
            # Use periodizer
            return self.periodizer(x, t=t, **kwargs)
        else:
            # Use original sampler
            return self.sampler.step(x, t, **kwargs)

    def reset(self):
        """Reset step counter (call before each new sample)."""
        self._step_count = 0


def measure_periodicity_error(x: torch.Tensor, dimension: int = 3) -> dict:
    """
    Measure how periodic a volume is by comparing opposite boundaries.

    Returns
    -------
    dict with:
        - 'mse_per_dim': MSE between opposite boundaries for each dimension
        - 'max_diff_per_dim': Maximum absolute difference per dimension
        - 'total_mse': Sum of MSE across all dimensions
    """
    errors = {}
    mse_per_dim = []
    max_diff_per_dim = []

    dim_names = ['H', 'W', 'D'][:dimension]

    for dim_idx in range(dimension):
        spatial_dim = dim_idx + 2  # Account for B, C

        # Get first and last slices
        first_slice = [slice(None)] * x.dim()
        first_slice[spatial_dim] = 0

        last_slice = [slice(None)] * x.dim()
        last_slice[spatial_dim] = -1

        first = x[tuple(first_slice)]
        last = x[tuple(last_slice)]

        diff = first - last
        mse = (diff ** 2).mean().item()
        max_diff = diff.abs().max().item()

        mse_per_dim.append(mse)
        max_diff_per_dim.append(max_diff)
        errors[f'mse_{dim_names[dim_idx]}'] = mse
        errors[f'max_diff_{dim_names[dim_idx]}'] = max_diff

    errors['total_mse'] = sum(mse_per_dim)
    errors['mse_per_dim'] = mse_per_dim
    errors['max_diff_per_dim'] = max_diff_per_dim

    return errors

"""
Sequential Inpainting for Z-direction Extension

This module implements sequential inpainting for extending volumes along the z-direction.
The algorithm works by:
1. Generating the first block using standard denoising/sampling
2. For subsequent blocks, using inpainting with overlap from the previous block
3. Stitching blocks together with cosine blending in the overlap region

Algorithm (from the paper/presentation):
    Input: x_T ∈ R^{C × H × W}, overlap δ, block width w, conditions {c_i}

    for i = 1, ..., ⌈W/w⌉ do
        Extract: x^{(i)}_T ← x_T|_{[:, :, (i-1)w-δ : iw+δ]}

        if i = 1 then
            x^{(i)}_0 ← Denoise(x^{(i)}_T, c_i)
        else
            x^{patch} ← x^{(i-1)}_0|_{[:, :, -2δ:]}, I^{patch} ← [:, :, 0:2δ]
            x^{(i)}_0 ← Inpaint(x^{(i)}_T, x^{patch}, I^{patch}, c_i)
        end if

        Stitch with cosine blending
    end for

    return x_0
"""

from typing import Literal
from jaxtyping import Float
from torch import Tensor
import torch
import numpy as np
import math


def _create_cosine_blend_weights(overlap_size: int, device: torch.device) -> torch.Tensor:
    """
    Creates cosine blending weights for smooth transitions between blocks.

    The weights go from 0 to 1 over the overlap region using a cosine curve,
    which provides a smooth transition that sums to 1 when combined with (1 - weights).

    Args:
        overlap_size: Number of voxels in the overlap region (2δ in the algorithm)
        device: Device to place the weights on

    Returns:
        Tensor of shape [overlap_size] with values from 0 to 1
    """
    # Cosine weights: 0 at start, 1 at end
    # Using (1 - cos(π * t)) / 2 which goes from 0 to 1 smoothly
    t = torch.linspace(0, 1, overlap_size, device=device)
    weights = (1 - torch.cos(math.pi * t)) / 2
    return weights


def _build_inpaint_mask_sequential(
    block_shape: list[int],
    overlap_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Creates mask for inpainting in sequential mode.
    Mask is 1 in the overlap region (first 2δ voxels in z), 0 elsewhere.

    Args:
        block_shape: [channels, dx, dy, dz] - shape of the block to generate
        overlap_size: Total overlap size (2δ) - the region where previous block data exists
        device: Device to place the mask on

    Returns:
        Mask tensor of shape [channels, dx, dy, dz]
        where 1 indicates known data (from previous block), 0 indicates to generate
    """
    channels, dx, dy, dz = block_shape
    mask = torch.zeros(block_shape, device=device)
    # The first overlap_size voxels in z contain known data from previous block
    mask[:, :, :, :overlap_size] = 1.0
    return mask


def sample_sequential_z(
    flow_module,
    num_blocks: int,
    base_shape: list[int],
    overlap_size: int,
    y: None | dict[str, torch.Tensor] | np.ndarray | list = None,
    guidance: float = 1.0,
    nsteps: int = 30,
    blend_mode: Literal['cosine', 'latest'] = 'cosine',
    inpaint_method: Literal['inpaint', 'inpaint_dps', 'inpaint_lanpaint'] = 'inpaint_dps',
    inpaint_kwargs: dict | None = None,
) -> Float[Tensor, "batch channels dx dy final_dz"]:
    """
    Generate extended volumes by sequential inpainting along the z-direction.

    This function generates a volume that extends in the z-direction by:
    1. Generating the first block using standard sampling (extended by overlap_size//2 on top)
    2. For middle blocks, using inpainting with overlap on both sides
    3. For the last block, using inpainting with overlap only on bottom
    4. Stitching blocks with cosine blending in overlap regions

    The blocks are extended to create overlapping regions:
    - First block: base_shape[3] + overlap_size//2 (extended on top/end)
    - Middle blocks: base_shape[3] + overlap_size (extended on both sides)
    - Last block: base_shape[3] + overlap_size//2 (extended on bottom/start)

    Args:
        flow_module: SIModule/FlowModule instance for generation
        num_blocks: Number of blocks to generate along z-direction
        base_shape: [channels, dx, dy, dz] - base block shape (the "core" size per block)
        overlap_size: Total overlap between adjacent blocks (must be even).
            Each block extends by overlap_size//2 into adjacent blocks.
        y: Conditioning for each block. Can be:
           - None: No conditioning
           - dict: Same conditioning for all blocks
           - list/np.ndarray of length num_blocks: Per-block conditioning
        guidance: Guidance scale for conditional generation
        nsteps: Number of integration steps
        blend_mode: How to handle overlaps:
            - 'cosine': Smooth blending using cosine weights
            - 'latest': Simply overwrite with latest block
        inpaint_method: Which inpainting algorithm to use:
            - 'inpaint': Original replacement-based
            - 'inpaint_dps': Diffusion Posterior Sampling (default)
            - 'inpaint_lanpaint': Langevin dynamics (experimental)
        inpaint_kwargs: Additional kwargs passed to the inpainting function

    Returns:
        Generated volume tensor of shape [1, channels, dx, dy, final_dz]
        where final_dz = base_shape[3] * num_blocks
    """
    # Validate inputs
    if num_blocks < 1:
        raise ValueError("num_blocks must be at least 1")
    if overlap_size < 0:
        raise ValueError("overlap_size must be non-negative")
    if overlap_size % 2 != 0:
        raise ValueError("overlap_size must be even")
    if overlap_size >= base_shape[3]:
        raise ValueError("overlap_size must be less than base block z-dimension")

    overlap_half = overlap_size // 2

    # Handle conditioning
    if isinstance(y, dict) or y is None:
        conditions = [y for _ in range(num_blocks)]
    elif isinstance(y, np.ndarray):
        conditions = list(y)
    else:
        conditions = y

    if len(conditions) != num_blocks:
        raise ValueError(f"Expected {num_blocks} conditions, got {len(conditions)}")

    # Compute final volume shape: exactly base_shape[3] * num_blocks
    final_dz = base_shape[3] * num_blocks
    final_shape = [base_shape[0], base_shape[1], base_shape[2], final_dz]

    device = flow_module.device

    # Initialize final volume
    volume = torch.zeros(1, *final_shape, device=device)

    # Previous block's generated data (for extracting overlap region)
    prev_block = None

    for i in range(num_blocks):
        is_first = (i == 0)
        is_last = (i == num_blocks - 1)

        # Compute the extended block shape for this block
        # First block: extends overlap_half on the end (top)
        # Last block: extends overlap_half on the start (bottom)
        # Middle blocks: extends overlap_half on both sides
        if num_blocks == 1:
            # Single block, no extension needed
            extended_dz = base_shape[3]
            extend_start = 0
            extend_end = 0
        elif is_first:
            extended_dz = base_shape[3] + overlap_half
            extend_start = 0
            extend_end = overlap_half
        elif is_last:
            extended_dz = base_shape[3] + overlap_half
            extend_start = overlap_half
            extend_end = 0
        else:
            extended_dz = base_shape[3] + overlap_size
            extend_start = overlap_half
            extend_end = overlap_half

        extended_shape = [base_shape[0], base_shape[1], base_shape[2], extended_dz]

        # Compute z-range for this block's core region in the final volume
        z_core_start = i * base_shape[3]
        z_core_end = z_core_start + base_shape[3]

        if is_first:
            # First block: use standard sampling (no inpainting)
            generated_block = flow_module.sample(
                nsamples=1,
                shape=extended_shape,
                y=conditions[i],
                guidance=guidance,
                nsteps=nsteps,
                is_latent_shape=True,
                return_latents=True,
            )
            # generated_block shape: [1, channels, dx, dy, extended_dz]
            generated_block = generated_block[0]  # Remove batch dim

            # Place the core region (without the extension on top)
            # generated_block[:, :, :, :base_shape[3]] is the core
            volume[0, :, :, :, z_core_start:z_core_end] = generated_block[:, :, :, :base_shape[3]]

        else:
            # Subsequent blocks: use inpainting with overlap from previous block

            # Extract the overlap region from previous block
            # The previous block's last overlap_half voxels overlap with this block's first overlap_half
            # But we need overlap_size total in x_orig for the mask region
            overlap_from_prev = prev_block[:, :, :, -overlap_size:]  # [channels, dx, dy, overlap_size]

            # Create x_orig with known data at the start
            x_orig = torch.zeros(extended_shape, device=device)
            x_orig[:, :, :, :overlap_size] = overlap_from_prev

            # Create inpainting mask: 1 where data is known (first overlap_size voxels)
            mask = _build_inpaint_mask_sequential(extended_shape, overlap_size, device)
            
            # Generate using inpainting
            inpaint_fn = getattr(flow_module, inpaint_method)
            generated_block = inpaint_fn(
                x_orig=x_orig,
                mask=mask,
                nsamples=1,
                y=conditions[i],
                guidance=guidance,
                nsteps=nsteps,
                **(inpaint_kwargs or {}),
            )
            generated_block = generated_block[0]  # Remove batch dim

            # Now we need to blend the overlap region and place the core
            # The overlap region in the final volume spans:
            #   [z_core_start - overlap_half, z_core_start + overlap_half]
            # which is [prev_block_core_end - overlap_half, z_core_start + overlap_half]

            overlap_vol_start = z_core_start - overlap_half
            overlap_vol_end = z_core_start + overlap_half

            print(z_core_start, z_core_end)
            print(overlap_vol_start, overlap_vol_end)
            if blend_mode == 'cosine':
                # Create blending weights for the overlap region
                blend_weights = _create_cosine_blend_weights(overlap_size, device)
                # Reshape for broadcasting: [1, 1, 1, overlap_size]
                blend_weights = blend_weights.view(1, 1, 1, overlap_size)

                # Get current values in overlap region (from previous block)
                current_overlap = volume[0, :, :, :, overlap_vol_start:overlap_vol_end]
                # New overlap from this block (first overlap_size voxels)
                new_overlap = generated_block[:, :, :, :overlap_size]

                # Weighted blend (new block gets increasing weight as z increases)
                blended = current_overlap * (1 - blend_weights) + new_overlap * blend_weights
                volume[0, :, :, :, overlap_vol_start:overlap_vol_end] = blended

                # Place the rest of the core region (after overlap)
                # In generated_block, after the overlap region, we have the rest of the core
                # Core region in generated_block: [overlap_size : overlap_size + (base_shape[3] - overlap_half)]
                # But for last block: [overlap_size : overlap_size + base_shape[3] - overlap_half] = [overlap_size : extended_dz]
                core_in_gen_start = overlap_size
                if is_last:
                    core_in_gen_end = extended_dz
                else:
                    core_in_gen_end = extended_dz  # This includes the extension for next overlap

                # Place non-overlap part of core
                volume[0, :, :, :, overlap_vol_end:z_core_end] = generated_block[:, :, :, overlap_size:overlap_size + (z_core_end - overlap_vol_end)]

            else:  # blend_mode == 'latest'
                # Simply overwrite
                volume[0, :, :, :, overlap_vol_start:z_core_end] = generated_block[:, :, :, :overlap_half + base_shape[3]]

        # Store for next iteration (we need the full extended block for overlap extraction)
        prev_block = generated_block

    return volume



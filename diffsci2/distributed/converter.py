"""
Recursive model converter: standard PUNetG -> spatially-parallel PUNetG.

Follows the same pattern as punetg_converters.convert_conv_to_circular.
"""

import copy

import torch
import torch.nn as nn

from .layers import SpatialParallelConv3d, SpatialParallelGroupNorm


def convert_to_spatial_parallel(module, ctx, inplace=False):
    """Convert all spatial layers in a module to spatial-parallel equivalents.

    Recursively replaces:
      - Conv3d / CircularConv3d -> SpatialParallelConv3d
      - GroupLNorm / GroupRMSNorm -> SpatialParallelGroupNorm

    Raises NotImplementedError if any attention layers are found.

    Parameters
    ----------
    module : nn.Module
        The model to convert (e.g. PUNetG or PUNetGCond).
    ctx : SpatialContext
        Distributed context.
    inplace : bool
        If False (default), deepcopy the model first.

    Returns
    -------
    nn.Module
        The converted model.
    """
    if not inplace:
        module = copy.deepcopy(module)

    stats = {'conv3d': 0, 'norm': 0, 'attention_error': False}
    _replace_recursive(module, ctx, stats)

    print(f"[Rank {ctx.rank}] Converted {stats['conv3d']} Conv3d, "
          f"{stats['norm']} norm layers to spatial-parallel.")
    return module


def _replace_recursive(module, ctx, stats):
    """Walk module tree and replace layers in-place."""
    # Import here to avoid circular imports at module level
    from diffsci2.nets.commonlayers import (
        CircularConv3d, GroupLNorm, GroupRMSNorm,
        ThreeDimensionalAttention,
    )

    for child_name, child in list(module.named_children()):
        # --- Attention: not supported ---
        if isinstance(child, ThreeDimensionalAttention):
            raise NotImplementedError(
                f"Spatial parallelism does not support ThreeDimensionalAttention "
                f"(found at '{child_name}'). Set number_resnet_attn_block=0."
            )

        # --- Conv3d (standard) ---
        elif isinstance(child, nn.Conv3d) and not isinstance(child, CircularConv3d):
            new_layer = _convert_conv3d(child, ctx)
            setattr(module, child_name, new_layer)
            stats['conv3d'] += 1

        # --- CircularConv3d ---
        elif isinstance(child, CircularConv3d):
            new_layer = _convert_circular_conv3d(child, ctx)
            setattr(module, child_name, new_layer)
            stats['conv3d'] += 1

        # --- torch.nn.GroupNorm (used for "GroupLN" in ResnetBlockC) ---
        elif isinstance(child, nn.GroupNorm) and not isinstance(child, SpatialParallelGroupNorm):
            new_layer = SpatialParallelGroupNorm(child, 'GroupNorm', ctx)
            setattr(module, child_name, new_layer)
            stats['norm'] += 1

        # --- GroupLNorm (custom) ---
        elif isinstance(child, GroupLNorm):
            new_layer = SpatialParallelGroupNorm(child, 'GroupLN', ctx)
            setattr(module, child_name, new_layer)
            stats['norm'] += 1

        # --- GroupRMSNorm ---
        elif isinstance(child, GroupRMSNorm):
            new_layer = SpatialParallelGroupNorm(child, 'GroupRMS', ctx)
            setattr(module, child_name, new_layer)
            stats['norm'] += 1

        # --- Everything else: recurse ---
        else:
            _replace_recursive(child, ctx, stats)


def _convert_conv3d(conv, ctx):
    """Convert a standard nn.Conv3d (with padding='same' or int) to SpatialParallelConv3d."""
    kernel_size = conv.kernel_size
    assert all(k == kernel_size[0] for k in kernel_size), \
        f"Non-cubic kernels not supported: {kernel_size}"
    k = kernel_size[0]
    assert k % 2 == 1, f"Even kernel sizes not supported: {k}"

    # Create inner conv with padding=0, copying weights
    inner_conv = nn.Conv3d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=k,
        stride=conv.stride,
        padding=0,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    inner_conv.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        inner_conv.bias.data.copy_(conv.bias.data)

    # Standard Conv3d uses zero padding on all spatial dims
    spatial_pad_modes = {2: 'zeros', 3: 'zeros', 4: 'zeros'}

    return SpatialParallelConv3d(inner_conv, k, spatial_pad_modes, ctx)


def _convert_circular_conv3d(circular_conv, ctx):
    """Convert a CircularConv3d to SpatialParallelConv3d."""
    k = circular_conv.kernel_size
    circular_dims = circular_conv.circular_dims  # set of ints

    # The inner Conv3d (padding=0) already has the weights
    inner_conv = circular_conv.conv

    # Determine padding modes for each spatial dim
    # In CircularConv3d: 0=D, 1=H, 2=W -> tensor dims 2, 3, 4
    spatial_pad_modes = {
        2: 'circular' if 0 in circular_dims else 'zeros',  # D
        3: 'circular' if 1 in circular_dims else 'zeros',  # H
        4: 'circular' if 2 in circular_dims else 'zeros',  # W
    }

    return SpatialParallelConv3d(inner_conv, k, spatial_pad_modes, ctx)


def count_spatial_parallel_layers(module):
    """Count layer types in a module (for verification)."""
    from diffsci2.nets.commonlayers import (
        CircularConv3d, GroupLNorm, GroupRMSNorm,
    )

    counts = {
        'Conv3d': 0,
        'CircularConv3d': 0,
        'SpatialParallelConv3d': 0,
        'GroupLNorm': 0,
        'GroupRMSNorm': 0,
        'SpatialParallelGroupNorm': 0,
    }
    for m in module.modules():
        if isinstance(m, SpatialParallelConv3d):
            counts['SpatialParallelConv3d'] += 1
        elif isinstance(m, CircularConv3d):
            counts['CircularConv3d'] += 1
        elif isinstance(m, nn.Conv3d):
            counts['Conv3d'] += 1
        elif isinstance(m, SpatialParallelGroupNorm):
            counts['SpatialParallelGroupNorm'] += 1
        elif isinstance(m, GroupLNorm):
            counts['GroupLNorm'] += 1
        elif isinstance(m, GroupRMSNorm):
            counts['GroupRMSNorm'] += 1
    return counts

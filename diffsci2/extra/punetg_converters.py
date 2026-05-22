"""
Post-training periodization utilities for PUNetG models.

This module provides functions to convert trained neural networks by replacing
their convolutional layers with circular (periodic) convolutional layers.
"""

import torch
import torch.nn as nn
from diffsci2.nets.commonlayers import (
    CircularConv2d,
    CircularConv3d,
    ReflectConv2d,
    ReflectConv3d,
)


def convert_conv_to_circular(
    module: nn.Module,
    circular_dims: list[int] | None = None,
    inplace: bool = False
) -> nn.Module:
    """
    Convert all Conv2d/Conv3d layers in a module to CircularConv2d/CircularConv3d.

    This enables post-training periodization: take a model trained with standard
    convolutions and make it periodic by replacing the padding strategy.

    Args:
        module: The neural network module to convert
        circular_dims: Which spatial dimensions should use circular padding.
                      For 3D [B, C, D, H, W]: [0]=D, [1]=H, [2]=W
                      For 2D [B, C, H, W]: [0]=H, [1]=W
                      None = all dimensions circular (default)
                      Example: [0, 1] for 3D = D and H periodic, W zero-padded
        inplace: If True, modify the module in place. If False, return a copy.

    Returns:
        The converted module with circular convolutions
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)

    _replace_conv_recursive(module, circular_dims)
    return module


def _replace_conv_recursive(
    module: nn.Module,
    circular_dims: list[int] | None,
    parent: nn.Module | None = None,
    name: str = ''
):
    """
    Recursively replace Conv2d/Conv3d with CircularConv2d/CircularConv3d.
    """
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, CircularConv2d):
            # Convert Conv2d to CircularConv2d
            new_conv = _conv2d_to_circular(child, circular_dims)
            setattr(module, child_name, new_conv)
        elif isinstance(child, nn.Conv3d) and not isinstance(child, CircularConv3d):
            # Convert Conv3d to CircularConv3d
            new_conv = _conv3d_to_circular(child, circular_dims)
            setattr(module, child_name, new_conv)
        else:
            # Recurse into child modules
            _replace_conv_recursive(child, circular_dims, module, child_name)


def _conv2d_to_circular(conv: nn.Conv2d, circular_dims: list[int] | None) -> CircularConv2d:
    """
    Convert a Conv2d layer to CircularConv2d, preserving weights.
    """
    # Check kernel size is odd (required for CircularConv)
    kernel_size = conv.kernel_size
    if isinstance(kernel_size, tuple):
        assert kernel_size[0] == kernel_size[1], "Non-square kernels not supported"
        kernel_size = kernel_size[0]

    if kernel_size % 2 == 0:
        raise ValueError(
            f"CircularConv requires odd kernel size, got {kernel_size}. "
            "This layer cannot be converted."
        )

    # Create CircularConv2d with same parameters
    circular_conv = CircularConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=kernel_size,
        circular_dims=circular_dims,
        stride=conv.stride,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None
    )

    # Copy weights
    circular_conv.conv.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        circular_conv.conv.bias.data.copy_(conv.bias.data)

    return circular_conv


def _conv3d_to_circular(conv: nn.Conv3d, circular_dims: list[int] | None) -> CircularConv3d:
    """
    Convert a Conv3d layer to CircularConv3d, preserving weights.
    """
    # Check kernel size is odd (required for CircularConv)
    kernel_size = conv.kernel_size
    if isinstance(kernel_size, tuple):
        assert all(k == kernel_size[0] for k in kernel_size), "Non-cubic kernels not supported"
        kernel_size = kernel_size[0]

    if kernel_size % 2 == 0:
        raise ValueError(
            f"CircularConv requires odd kernel size, got {kernel_size}. "
            "This layer cannot be converted."
        )

    # Create CircularConv3d with same parameters
    circular_conv = CircularConv3d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=kernel_size,
        circular_dims=circular_dims,
        stride=conv.stride,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None
    )

    # Copy weights
    circular_conv.conv.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        circular_conv.conv.bias.data.copy_(conv.bias.data)

    return circular_conv


def convert_conv_to_reflect(
    module: nn.Module,
    reflect_dims: list[int] | None = None,
    inplace: bool = False
) -> nn.Module:
    """
    Convert all Conv2d/Conv3d layers in a module to ReflectConv2d/ReflectConv3d.

    Mirrors convert_conv_to_circular but uses reflection padding instead of
    circular padding. Use case: train with reflection padding as a soft-border
    regularizer.

    Args:
        module: The neural network module to convert
        reflect_dims: Which spatial dimensions should use reflect padding.
                      For 3D [B, C, D, H, W]: [0]=D, [1]=H, [2]=W
                      For 2D [B, C, H, W]: [0]=H, [1]=W
                      None = all dimensions reflect (default)
                      Example: [0, 1] for 3D = D and H reflect, W zero-padded
        inplace: If True, modify the module in place. If False, return a copy.

    Returns:
        The converted module with reflection convolutions
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)

    _replace_conv_recursive_reflect(module, reflect_dims)
    return module


def _replace_conv_recursive_reflect(
    module: nn.Module,
    reflect_dims: list[int] | None,
    parent: nn.Module | None = None,
    name: str = ''
):
    """
    Recursively replace Conv2d/Conv3d with ReflectConv2d/ReflectConv3d.
    """
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, ReflectConv2d):
            new_conv = _conv2d_to_reflect(child, reflect_dims)
            setattr(module, child_name, new_conv)
        elif isinstance(child, nn.Conv3d) and not isinstance(child, ReflectConv3d):
            new_conv = _conv3d_to_reflect(child, reflect_dims)
            setattr(module, child_name, new_conv)
        else:
            _replace_conv_recursive_reflect(child, reflect_dims, module, child_name)


def _conv2d_to_reflect(conv: nn.Conv2d, reflect_dims: list[int] | None) -> ReflectConv2d:
    """
    Convert a Conv2d layer to ReflectConv2d, preserving weights.
    """
    kernel_size = conv.kernel_size
    if isinstance(kernel_size, tuple):
        assert kernel_size[0] == kernel_size[1], "Non-square kernels not supported"
        kernel_size = kernel_size[0]

    if kernel_size % 2 == 0:
        raise ValueError(
            f"ReflectConv requires odd kernel size, got {kernel_size}. "
            "This layer cannot be converted."
        )

    reflect_conv = ReflectConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=kernel_size,
        reflect_dims=reflect_dims,
        stride=conv.stride,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None
    )

    reflect_conv.conv.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        reflect_conv.conv.bias.data.copy_(conv.bias.data)

    return reflect_conv


def _conv3d_to_reflect(conv: nn.Conv3d, reflect_dims: list[int] | None) -> ReflectConv3d:
    """
    Convert a Conv3d layer to ReflectConv3d, preserving weights.
    """
    kernel_size = conv.kernel_size
    if isinstance(kernel_size, tuple):
        assert all(k == kernel_size[0] for k in kernel_size), "Non-cubic kernels not supported"
        kernel_size = kernel_size[0]

    if kernel_size % 2 == 0:
        raise ValueError(
            f"ReflectConv requires odd kernel size, got {kernel_size}. "
            "This layer cannot be converted."
        )

    reflect_conv = ReflectConv3d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=kernel_size,
        reflect_dims=reflect_dims,
        stride=conv.stride,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None
    )

    reflect_conv.conv.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        reflect_conv.conv.bias.data.copy_(conv.bias.data)

    return reflect_conv


def count_conv_layers(module: nn.Module) -> dict:
    """
    Count the number of convolutional layers in a module.

    Returns:
        Dictionary with counts for plain and wrapped Conv2d/Conv3d variants.
    """
    counts = {
        'Conv2d': 0,
        'Conv3d': 0,
        'CircularConv2d': 0,
        'CircularConv3d': 0,
        'ReflectConv2d': 0,
        'ReflectConv3d': 0,
    }

    for m in module.modules():
        if isinstance(m, CircularConv2d):
            counts['CircularConv2d'] += 1
        elif isinstance(m, CircularConv3d):
            counts['CircularConv3d'] += 1
        elif isinstance(m, ReflectConv2d):
            counts['ReflectConv2d'] += 1
        elif isinstance(m, ReflectConv3d):
            counts['ReflectConv3d'] += 1
        elif isinstance(m, nn.Conv2d):
            counts['Conv2d'] += 1
        elif isinstance(m, nn.Conv3d):
            counts['Conv3d'] += 1

    return counts

# cached_norms.py
# -----------------------------------------------------------------------------
# Normalization layers with caching support for chunked decoding.
#
# The Problem:
#   During chunk decode, each tile computes its own normalization statistics.
#   This creates inconsistencies at tile boundaries ("patches").
#
# The Solution:
#   1. Compute normalization statistics once on the FULL data (or global estimate)
#   2. Cache those statistics in the norm layers
#   3. Use cached statistics during chunked decode (instead of per-tile stats)
#
# Usage Modes:
#   - NORMAL: Compute stats from input (standard behavior)
#   - CACHE_STATS: Compute stats and cache them (for stats gathering pass)
#   - USE_CACHED: Use previously cached stats (for chunked decode)
#
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Literal, Dict, Any
from enum import Enum

import torch
import torch.nn as nn


class NormMode(Enum):
    """Operating mode for cached normalization layers."""
    NORMAL = "normal"           # Standard: compute stats from input
    CACHE_STATS = "cache"       # Compute stats and cache them
    USE_CACHED = "use_cached"   # Use previously cached stats


# ============================================================================
# CACHED GROUP RMS NORM
# ============================================================================

class CachedGroupRMSNorm(nn.Module):
    """
    GroupRMSNorm with caching support.

    In NORMAL mode: computes RMS from input (standard behavior)
    In CACHE_STATS mode: computes RMS and caches it
    In USE_CACHED mode: uses cached RMS for normalization
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

        # Caching state
        self._mode = NormMode.NORMAL
        self._cached_rms: Optional[torch.Tensor] = None

    @property
    def mode(self) -> NormMode:
        return self._mode

    @mode.setter
    def mode(self, value: NormMode):
        self._mode = value

    def set_mode(self, mode: Literal["normal", "cache", "use_cached"]):
        """Set the operating mode."""
        self._mode = NormMode(mode)

    def clear_cache(self):
        """Clear cached statistics."""
        self._cached_rms = None

    def has_cached_stats(self) -> bool:
        """Check if statistics are cached."""
        return self._cached_rms is not None

    def get_cached_stats(self) -> Optional[torch.Tensor]:
        """Get cached RMS (for inspection/debugging)."""
        return self._cached_rms

    def set_cached_stats(self, rms: torch.Tensor):
        """Manually set cached RMS."""
        self._cached_rms = rms

    def _compute_rms(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS over groups."""
        B, C = x.shape[:2]
        G = self.num_groups

        # Reshape to [B, G, C//G, *spatial]
        x_grouped = x.view(B, G, C // G, *x.shape[2:])

        # Normalize dims: all except B and G
        normalize_dims = tuple(range(2, x_grouped.dim()))

        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x_grouped.pow(2).mean(dim=normalize_dims, keepdim=True) + self.eps)

        return rms  # Shape: [B, G, 1, 1, ...] or [B, G, 1, 1, 1, ...]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mode-dependent behavior.

        Args:
            x: Input tensor [B, C, *spatial]

        Returns:
            Normalized tensor [B, C, *spatial]
        """
        B, C = x.shape[:2]
        G = self.num_groups

        # Reshape to groups
        x_grouped = x.view(B, G, C // G, *x.shape[2:])

        if self._mode == NormMode.NORMAL:
            # Standard: compute RMS from input
            rms = self._compute_rms(x)

        elif self._mode == NormMode.CACHE_STATS:
            # Compute and cache
            rms = self._compute_rms(x)
            self._cached_rms = rms.detach().clone()

        elif self._mode == NormMode.USE_CACHED:
            # Use cached (must exist)
            if self._cached_rms is None:
                raise RuntimeError(
                    "CachedGroupRMSNorm in USE_CACHED mode but no stats cached. "
                    "Run a stats gathering pass first."
                )
            rms = self._cached_rms
            # Ensure RMS is on same device/dtype as input
            if rms.device != x.device or rms.dtype != x.dtype:
                rms = rms.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        # Normalize
        x_normed = x_grouped / rms

        # Reshape back to [B, C, *spatial]
        x_normed = x_normed.view(B, C, *x.shape[2:])

        # Apply affine transform
        if self.affine:
            w = self.weight.view(1, C, *([1] * (x.dim() - 2)))
            b = self.bias.view(1, C, *([1] * (x.dim() - 2)))
            x_normed = x_normed * w + b

        return x_normed


# ============================================================================
# CACHED GROUP LAYER NORM
# ============================================================================

class CachedGroupLNorm(nn.Module):
    """
    GroupLNorm (LayerNorm style) with caching support.

    Computes: (x - mean) / std, where stats are over (C//G, *spatial) per group.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

        # Caching state
        self._mode = NormMode.NORMAL
        self._cached_mean: Optional[torch.Tensor] = None
        self._cached_std: Optional[torch.Tensor] = None

    @property
    def mode(self) -> NormMode:
        return self._mode

    @mode.setter
    def mode(self, value: NormMode):
        self._mode = value

    def set_mode(self, mode: Literal["normal", "cache", "use_cached"]):
        self._mode = NormMode(mode)

    def clear_cache(self):
        self._cached_mean = None
        self._cached_std = None

    def has_cached_stats(self) -> bool:
        return self._cached_mean is not None and self._cached_std is not None

    def get_cached_stats(self) -> Dict[str, Optional[torch.Tensor]]:
        return {"mean": self._cached_mean, "std": self._cached_std}

    def set_cached_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self._cached_mean = mean
        self._cached_std = std

    def _compute_stats(self, x: torch.Tensor):
        """Compute mean and std over groups."""
        B, C = x.shape[:2]
        G = self.num_groups

        x_grouped = x.view(B, G, C // G, *x.shape[2:])
        normalize_dims = tuple(range(2, x_grouped.dim()))

        mean = x_grouped.mean(dim=normalize_dims, keepdim=True)
        std = torch.sqrt((x_grouped - mean).pow(2).mean(dim=normalize_dims, keepdim=True) + self.eps)

        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        G = self.num_groups

        x_grouped = x.view(B, G, C // G, *x.shape[2:])

        if self._mode == NormMode.NORMAL:
            mean, std = self._compute_stats(x)

        elif self._mode == NormMode.CACHE_STATS:
            mean, std = self._compute_stats(x)
            self._cached_mean = mean.detach().clone()
            self._cached_std = std.detach().clone()

        elif self._mode == NormMode.USE_CACHED:
            if self._cached_mean is None or self._cached_std is None:
                raise RuntimeError(
                    "CachedGroupLNorm in USE_CACHED mode but no stats cached."
                )
            mean = self._cached_mean.to(device=x.device, dtype=x.dtype)
            std = self._cached_std.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        # Normalize
        x_normed = (x_grouped - mean) / std
        x_normed = x_normed.view(B, C, *x.shape[2:])

        if self.affine:
            w = self.weight.view(1, C, *([1] * (x.dim() - 2)))
            b = self.bias.view(1, C, *([1] * (x.dim() - 2)))
            x_normed = x_normed * w + b

        return x_normed


# ============================================================================
# CACHED STANDARD GROUP NORM (wraps torch.nn.GroupNorm)
# ============================================================================

class CachedGroupNorm(nn.Module):
    """
    Wrapper around torch.nn.GroupNorm with caching support.

    This replicates GroupNorm behavior but allows caching of mean/var stats.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

        self._mode = NormMode.NORMAL
        self._cached_mean: Optional[torch.Tensor] = None
        self._cached_var: Optional[torch.Tensor] = None

    @property
    def mode(self) -> NormMode:
        return self._mode

    @mode.setter
    def mode(self, value: NormMode):
        self._mode = value

    def set_mode(self, mode: Literal["normal", "cache", "use_cached"]):
        self._mode = NormMode(mode)

    def clear_cache(self):
        self._cached_mean = None
        self._cached_var = None

    def has_cached_stats(self) -> bool:
        return self._cached_mean is not None and self._cached_var is not None

    def get_cached_stats(self) -> Dict[str, Optional[torch.Tensor]]:
        return {"mean": self._cached_mean, "var": self._cached_var}

    def set_cached_stats(self, mean: torch.Tensor, var: torch.Tensor):
        self._cached_mean = mean
        self._cached_var = var

    def _compute_stats(self, x: torch.Tensor):
        """Compute mean and variance over groups."""
        B, C = x.shape[:2]
        G = self.num_groups

        x_grouped = x.view(B, G, C // G, *x.shape[2:])
        normalize_dims = tuple(range(2, x_grouped.dim()))

        mean = x_grouped.mean(dim=normalize_dims, keepdim=True)
        var = (x_grouped - mean).pow(2).mean(dim=normalize_dims, keepdim=True)

        return mean, var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        G = self.num_groups

        x_grouped = x.view(B, G, C // G, *x.shape[2:])

        if self._mode == NormMode.NORMAL:
            mean, var = self._compute_stats(x)

        elif self._mode == NormMode.CACHE_STATS:
            mean, var = self._compute_stats(x)
            self._cached_mean = mean.detach().clone()
            self._cached_var = var.detach().clone()

        elif self._mode == NormMode.USE_CACHED:
            if self._cached_mean is None or self._cached_var is None:
                raise RuntimeError(
                    "CachedGroupNorm in USE_CACHED mode but no stats cached."
                )
            mean = self._cached_mean.to(device=x.device, dtype=x.dtype)
            var = self._cached_var.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        # Normalize: (x - mean) / sqrt(var + eps)
        x_normed = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x_normed = x_normed.view(B, C, *x.shape[2:])

        if self.affine:
            w = self.weight.view(1, C, *([1] * (x.dim() - 2)))
            b = self.bias.view(1, C, *([1] * (x.dim() - 2)))
            x_normed = x_normed * w + b

        return x_normed


# ============================================================================
# UTILITY FUNCTIONS FOR MODEL CONVERSION
# ============================================================================

def _is_cached_norm(module: nn.Module) -> bool:
    """Check if a module is a cached norm layer."""
    return isinstance(module, (CachedGroupNorm, CachedGroupRMSNorm, CachedGroupLNorm))


def _get_cached_norms(model: nn.Module) -> Dict[str, nn.Module]:
    """Get all cached norm layers in a model."""
    cached = {}
    for name, module in model.named_modules():
        if _is_cached_norm(module):
            cached[name] = module
    return cached


def set_all_norms_mode(
    model: nn.Module,
    mode: Literal["normal", "cache", "use_cached"]
):
    """
    Set mode for all cached norm layers in a model.

    Args:
        model: The model containing cached norm layers
        mode: The mode to set
    """
    for module in model.modules():
        if _is_cached_norm(module):
            module.set_mode(mode)


def clear_all_norm_caches(model: nn.Module):
    """Clear cached stats from all cached norm layers."""
    for module in model.modules():
        if _is_cached_norm(module):
            module.clear_cache()


def convert_to_cached_norms(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    Convert all GroupNorm-like layers in a model to cached versions.

    Handles:
        - torch.nn.GroupNorm -> CachedGroupNorm
        - GroupRMSNorm -> CachedGroupRMSNorm
        - GroupLNorm -> CachedGroupLNorm

    Args:
        model: The model to convert
        inplace: If True, modify model in place. If False, return a copy.

    Returns:
        Model with converted norm layers
    """
    from . import commonlayers  # Local import to avoid circular dependency

    if not inplace:
        import copy
        model = copy.deepcopy(model)

    def _convert_module(parent: nn.Module, name: str, module: nn.Module):
        """Convert a single module and set it as attribute of parent."""
        new_module = None

        if isinstance(module, nn.GroupNorm):
            new_module = CachedGroupNorm(
                num_groups=module.num_groups,
                num_channels=module.num_channels,
                eps=module.eps,
                affine=module.affine,
            )
            if module.affine:
                new_module.weight.data.copy_(module.weight.data)
                new_module.bias.data.copy_(module.bias.data)

        elif isinstance(module, commonlayers.GroupRMSNorm):
            new_module = CachedGroupRMSNorm(
                num_groups=module.num_groups,
                num_channels=module.num_channels,
                eps=module.eps,
                affine=module.affine,
            )
            if module.affine:
                new_module.weight.data.copy_(module.weight.data)
                new_module.bias.data.copy_(module.bias.data)

        elif isinstance(module, commonlayers.GroupLNorm):
            new_module = CachedGroupLNorm(
                num_groups=module.num_groups,
                num_channels=module.num_channels,
                eps=module.eps,
                affine=module.affine,
            )
            if module.affine:
                new_module.weight.data.copy_(module.weight.data)
                new_module.bias.data.copy_(module.bias.data)

        if new_module is not None:
            setattr(parent, name, new_module)
            return True
        return False

    # Iterate through named children and convert
    converted_count = 0

    def _recursive_convert(module: nn.Module):
        nonlocal converted_count
        for name, child in list(module.named_children()):
            if _convert_module(module, name, child):
                converted_count += 1
            else:
                # Recurse into children
                _recursive_convert(child)

    _recursive_convert(model)

    return model


def gather_norm_stats_full(
    model: nn.Module,
    x: torch.Tensor,
    forward_fn=None,
):
    """
    Run a forward pass to gather and cache normalization statistics.

    This runs the model once with the FULL input to compute global stats,
    which are then cached for use during chunked decode.

    Args:
        model: Model with cached norm layers
        x: Full input tensor (will be run through model)
        forward_fn: Optional custom forward function. If None, uses model(x)

    Note:
        After this call, all cached norm layers will be in USE_CACHED mode
        with stats populated.
    """
    # Set all norms to cache mode
    set_all_norms_mode(model, "cache")

    # Run forward pass (stats will be cached)
    with torch.no_grad():
        if forward_fn is not None:
            forward_fn(model, x)
        else:
            model(x)

    # Switch to use_cached mode
    set_all_norms_mode(model, "use_cached")


# ============================================================================
# CONTEXT MANAGER FOR CACHED NORM MODE
# ============================================================================

class cached_norm_mode:
    """
    Context manager for temporarily setting cached norm mode.

    Usage:
        with cached_norm_mode(model, "use_cached"):
            # All cached norms use cached stats in this block
            output = chunk_decode(model, input)
        # After block, modes are restored to original
    """

    def __init__(
        self,
        model: nn.Module,
        mode: Literal["normal", "cache", "use_cached"]
    ):
        self.model = model
        self.target_mode = mode
        self.original_modes: Dict[nn.Module, NormMode] = {}

    def __enter__(self):
        # Save original modes
        for module in self.model.modules():
            if _is_cached_norm(module):
                self.original_modes[module] = module.mode
                module.set_mode(self.target_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original modes
        for module, original_mode in self.original_modes.items():
            module.mode = original_mode
        return False

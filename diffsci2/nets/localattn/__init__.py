"""Conv-free local-attention building blocks for 2D and 3D latent
diffusion.

Public surface:

- :class:`LocalSelfAttention2D`, :class:`LocalAttentionBlock2D`,
  :class:`LocalSelfAttention3D`, :class:`LocalAttentionBlock3D`
- :class:`PatchMerge2D`, :class:`PatchUnmerge2D`,
  :class:`PatchMerge3D`, :class:`PatchUnmerge3D`
- :class:`LAUNet`, :class:`LAUNetConfig`,
  :class:`LAUNet3D`, :class:`LAUNet3DConfig`
- :class:`LADit`, :class:`LADitConfig`
- :func:`set_periodic` — walk a module tree and flip every local-attn
  layer between aperiodic and periodic modes. Works for both 2D and 3D.

The 2D layers expose two backends: ``'mask'`` (dense + Chebyshev sparsity
mask) and ``'natten'`` (NATTEN's fused ``na2d`` kernel). The 3D layers
require NATTEN. Periodicity is implemented as a circular pad-and-crop
halo around an aperiodic kernel, which means a checkpoint trained
aperiodic can be flipped to periodic inference without retraining.
"""

# flake8: noqa

from .local_attention_2d import (
    LocalSelfAttention2D,
    LocalAttentionBlock2D,
    AdaLN,
    MLP,
    RadialBias,
    chebyshev_local_mask,
    modulate,
    set_periodic,
)
from .local_attention_3d import (
    LocalSelfAttention3D,
    LocalAttentionBlock3D,
    AdaLN3D,
    MLP3D,
    RadialBias3D,
    modulate3d,
)
from .patch_merge import PatchMerge2D, PatchUnmerge2D
from .patch_merge_3d import PatchMerge3D, PatchUnmerge3D
from .la_unet import (
    LAUNet,
    LAUNetConfig,
    GaussianFourierProjection,
    TimeMLP,
)
from .la_unet_3d import (
    LAUNet3D,
    LAUNet3DConfig,
    GaussianFourierProjection3D,
    TimeMLP3D,
)
from .la_dit import LADit, LADitConfig

__all__ = [
    # 2D primitives
    'LocalSelfAttention2D',
    'LocalAttentionBlock2D',
    'AdaLN',
    'MLP',
    'RadialBias',
    'chebyshev_local_mask',
    'modulate',
    # 3D primitives
    'LocalSelfAttention3D',
    'LocalAttentionBlock3D',
    'AdaLN3D',
    'MLP3D',
    'RadialBias3D',
    'modulate3d',
    # Sampling
    'PatchMerge2D', 'PatchUnmerge2D',
    'PatchMerge3D', 'PatchUnmerge3D',
    # Networks
    'LAUNet', 'LAUNetConfig',
    'LAUNet3D', 'LAUNet3DConfig',
    'LADit', 'LADitConfig',
    'GaussianFourierProjection', 'TimeMLP',
    'GaussianFourierProjection3D', 'TimeMLP3D',
    # Mode flip
    'set_periodic',
]

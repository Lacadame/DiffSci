"""diffsci2.vaesft — supervised fine-tuning of the 3D pore VAE.

Surface:

- `load_autoencoder(...)` / `load_autoencoder_module(...)` — convenience
  loaders for the production VAEs (`vae_groupnorm_legacy`,
  `vae_pixnorm_s{4,8}_{raw,sft}`).
- `VAESFTModule` + `SFTConfig` — Lightning training module for the
  PoreRegressor-guided supervised fine-tuning recipe.
- `FrozenRegressor` + `load_frozen_regressor()` — frozen perceptual-loss
  reward model (PoreRegressor3D, ~7.8 M params, R²≈0.98 on the test
  cache).
- losses (`HuberCfg`, `bce_pixel_anchor`, `regressor_huber`,
  `regressor_z_err_mean`) and schedule (`ScheduleCfg`, `LossWeights`,
  `weights_at_step`).
- datasets (`CachedChunkSampler`, `EvalPackDataset`, `collate_cached`,
  `ChunkSample`).
- input transforms (`normalize_to_logits`, `near_binary`,
  `deterministic_binary`, `LOGIT_CLAMP`).

Note: this subpackage imports symbols from `poreregressor.*`
(`PoreRegressorModule`, `STONES`, `TARGET_NAMES`, etc.). `poreregressor`
is *not* installed as a Python package; we bootstrap it onto sys.path at
import time (see `diffsci2.vaesft._paths`). If you move
`notebooks/exploratory/dfnai/scripts/poreregressor/`, update that file.
"""
from .loaders import (
    load_autoencoder,
    load_autoencoder_module,
    list_variants,
    DEFAULT_VARIANT,
    VARIANT_REGISTRY,
    VariantSpec,
)
from .sft_module import VAESFTModule, SFTConfig
from .loss import (
    HuberCfg,
    bce_pixel_anchor,
    regressor_huber,
    regressor_z_err_mean,
)
from .schedule import LossWeights, ScheduleCfg, weights_at_step
from .regressor import (
    FrozenRegressor,
    load_frozen_regressor,
    normalize_to_logits,
    near_binary,
    deterministic_binary,
    LOGIT_CLAMP,
)
from .chunk_loader import (
    CachedChunkSampler,
    EvalPackDataset,
    ChunkSample,
    collate_cached,
)

__all__ = [
    # loaders
    "load_autoencoder",
    "load_autoencoder_module",
    "list_variants",
    "DEFAULT_VARIANT",
    "VARIANT_REGISTRY",
    "VariantSpec",
    # SFT module
    "VAESFTModule",
    "SFTConfig",
    # loss + schedule
    "HuberCfg",
    "bce_pixel_anchor",
    "regressor_huber",
    "regressor_z_err_mean",
    "LossWeights",
    "ScheduleCfg",
    "weights_at_step",
    # regressor wrapper
    "FrozenRegressor",
    "load_frozen_regressor",
    "normalize_to_logits",
    "near_binary",
    "deterministic_binary",
    "LOGIT_CLAMP",
    # datasets
    "CachedChunkSampler",
    "EvalPackDataset",
    "ChunkSample",
    "collate_cached",
]

"""Convenience loaders for the production 3D pore VAEs.

Five variants are supported:

| variant                     | architecture                              | downsample |
| --------------------------- | ----------------------------------------- | ---------- |
| `vae_groupnorm_legacy`      | `VAENet` (GroupNorm), z_dim=4              | 8x         |
| `vae_pixnorm_s4_raw`        | `VAENet(norm_type='pixel')`, z_dim=2       | 4x         |
| `vae_pixnorm_s4_sft`        | same as `_s4_raw`, SFT'd in run03          | 4x         |
| `vae_pixnorm_s8_raw`        | `VAENetMP` approach_b, z_dim=4             | 8x         |
| `vae_pixnorm_s8_sft`        | same as `_s8_raw`, SFT'd in run04          | 8x         |

All variants return a **bare** VAE-like module with ``.encoder``,
``.decoder``, ``.config.z_dim`` (so anything that consumed the old
`model_loaders.load_autoencoder` works unchanged when wrapped via
``load_autoencoder_module``).

Usage:

    >>> from diffsci2.vaesft import load_autoencoder
    >>> vae = load_autoencoder()                          # default = vae_groupnorm_legacy
    >>> vae = load_autoencoder("vae_pixnorm_s8_sft")
    >>> vae = load_autoencoder(path="/custom.ckpt")
    >>> vae = load_autoencoder("vae_pixnorm_s4_sft", device="cuda:0")

If both `variant` and `path` are supplied, `path` wins (the variant
name is treated as a hint for which architecture to build, but the
state on disk is what's loaded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import os

import torch
import torch.nn as nn

from diffsci2.extra.legacy_loaders import (
    build_groupnorm_legacy_vae,
    build_pixnorm_s4_vae,
    remap_pixnorm_raw_state,
)
from diffsci2.nets import VAENetMP, VAENetMPConfig


# ----------------------------------------------------------------------------
# Variant registry
# ----------------------------------------------------------------------------

_REPO_ROOT = "/home/ubuntu/repos/DiffSci2"


# NOTE: paths below point at where the ckpts live *today* (training-run
# directories under `notebooks/exploratory/...` and the existing
# `savedmodels/pore/production/` slot). Danilo will physically copy
# them into `savedmodels/vae/vae_<variant>.ckpt` as a separate step;
# at that point the constants below should be updated to:
#
#   path=os.path.join(_REPO_ROOT, "savedmodels/vae/vae_<variant>.ckpt")
#
# Instructions for the migration: claude/report/VAE_PORT_MIGRATION.md.


@dataclass(frozen=True)
class VariantSpec:
    """One row in `VARIANT_REGISTRY` — name, on-disk location, blurb."""
    name: str
    path: str
    description: str


VARIANT_REGISTRY: dict[str, VariantSpec] = {
    "vae_groupnorm_legacy": VariantSpec(
        name="vae_groupnorm_legacy",
        path=os.path.join(
            _REPO_ROOT, "savedmodels/pore/production/converted_vaenet.ckpt"
        ),
        description=(
            "Legacy GroupNorm VAENet (z_dim=4, ch_mult=[1,2,4,4], 8x "
            "downsample). The historical production decoder."
        ),
    ),
    "vae_pixnorm_s4_raw": VariantSpec(
        name="vae_pixnorm_s4_raw",
        path=os.path.join(
            _REPO_ROOT,
            "savedmodels/pore/production/converted_vaenet_s4_pixnorm.ckpt",
        ),
        description=(
            "PixelNorm VAENet (z_dim=2, ch_mult=[1,2,4], 4x downsample). "
            "Pretrained PixelNorm decoder, no SFT."
        ),
    ),
    "vae_pixnorm_s4_sft": VariantSpec(
        name="vae_pixnorm_s4_sft",
        path=os.path.join(
            _REPO_ROOT,
            "savedmodels/pore/production/converted_vaenet_s4_pixnorm_sft.ckpt"
        ),
        description=(
            "PixelNorm-s4 VAE after run03 supervised fine-tuning with "
            "the PoreRegressor reward. K_abs |bias| 0.092 % pooled at "
            "256 cubed (vs 0.65 % for the legacy GroupNorm decoder)."
        ),
    ),
    "vae_pixnorm_s8_raw": VariantSpec(
        name="vae_pixnorm_s8_raw",
        path=os.path.join(
            _REPO_ROOT,
            "savedmodels/pore/production/converted_vaenet_s8_pixnorm.ckpt"
        ),
        description=(
            "VAENetMP approach_b (z_dim=4, ch_mult=[1,2,4,4], 8x "
            "downsample, bare convs). Pretrained, no SFT."
        ),
    ),
    "vae_pixnorm_s8_sft": VariantSpec(
        name="vae_pixnorm_s8_sft",
        path=os.path.join(
            _REPO_ROOT,
            # "savedmodels/pore/production/converted_vaenet_s8_pixnorm_sft.ckpt"
            "notebooks/exploratory/dfnai/scripts/vaeporesft/checkpoints/run04_pixnorm_s8/best-step4400-z0.1302.ckpt"
        ),
        description=(
            "VAENetMP after run04 supervised fine-tuning. Same 8x "
            "compression as the legacy GroupNorm decoder but with "
            "finite RF (bit-exact tiled inference)."
        ),
    ),
}


DEFAULT_VARIANT: str = "vae_groupnorm_legacy"


def list_variants() -> list[str]:
    return list(VARIANT_REGISTRY.keys())


# ----------------------------------------------------------------------------
# State-dict sniffing + per-architecture builders
# ----------------------------------------------------------------------------

def _sniff_format(state: dict) -> str:
    """Return the variant-name string implied by the state_dict layout."""
    has_encdec = any(k.startswith("encdec.") for k in state)
    has_regressor = any(k.startswith("regressor.") for k in state)
    is_4level = (
        any(k.startswith("encdec.encoder.down.3.") for k in state)
        or any(k.startswith("encoder.down.3.") for k in state)
    )

    if has_regressor:
        return "vae_pixnorm_s8_sft" if is_4level else "vae_pixnorm_s4_sft"
    if has_encdec:
        return "vae_pixnorm_s8_raw" if is_4level else "vae_pixnorm_s4_raw"
    return "vae_groupnorm_legacy"


def _build_pixnorm_s8_vaenetmp() -> nn.Module:
    cfg = VAENetMPConfig.approach_b(
        dimension=3, in_channels=1, out_channels=1,
        z_channels=4, z_dim=4,
        ch=32, ch_mult=[1, 2, 4, 4],
        num_res_blocks=2, resolution=64,
        output_gain_init=1.0,
    )
    return VAENetMP(cfg)


def _build_groupnorm_legacy(state: dict) -> nn.Module:
    model = build_groupnorm_legacy_vae()
    model.load_state_dict(state, strict=True)
    return model


def _build_pixnorm_s4_raw(state: dict) -> nn.Module:
    model = build_pixnorm_s4_vae()
    state = remap_pixnorm_raw_state(state)
    model.load_state_dict(state, strict=True)
    return model


def _build_pixnorm_s4_sft(state: dict) -> nn.Module:
    """SFT ckpt with `encoder.*`, `decoder.*`, `regressor.*` keys
    (PatchedConv layout); drop regressor."""
    model = build_pixnorm_s4_vae()
    sub_state = {
        k: v for k, v in state.items()
        if k.startswith("encoder.") or k.startswith("decoder.")
    }
    model.load_state_dict(sub_state, strict=True)
    return model


def _build_pixnorm_s8_raw(state: dict) -> nn.Module:
    """Raw pixnorm-s8 ckpt with `encdec.` prefix; strip it and load
    into a VAENetMP."""
    model = _build_pixnorm_s8_vaenetmp()
    sub_state = {
        k[len("encdec."):]: v
        for k, v in state.items()
        if k.startswith("encdec.")
    }
    model.load_state_dict(sub_state, strict=True)
    return model


def _build_pixnorm_s8_sft(state: dict) -> nn.Module:
    """SFT ckpt with `encoder.*`, `decoder.*`, `regressor.*` keys
    (bare VAENetMP layout); drop regressor."""
    model = _build_pixnorm_s8_vaenetmp()
    sub_state = {
        k: v for k, v in state.items()
        if k.startswith("encoder.") or k.startswith("decoder.")
    }
    model.load_state_dict(sub_state, strict=True)
    return model


_BUILDERS: dict[str, Callable[[dict], nn.Module]] = {
    "vae_groupnorm_legacy": _build_groupnorm_legacy,
    "vae_pixnorm_s4_raw":   _build_pixnorm_s4_raw,
    "vae_pixnorm_s4_sft":   _build_pixnorm_s4_sft,
    "vae_pixnorm_s8_raw":   _build_pixnorm_s8_raw,
    "vae_pixnorm_s8_sft":   _build_pixnorm_s8_sft,
}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def load_autoencoder(
    variant: Optional[str] = None,
    path: Optional[str] = None,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> nn.Module:
    """Return a ready-to-use bare VAE.

    Resolution order:
      1. If `path` is given, sniff the on-disk state, dispatch to the
         right builder, return.
      2. If `variant` is given, look up `VARIANT_REGISTRY[variant]`
         and load from `variant_spec.resolve_path()`.
      3. Otherwise, use `DEFAULT_VARIANT`.

    The returned module is `.eval()` (unless `eval_mode=False`) and on
    `device` (CPU by default).
    """
    if path is None:
        v = variant or DEFAULT_VARIANT
        if v not in VARIANT_REGISTRY:
            raise KeyError(
                f"Unknown VAE variant {v!r}. Known: {sorted(VARIANT_REGISTRY)}"
            )
        spec = VARIANT_REGISTRY[v]
        path = spec.path

    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)

    detected = _sniff_format(state)
    builder = _BUILDERS[detected]
    model = builder(state)

    if eval_mode:
        model.eval()
    model.to(device)
    return model


def load_autoencoder_module(
    variant: Optional[str] = None,
    path: Optional[str] = None,
    device: str | torch.device = "cpu",
):
    """Return `diffsci2.models.VAEModule(encdec=load_autoencoder(...))`.

    Drop-in replacement for the old
    `notebooks/exploratory/dfn/aux/model_loaders.load_autoencoder()`
    which returned a `VAEModule` wrapping the bare VAE.

    Imported lazily so that ``diffsci2.vaesft`` doesn't pull in the
    `models` subpackage at import time.
    """
    import diffsci2.models  # lazy
    vae = load_autoencoder(variant=variant, path=path, device=device)
    cfg = diffsci2.models.VAEModuleConfig()
    module = diffsci2.models.VAEModule(config=cfg, encdec=vae)
    return module

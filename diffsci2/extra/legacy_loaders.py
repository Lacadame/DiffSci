"""Legacy VAE checkpoint loaders absorbed from the
`notebooks/exploratory/dfn/aux/model_loaders.py` script.

Use these directly only if you know the format on disk. For
auto-dispatched loading from a variant name, prefer
`diffsci2.vaesft.load_autoencoder(...)`.

Three architectures are covered:

- **Legacy GroupNorm `VAENet`** (`load_teacher_autoencoder`): z_dim=4,
  ch_mult=[1,2,4,4], GroupNorm, 256-resolution training, PatchedConv
  wrapping. Matches `savedmodels/.../converted_vaenet.ckpt`.
- **PixelNorm-s4 `VAENet`** (`build_pixnorm_s4_vae`): z_dim=2,
  ch_mult=[1,2,4], `norm_type='pixel'`, PatchedConv wrapping. The
  on-disk pretrained checkpoint
  (`savedmodels/.../converted_vaenet_pixnorm.ckpt`) has bare-conv keys
  and an `encdec.` prefix; `remap_pixnorm_raw_state` rewrites them
  before load.
- **PixelNorm-s8 `VAENetMP`** is handled directly in
  `diffsci2.vaesft.loaders` (it lives next to its loader because there
  is no PatchedConv layer-rename to deal with).

The original `load_teacher_autoencoder` lives in
`notebooks/exploratory/dfn/aux/model_loaders.py`. This module is the
canonical absorbed copy. The original file is left untouched per
Danilo's instruction so the two paths coexist temporarily.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import diffsci2.nets as _dn


# ----------------------------------------------------------------------------
# Legacy GroupNorm VAENet (z_dim=4, ch_mult=[1,2,4,4]).
# ----------------------------------------------------------------------------

_TEACHER_CONFIG = dict(
    dimension=3, in_channels=1, out_channels=1,
    z_channels=4, z_dim=4,
    ch=32, ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
    resolution=256,
    has_mid_attn=False,
    resamp_with_conv=True,
    attn_type='vanilla',
    tanh_out=False,
    input_bias=True,
    output_bias=True,
    with_time_emb=False,
    double_z=True,
    num_groups=32,
    patch_size=None,
    memory_efficient_variant=False,
    use_flash_attention=True,
    minimal_rf_mode=False,
)


def build_groupnorm_legacy_vae() -> nn.Module:
    """Construct an *uninitialized* legacy GroupNorm VAENet (z_dim=4)."""
    return _dn.VAENet(_dn.VAENetConfig(**_TEACHER_CONFIG))


def load_teacher_autoencoder(checkpoint_path: str) -> nn.Module:
    """Build + strict-load the legacy GroupNorm VAENet from disk.

    The on-disk format matches the VAENet PatchedConv layout (keys
    like `encoder.conv_in.conv.weight`), so no remapping is needed.
    """
    model = build_groupnorm_legacy_vae()
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# PixelNorm-s4 VAENet (z_dim=2, ch_mult=[1,2,4], norm_type='pixel').
# ----------------------------------------------------------------------------

_PIXNORM_S4_CONFIG = dict(
    dimension=3, in_channels=1, out_channels=1,
    z_channels=4, z_dim=2,
    ch=32, ch_mult=[1, 2, 4],
    num_res_blocks=2, resolution=64,
    attn_resolutions=[], has_mid_attn=False,
    dropout=0.0, num_groups=32,
    norm_type='pixel', output_gain_init=1.0,
)


def build_pixnorm_s4_vae() -> nn.Module:
    """Construct an *uninitialized* PixelNorm-s4 VAENet (z_dim=2)."""
    return _dn.VAENet(_dn.VAENetConfig(**_PIXNORM_S4_CONFIG))


# Layer names that get wrapped in `PatchedConv` inside `VAENet`. On
# disk (in `converted_vaenet_pixnorm.ckpt`) these appear as
# `<name>.weight`; the in-memory `VAENet` exposes them as
# `<name>.conv.weight`.
_PATCHED_NAMES = {
    "conv_in", "conv_out", "conv1", "conv2", "nin_shortcut",
    "quant_conv", "post_quant_conv",
}


def remap_pixnorm_raw_state(state: dict) -> dict:
    """Rewrite the bare on-disk state_dict of the pixnorm-s4 *raw*
    checkpoint to the PatchedConv layout that `VAENet` expects.

    Drops `loss_module.*` and any non-`encdec.` entries. This is the
    exact same transform `vaeporesft.vae_loader._remap_pixnorm_state`
    used to perform — moved here so the convenience loader can call
    it without a dependency on the notebook code.
    """
    out: dict = {}
    for k, v in state.items():
        if not k.startswith("encdec."):
            continue
        k = k[len("encdec."):]
        k = k.replace(".shortcut.", ".nin_shortcut.")
        k = k.replace(".mid_block_1.", ".mid.block_1.")
        k = k.replace(".mid_block_2.", ".mid.block_2.")
        if k.endswith(".weight") or k.endswith(".bias"):
            base, _, suffix = k.rpartition(".")
            last = base.rsplit(".", 1)[-1] if "." in base else base
            if last in _PATCHED_NAMES or base.endswith(".upsample.conv"):
                k = f"{base}.conv.{suffix}"
        out[k] = v
    return out


def load_pixnorm_s4_raw(checkpoint_path: str) -> nn.Module:
    """Build + strict-load the pixnorm-s4 raw checkpoint."""
    blob = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = build_pixnorm_s4_vae()
    state = remap_pixnorm_raw_state(blob['state_dict'])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Flow models (PUNetG) — copied from
# `notebooks/exploratory/dfn/aux/model_loaders.py`. The original lives on per
# Danilo's instruction so the two paths coexist for now.
# ----------------------------------------------------------------------------

_FLOW_MODEL_PATHS = {
    "Bentheimer":  "/home/ubuntu/repos/DiffSci2/savedmodels/pore/production/bentheimer_pcond.ckpt",
    "Doddington":  "/home/ubuntu/repos/DiffSci2/savedmodels/pore/production/doddington_pcond.ckpt",
    "Estaillades": "/home/ubuntu/repos/DiffSci2/savedmodels/pore/production/estaillades_pcond.ckpt",
    "Ketton":      "/home/ubuntu/repos/DiffSci2/savedmodels/pore/production/ketton_pcond.ckpt",
}


def _load_model_substate(checkpoint_path: str, identifier: str = "model") -> dict:
    """Strip the Lightning top-level prefix (e.g. `model.X` → `X`)."""
    loaded = torch.load(checkpoint_path)['state_dict']
    out: dict = {}
    for key, value in loaded.items():
        head = key.split(".")[0]
        if head == identifier:
            out[".".join(key.split(".")[1:])] = value
    return out


def load_flow_model(name: str, custom_checkpoint_path: bool = False) -> nn.Module:
    """Build a `PUNetG` flow model and load weights from the per-stone
    production checkpoint. Mirrors
    `notebooks/exploratory/dfn/aux/model_loaders.load_flow_model`.

    `name` is one of `'Bentheimer' / 'Doddington' / 'Estaillades' / 'Ketton'`
    (resolved against the `_FLOW_MODEL_PATHS` table), or — when
    `custom_checkpoint_path=True` — an arbitrary checkpoint path.
    """
    if custom_checkpoint_path:
        checkpoint_path = name
    else:
        checkpoint_path = _FLOW_MODEL_PATHS[name]
    weights = _load_model_substate(checkpoint_path)
    embedder = _dn.ScalarEmbedder(dembed=64, key='porosity')
    cfg = _dn.PUNetGConfig(
        input_channels=4,
        output_channels=4,
        dimension=3,
        model_channels=64,
        channel_expansion=[2, 4],
        number_resnet_downward_block=2,
        number_resnet_upward_block=2,
        number_resnet_attn_block=0,
        number_resnet_before_attn_block=3,
        number_resnet_after_attn_block=3,
        kernel_size=3,
        in_out_kernel_size=3,
        in_embedding=False,
        time_projection_scale=10.0,
        input_projection_scale=1.0,
        transition_scale_factor=2,
        transition_kernel_size=3,
        dropout=0.1,
        cond_dropout=0.1,
        first_resblock_norm="GroupLN",
        second_resblock_norm="GroupRMS",
        affine_norm=True,
        convolution_type="default",
        num_groups=1,
        attn_residual=False,
        attn_type="default",
        bias=True,
    )
    model = _dn.PUNetG(cfg, conditional_embedding=embedder)
    model.load_state_dict(weights)
    return model

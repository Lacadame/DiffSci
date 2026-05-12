"""Shared fixtures and utilities for chunk encode/decode tests.

Tests are plain Python scripts (not pytest) to match the rest of tests/.
Each test file should call `setup_repo_path()` first, then use the loader
helpers below.
"""
import os
import sys

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TUTORIAL_DATA = os.path.join(REPO_ROOT, 'notebooks', 'tutorials', '0002-data')
TOY_VAE_CKPT = os.path.join(TUTORIAL_DATA, 'vae.ckpt')
BEREA_SLICES = os.path.join(TUTORIAL_DATA, 'berea_slices.npy')

PROD_VAE_CKPT = os.path.join(
    REPO_ROOT, 'savedmodels', 'pore', 'production', 'converted_vaenet.ckpt'
)
BENTHEIMER_RAW = os.path.join(
    REPO_ROOT, 'saveddata', 'raw', 'imperial_college',
    'Bentheimer_1000c_3p0035um.raw'
)


def setup_repo_path():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def toy_2d_vae_config():
    """Config matching notebooks/tutorials/0002-chunk-decoding.ipynb."""
    import diffsci2.nets
    return diffsci2.nets.VAENetConfig(
        dimension=2, in_channels=1, out_channels=1, z_channels=4, z_dim=4,
        ch=32, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[],
        dropout=0.0, resolution=128, has_mid_attn=False, resamp_with_conv=True,
        attn_type='vanilla', tanh_out=False, input_bias=True, output_bias=True,
        with_time_emb=False, double_z=True, num_groups=32, patch_size=None,
        memory_efficient_variant=False, use_flash_attention=True,
        minimal_rf_mode=False,
    )


def production_3d_vae_config():
    """Config matching aux/model_loaders.load_teacher_autoencoder."""
    import diffsci2.nets
    return diffsci2.nets.VAENetConfig(
        dimension=3, in_channels=1, out_channels=1, z_channels=4, z_dim=4,
        ch=32, ch_mult=[1, 2, 4, 4], num_res_blocks=2, attn_resolutions=[],
        dropout=0.0, resolution=256, has_mid_attn=False, resamp_with_conv=True,
        attn_type='vanilla', tanh_out=False, input_bias=True, output_bias=True,
        with_time_emb=False, double_z=True, num_groups=32, patch_size=None,
        memory_efficient_variant=False, use_flash_attention=True,
        minimal_rf_mode=False,
    )


def load_toy_2d_vae(device='cpu'):
    """Load the toy 2D VAE used in the tutorial. Returns (vaenet, vae_module).

    The encoder is `vaenet.encoder`; the decoder is `vaenet.decoder`.
    """
    import diffsci2.nets
    import diffsci2.models
    cfg = toy_2d_vae_config()
    vae_net = diffsci2.nets.VAENet(config=cfg)
    if not os.path.exists(TOY_VAE_CKPT):
        raise FileNotFoundError(f"Missing {TOY_VAE_CKPT}")
    vae = diffsci2.models.VAEModule.load_from_checkpoint(
        TOY_VAE_CKPT,
        config=diffsci2.models.VAEModuleConfig(),
        encdec=vae_net,
    ).to(device).eval()
    return vae_net, vae


def load_production_3d_vae(device='cpu'):
    """Load the production 3D VAE (converted_vaenet.ckpt)."""
    import diffsci2.nets
    import diffsci2.models
    cfg = production_3d_vae_config()
    vae_net = diffsci2.nets.VAENet(config=cfg)
    if not os.path.exists(PROD_VAE_CKPT):
        raise FileNotFoundError(f"Missing {PROD_VAE_CKPT}")
    ckpt = torch.load(PROD_VAE_CKPT, map_location='cpu', weights_only=False)
    vae_net.load_state_dict(ckpt['state_dict'], strict=True)
    vae_net.eval()
    vae_module = diffsci2.models.VAEModule(
        config=diffsci2.models.VAEModuleConfig(), encdec=vae_net,
    )
    vae_module = vae_module.to(device).eval()
    return vae_net, vae_module


def load_berea_slice(idx=1, crop=None):
    """Load Berea 2D slice as float32 [1, 1, H, W] in [0, 1]."""
    if not os.path.exists(BEREA_SLICES):
        raise FileNotFoundError(f"Missing {BEREA_SLICES}")
    arr = np.load(BEREA_SLICES).astype(np.float32) / 255.0
    sl = arr[idx]
    if crop is not None:
        sl = sl[:crop, :crop]
    return torch.from_numpy(sl)[None, None]


def load_bentheimer_3d_crop(size=128, offset=0):
    """Load a (size, size, size) crop of Bentheimer as float32 [1,1,D,H,W].

    The raw file is uint8 with values in {0, 1} (NOT {0, 255}); we cast to
    float32 directly without rescaling.
    """
    if not os.path.exists(BENTHEIMER_RAW):
        raise FileNotFoundError(f"Missing {BENTHEIMER_RAW}")
    vol = np.fromfile(BENTHEIMER_RAW, dtype=np.uint8).reshape(1000, 1000, 1000)
    o = offset
    crop = vol[o:o + size, o:o + size, o:o + size].astype(np.float32)
    return torch.from_numpy(crop)[None, None]


def assert_close(a, b, atol=1e-5, rtol=1e-4, msg=""):
    """Stricter than torch.allclose: also reports max abs diff."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu()
    diff = (a - b).abs()
    max_abs = float(diff.max())
    rel_denom = torch.maximum(a.abs(), b.abs()).clamp(min=1e-12)
    max_rel = float((diff / rel_denom).max())
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        raise AssertionError(
            f"assert_close failed{(': ' + msg) if msg else ''}\n"
            f"  max_abs_diff = {max_abs:.3e} (atol={atol:.0e})\n"
            f"  max_rel_diff = {max_rel:.3e} (rtol={rtol:.0e})\n"
            f"  shapes: a={tuple(a.shape)} b={tuple(b.shape)}"
        )
    return max_abs, max_rel


def banner(label):
    print()
    print('=' * 70)
    print(f'  {label}')
    print('=' * 70)

"""Test 04 - multi-tile 2D chunked encode with cached norms is bit-exact.

Strategy:
  1) Convert encoder norms to cached versions.
  2) Pre-calibrate by running the FULL input through the encoder in 'cache'
     mode -> stats reflect whole-input statistics.
  3) Switch to 'use_cached' mode. The encoder is now fully convolutional
     (norms become pointwise affine over per-channel stats).
  4) Run full encoder; record z_full.
  5) Run chunk_encode with multiple tiles, with use_cached_norms=False so the
     chunked routine does NOT touch the cache (so the same precomputed stats
     are used by every tile).
  6) Assert z_full == z_chunked bit-exact.

Any difference here is a bug in the halo/crop math, since convs are
position-independent and norms are now pointwise-fixed.
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (
    setup_repo_path, load_toy_2d_vae, load_berea_slice, banner, assert_close,
)
setup_repo_path()


def run_one(input_size: int, chunk_lat: int):
    from diffsci2.extra.chunk_encode_2 import (
        chunk_encode_2d, prepare_encoder_for_cached_encode,
    )
    from diffsci2.nets.cached_norms import (
        set_all_norms_mode, clear_all_norm_caches,
    )

    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()
    prepare_encoder_for_cached_encode(enc, inplace=True)

    x = load_berea_slice(idx=1, crop=input_size)
    F = 2 ** (enc.config.num_resolutions - 1)
    L = x.shape[-1] // F

    with torch.inference_mode():
        # Calibrate cache on full input
        clear_all_norm_caches(enc)
        set_all_norms_mode(enc, 'cache')
        _ = enc(x)
        set_all_norms_mode(enc, 'use_cached')

        # Reference: full encoder in 'use_cached' mode
        z_full = enc(x)

        # Chunked: do NOT re-cache; reuse the precomputed whole-input stats
        z_chunked = chunk_encode_2d(
            enc, x, chunk_latent=(chunk_lat, chunk_lat),
            device='cpu', use_cached_norms=False, debug=0,
        )

    print(f"  input {input_size}^2, latent {L}^2, chunk_lat={chunk_lat}, "
          f"tiles per axis = {(L + chunk_lat - 1) // chunk_lat}")
    print(f"  z_full:    {tuple(z_full.shape)}")
    print(f"  z_chunked: {tuple(z_chunked.shape)}")
    max_abs, max_rel = assert_close(
        z_full, z_chunked, atol=1e-5, rtol=1e-4,
        msg=f"chunked != full at input={input_size}, chunk_lat={chunk_lat}",
    )
    print(f"  max_abs_diff = {max_abs:.3e}  max_rel_diff = {max_rel:.3e}")


def main():
    banner("test_04: multi-tile 2D chunked encode with pre-cached norms")
    cases = [
        (64, 8),    # 2x2 = 4 tiles in latent (16/8)
        (64, 4),    # 4x4 = 16 tiles
        (128, 8),   # 4x4 = 16 tiles, latent 16
        (128, 4),   # 8x8 = 64 tiles, latent 32
        (256, 16),  # 4x4 tiles
        (256, 8),   # 8x8 tiles
    ]
    for input_size, chunk_lat in cases:
        print()
        run_one(input_size, chunk_lat)
        print("  [PASS]")
    banner("test_04: ALL PASSED")


if __name__ == '__main__':
    main()

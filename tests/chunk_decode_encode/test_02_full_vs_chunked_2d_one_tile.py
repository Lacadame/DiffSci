"""Test 02 - chunked encode with one tile == full encoder forward.

When chunk_latent == latent shape, there is exactly one tile. Halo logic is
exercised at boundaries (clamping) but no inter-tile seams. This isolates
basic plumbing: input fetch, stage runners, buffer alloc, crop, write.

Two cases:
  A) without cached norms (encoder is in 'normal' mode, GroupNorm uses live
     stats; with a single tile the GroupNorm sees the same data as the full
     forward, so output should match).
  B) with cached norms (calibrated on the same input via first-tile cache).
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


def test_one_tile_no_cached_norms():
    banner("test_02.A: one-tile chunked encode == full encode (no cached norms)")
    from diffsci2.extra.chunk_encode_2 import chunk_encode_2d

    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()

    x = load_berea_slice(idx=1, crop=64)  # [1, 1, 64, 64]
    F = 2 ** (enc.config.num_resolutions - 1)
    L = x.shape[-1] // F  # 16
    print(f"  input shape: {tuple(x.shape)}, latent shape: ({L}, {L})")

    with torch.inference_mode():
        z_full = enc(x)
        z_chunked = chunk_encode_2d(
            enc, x, chunk_latent=(L, L),
            device='cpu', use_cached_norms=False, debug=0,
        )

    print(f"  z_full:    {tuple(z_full.shape)}")
    print(f"  z_chunked: {tuple(z_chunked.shape)}")
    max_abs, max_rel = assert_close(
        z_full, z_chunked, atol=1e-4, rtol=1e-3,
        msg="single-tile chunked encode != full encode (no cached norms)",
    )
    print(f"  max_abs_diff = {max_abs:.3e}, max_rel_diff = {max_rel:.3e}")
    print("  [PASS]")


def test_one_tile_cached_norms():
    banner("test_02.B: one-tile chunked encode == full encode (with cached norms)")
    from diffsci2.extra.chunk_encode_2 import (
        chunk_encode_2d, prepare_encoder_for_cached_encode,
    )
    from diffsci2.nets.cached_norms import (
        set_all_norms_mode, clear_all_norm_caches,
    )

    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()

    x = load_berea_slice(idx=1, crop=64)
    F = 2 ** (enc.config.num_resolutions - 1)
    L = x.shape[-1] // F

    prepare_encoder_for_cached_encode(enc, inplace=True)

    with torch.inference_mode():
        # Reference: full encoder in 'normal' mode (live stats from full input)
        set_all_norms_mode(enc, 'normal')
        clear_all_norm_caches(enc)
        z_full = enc(x)

        # Chunked: cached norms ON (will cache on the single tile, then use them)
        z_chunked = chunk_encode_2d(
            enc, x, chunk_latent=(L, L),
            device='cpu', use_cached_norms=True, debug=0,
        )

    max_abs, max_rel = assert_close(
        z_full, z_chunked, atol=1e-4, rtol=1e-3,
        msg="one-tile chunked encode (cached norms) != full encode",
    )
    print(f"  max_abs_diff = {max_abs:.3e}, max_rel_diff = {max_rel:.3e}")
    print("  [PASS]")


def main():
    test_one_tile_no_cached_norms()
    test_one_tile_cached_norms()
    banner("test_02: ALL PASSED")


if __name__ == '__main__':
    main()

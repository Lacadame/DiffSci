"""Test 03 - multi-tile chunked encode with first-tile-cache norms.

Realistic usage path: caller calls chunk_encode with use_cached_norms=True
and the routine handles per-stage first-tile caching internally. This is
the same strategy chunk_decode uses.

Because the first tile is a strict subset of the full input, its GroupNorm
statistics differ from full-input stats. For an ergodic-like input (Berea,
porosity ~ uniform across position), the difference should be small but
nonzero.

This test:
  - confirms the routine runs to completion;
  - checks output shape and finiteness;
  - asserts a small relative error vs full-encoder forward (loose tolerance).

For a tight bit-exact correctness check see test_04.
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (
    setup_repo_path, load_toy_2d_vae, load_berea_slice, banner,
)
setup_repo_path()


def main():
    banner("test_03: multi-tile chunked encode with first-tile-cache norms")
    from diffsci2.extra.chunk_encode_2 import (
        chunk_encode_2d, prepare_encoder_for_cached_encode,
    )

    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()
    prepare_encoder_for_cached_encode(enc, inplace=True)

    x = load_berea_slice(idx=1, crop=128)  # 128x128 Berea slice (ergodic)
    F = 2 ** (enc.config.num_resolutions - 1)
    L = x.shape[-1] // F

    with torch.inference_mode():
        # Reference: full encoder in 'normal' mode (live whole-input stats)
        z_full = enc(x)
        # Chunked with multiple tiles, internally caches from first tile
        z_chunked = chunk_encode_2d(
            enc, x, chunk_latent=(8, 8),
            device='cpu', use_cached_norms=True, debug=0,
        )

    assert z_chunked.shape == z_full.shape, f"shape mismatch: {z_chunked.shape} vs {z_full.shape}"
    assert torch.isfinite(z_chunked).all(), "z_chunked has non-finite values"

    diff = (z_chunked - z_full).abs()
    denom = z_full.abs().clamp(min=1e-3)
    rel = (diff / denom).max().item()
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()
    print(f"  z_full range: [{float(z_full.min()):.3f}, {float(z_full.max()):.3f}]")
    print(f"  abs diff:  max = {abs_max:.3e}, mean = {abs_mean:.3e}")
    print(f"  max rel diff (vs |z_full|>=1e-3): {rel:.3e}")
    # First-tile-cache uses stats from a 1/(num_tiles) subset of input, so
    # it does drift; we only enforce "not catastrophically broken" here.
    # The rigorous correctness check is test_04 (bit-exact via pre-cached norms).
    z_range = float(z_full.max() - z_full.min())
    assert abs_mean < 0.1 * z_range, (
        f"mean abs error {abs_mean} is more than 10% of z range {z_range}"
    )
    print(f"  mean abs / z range = {abs_mean / z_range:.2%}  (must be < 10%)")
    banner("test_03: PASSED")


if __name__ == '__main__':
    main()

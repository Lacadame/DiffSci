"""Test 05 - stage-by-stage equivalence (diagnostic).

For each stage k in 0..N: run the chunked encoder with max_stages=k+1 and
compare the resulting intermediate stage buffer to the full encoder run
through the same stages (composed via the per-stage runners).

This pinpoints which stage breaks if test_04 fails: a stage-0 bug shows up
at k=0 while a final-stage bug shows up only at k=N.

Uses pre-cached whole-input norms (same setup as test_04) so divergence
must come from halo / crop / buffer math.
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


def main():
    banner("test_05: stage-by-stage chunked encoder equals staged full encoder")
    from diffsci2.extra.chunk_encode_2 import (
        chunk_encode_2d, prepare_encoder_for_cached_encode,
        make_vae_encoder_stage_runner,
    )
    from diffsci2.nets.cached_norms import (
        set_all_norms_mode, clear_all_norm_caches,
    )

    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()
    prepare_encoder_for_cached_encode(enc, inplace=True)

    x = load_berea_slice(idx=1, crop=128)
    F = 2 ** (enc.config.num_resolutions - 1)
    L = x.shape[-1] // F
    N = enc.config.num_resolutions
    num_stages = N + 1

    with torch.inference_mode():
        # Pre-calibrate cache on full input
        clear_all_norm_caches(enc)
        set_all_norms_mode(enc, 'cache')
        _ = enc(x)
        set_all_norms_mode(enc, 'use_cached')

        # Reference: per-stage runners on the full input
        ref_intermediates = []
        h = x
        for k in range(num_stages):
            run = make_vae_encoder_stage_runner(enc, k, num_stages, None)
            h = run(h)
            ref_intermediates.append(h.clone())

        # Chunked: max_stages = k+1 for k in 0..N
        chunk_lat = 8
        for k in range(num_stages):
            z_chunked_partial = chunk_encode_2d(
                enc, x, chunk_latent=(chunk_lat, chunk_lat),
                device='cpu', use_cached_norms=False,
                max_stages=k + 1, debug=0,
            )
            ref = ref_intermediates[k]
            print(f"  stage {k}: full intermediate {tuple(ref.shape)}, "
                  f"chunked {tuple(z_chunked_partial.shape)}")
            max_abs, max_rel = assert_close(
                ref, z_chunked_partial, atol=1e-5, rtol=1e-4,
                msg=f"chunked != full at stage {k}",
            )
            print(f"    max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    banner("test_05: ALL STAGES PASSED")


if __name__ == '__main__':
    main()

"""Test 01 - encoder stage decomposition and receptive field.

Verifies:
1. The N+1 stage decomposition of VAEEncoder reconstructs the full forward.
   - S0  = conv_in + down[0].block[*]
   - Sk  = down[k-1].downsample + down[k].block[*]  for k = 1..N-1
   - SN  = mid.block_1 + mid.block_2 + norm_out + nonlin + conv_out + quant_conv
2. compute_encoder_stage_radii_and_scales returns the analytical per-stage
   cumulative radii (in INPUT pixel units) and scales.
3. The analytical input-space RF radius matches brute-force probing.
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (
    setup_repo_path, load_toy_2d_vae, banner, assert_close,
)
setup_repo_path()


def stage_runners_for_encoder(encoder):
    """Return a list of N+1 callables, each running one encoder stage.

    Mirrors the decoder helpers in chunk_decode_2.py. Tests that the natural
    cut reproduces encoder.forward when concatenated.
    """
    from diffsci2.extra.chunk_encode_2 import (
        run_vae_enc_stage0,
        run_vae_enc_down_stage,
        run_vae_enc_final_stage,
    )
    cfg = encoder.config
    N = cfg.num_resolutions
    runners = [lambda x: run_vae_enc_stage0(encoder, x, None)]
    for k in range(1, N):
        runners.append(
            (lambda kk: lambda x: run_vae_enc_down_stage(encoder, x, kk, None))(k)
        )
    runners.append(lambda x: run_vae_enc_final_stage(encoder, x, None))
    return runners


def test_stage_decomposition():
    banner("test_01.A: stage decomposition reconstructs encoder.forward")
    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()

    x = torch.randn(1, 1, 64, 64)
    with torch.inference_mode():
        full = enc(x)

        runners = stage_runners_for_encoder(enc)
        h = x
        for k, r in enumerate(runners):
            h = r(h)
            print(f"  after stage {k}: shape = {tuple(h.shape)}")
        composed = h

    print(f"  full encoder out: {tuple(full.shape)}")
    print(f"  composed staged:  {tuple(composed.shape)}")
    assert_close(full, composed, atol=1e-5, rtol=1e-4,
                 msg="stage-runner composition != encoder.forward")
    print("  [PASS] stage runners reconstruct encoder.forward")


def test_analytical_radii():
    banner("test_01.B: analytical per-stage radii (toy 2D)")
    from diffsci2.extra.chunk_encode_2 import (
        compute_encoder_stage_radii_and_scales,
    )
    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    delta, radii_cum, scales_after, scales_src = (
        compute_encoder_stage_radii_and_scales(enc)
    )
    # Toy config: N=3, R=2, F=4
    expected_delta       = [5, 9, 18, 20]
    expected_radii_cum   = [5, 14, 32, 52]
    expected_scales_after = [1, 2, 4, 4]
    expected_scales_src  = [1, 1, 2, 4]
    print(f"  delta             = {delta}")
    print(f"  radii_cum (input) = {radii_cum}")
    print(f"  scales_after      = {scales_after}")
    print(f"  scales_src        = {scales_src}")
    assert delta == expected_delta, f"delta mismatch: got {delta}, expected {expected_delta}"
    assert radii_cum == expected_radii_cum, f"radii_cum mismatch: got {radii_cum}, expected {expected_radii_cum}"
    assert scales_after == expected_scales_after, f"scales_after mismatch: got {scales_after}, expected {expected_scales_after}"
    assert scales_src == expected_scales_src, f"scales_src mismatch: got {scales_src}, expected {expected_scales_src}"
    print("  [PASS] analytical per-stage values match expectation")


def test_brute_force_rf():
    banner("test_01.C: brute-force RF radius matches analytical (toy 2D)")
    from diffsci2.nets.cached_norms import (
        convert_to_cached_norms, set_all_norms_mode,
    )
    from diffsci2.extra.chunk_encode_2 import (
        compute_encoder_stage_radii_and_scales,
    )
    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()

    # Cached norms mode -> RF becomes finite (otherwise GroupNorm makes RF inf).
    convert_to_cached_norms(enc, inplace=True)
    set_all_norms_mode(enc, 'cache')
    with torch.inference_mode():
        _ = enc(torch.randn(1, 1, 256, 256))
    set_all_norms_mode(enc, 'use_cached')

    H = 256
    # Perturb a single input pixel and measure how far in latent space the
    # influence reaches; convert to required input-space halo.
    with torch.inference_mode():
        base = enc(torch.zeros(1, 1, H, H))
        Hc = H // 2
        x = torch.zeros(1, 1, H, H)
        x[0, 0, Hc, Hc] = 1.0
        z = enc(x)
        diff_map = (z - base).abs().sum(dim=(0, 1))  # [Hl, Wl]
        nz = (diff_map > 1e-7).nonzero()
        if len(nz) == 0:
            raise AssertionError("perturbation had zero effect — encoder may be degenerate")
        F_factor = 2 ** (enc.config.num_resolutions - 1)
        # required radius (in input pixels) = max over affected latents of |L*F - Hc|
        r_in_required = int(max(
            abs(int(nz[:, 0].max()) * F_factor - Hc),
            abs(int(nz[:, 0].min()) * F_factor - Hc),
            abs(int(nz[:, 1].max()) * F_factor - Hc),
            abs(int(nz[:, 1].min()) * F_factor - Hc),
        ))

    _, radii_cum, _, _ = compute_encoder_stage_radii_and_scales(enc)
    analytical_radius = radii_cum[-1]
    print(f"  brute-force required input halo radius: {r_in_required}")
    print(f"  analytical total input RF radius:       {analytical_radius}")
    assert analytical_radius >= r_in_required, (
        f"analytical RF {analytical_radius} smaller than brute-force {r_in_required} - unsafe"
    )
    # Sanity: not absurdly larger either
    assert analytical_radius <= r_in_required + 4, (
        f"analytical RF {analytical_radius} much larger than brute-force {r_in_required} - bad bound"
    )
    print("  [PASS] analytical RF is a tight upper bound")


def main():
    test_stage_decomposition()
    test_analytical_radii()
    test_brute_force_rf()
    banner("test_01: ALL PASSED")


if __name__ == '__main__':
    main()

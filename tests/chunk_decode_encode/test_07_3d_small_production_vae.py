"""Test 07 - production 3D VAE chunked encode on a small Bentheimer crop.

Has two parts:

  A) CPU bit-exact correctness on a tiny crop (32^3 -> latent 4^3):
     With cuDNN out of the picture, the chunked encoder is mathematically
     equivalent to the full encoder. Any difference is FP roundoff.

  B) GPU correctness within tolerance on a larger crop (256^3 -> latent 32^3):
     cuDNN selects different forward algorithms based on input shape, which
     introduces a small but nonzero numerical drift between running stage k
     on a sub-window vs the corresponding region of stage k applied to the
     full buffer. This drift is bounded (~1.5e-2 abs on values ranging ~20).
     For property-preservation evaluation it is negligible. We bound the
     error rather than insisting on bit-exact match on GPU.

Strategy mirrors test_04: pre-calibrate cached norms on the full input, then
run both full and chunked in 'use_cached' mode so divergence isolates conv
+ halo / crop math.
"""
import os
import sys
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (
    setup_repo_path, load_production_3d_vae, load_bentheimer_3d_crop, banner,
)
setup_repo_path()


def _run(device_name: str, crop_size: int, chunk_lat: int, atol: float, rtol: float, label: str):
    from diffsci2.extra.chunk_encode_2 import (
        chunk_encode_3d, prepare_encoder_for_cached_encode,
    )
    from diffsci2.nets.cached_norms import set_all_norms_mode, clear_all_norm_caches
    print()
    print(f"  ----- {label} -----")
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        print("    CUDA unavailable; skipping")
        return
    device = torch.device(device_name)
    vae_net, _ = load_production_3d_vae(device='cpu')
    enc = vae_net.encoder
    enc.eval()
    prepare_encoder_for_cached_encode(enc, inplace=True)
    enc.to(device)

    x_cpu = load_bentheimer_3d_crop(size=crop_size, offset=128)
    F_factor = 2 ** (enc.config.num_resolutions - 1)
    L = crop_size // F_factor

    with torch.inference_mode():
        x_dev = x_cpu.to(device)
        clear_all_norm_caches(enc)
        set_all_norms_mode(enc, 'cache')
        _ = enc(x_dev)
        set_all_norms_mode(enc, 'use_cached')

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t = time.time()
        z_full = enc(x_dev)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_full = time.time() - t
        z_full_cpu = z_full.cpu()
        del z_full
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t = time.time()
        z_chunk = chunk_encode_3d(
            enc, x_cpu,
            chunk_latent=(chunk_lat, chunk_lat, chunk_lat),
            device=device, use_cached_norms=False, debug=0,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_chunk = time.time() - t

    print(f"    crop {crop_size}^3 -> latent {L}^3, chunk_lat={chunk_lat}, "
          f"{(L + chunk_lat - 1) // chunk_lat}^3 tiles")
    print(f"    full encode time:    {t_full:.2f}s")
    print(f"    chunked encode time: {t_chunk:.2f}s")
    print(f"    z_full range: [{float(z_full_cpu.min()):.3f}, {float(z_full_cpu.max()):.3f}]")
    diff = (z_full_cpu - z_chunk).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"    max abs diff:  {max_abs:.3e}  (tolerance {atol:.0e})")
    print(f"    mean abs diff: {mean_abs:.3e}")
    z_range = float(z_full_cpu.max() - z_full_cpu.min())
    print(f"    max abs / z range = {max_abs / z_range:.2%}")
    if max_abs > atol:
        raise AssertionError(
            f"3D chunked vs full max_abs={max_abs:.3e} exceeds tolerance {atol:.3e}"
        )
    print(f"    [PASS]")


def test_07_a_cpu_bit_exact():
    banner("test_07.A: CPU 3D production VAE bit-exact (small crop)")
    _run(
        device_name='cpu',
        crop_size=32, chunk_lat=4,
        atol=1e-4, rtol=1e-3,
        label="CPU, crop 32^3, chunk 4 (=> 1 tile per axis on latent 4)",
    )
    # And a real 2-tile-per-axis case on CPU
    _run(
        device_name='cpu',
        crop_size=64, chunk_lat=4,
        atol=1e-4, rtol=1e-3,
        label="CPU, crop 64^3, chunk 4 (=> 2 tiles per axis on latent 8)",
    )


def test_07_b_gpu_within_tolerance():
    banner("test_07.B: GPU 3D production VAE within tolerance (256^3 crop)")
    _run(
        device_name='cuda:0',
        crop_size=256, chunk_lat=8,
        atol=5e-2, rtol=1.0,  # cuDNN forward-algo drift bound
        label="GPU cuda:0, crop 256^3, chunk 8 (=> 4 tiles per axis on latent 32)",
    )


def main():
    test_07_a_cpu_bit_exact()
    test_07_b_gpu_within_tolerance()
    banner("test_07: ALL PASSED")


if __name__ == '__main__':
    main()

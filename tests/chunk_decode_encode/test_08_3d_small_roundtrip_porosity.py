"""Test 08 - 3D small roundtrip porosity preservation.

Encode a small Bentheimer crop with chunk_encode_3d, decode the resulting
mean (z_dim channels of the encoder output) with chunk_decode_3d, and
compare the reconstructed porosity to the original. This is the property
the user actually cares about for evaluation.

We expect:
  - The reconstruction is not pixel-perfect (VAE compression is lossy).
  - The bulk porosity should be preserved within a few percent.
  - For binary porous media, after thresholding the reconstruction at 0.5
    we expect bulk porosity within ~5%.

Uses cuda:0. Small crop only (256^3 -> 32^3 latent).
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


def _run_roundtrip(crop_size: int, enc_chunk_lat, dec_chunk_lat, label: str):
    from diffsci2.extra.chunk_encode_2 import (
        chunk_encode_3d, prepare_encoder_for_cached_encode,
    )
    from diffsci2.extra.chunk_decode_2 import (
        chunk_decode_3d, prepare_decoder_for_cached_decode,
    )
    from diffsci2.nets.cached_norms import set_all_norms_mode, clear_all_norm_caches

    print()
    print(f"  ----- {label} -----")
    device = torch.device('cuda:0')
    # Load production VAE
    vae_net, _ = load_production_3d_vae(device='cpu')
    enc = vae_net.encoder
    dec = vae_net.decoder
    enc.eval()
    dec.eval()
    prepare_encoder_for_cached_encode(enc, inplace=True)
    prepare_decoder_for_cached_decode(dec, inplace=True)
    enc.to(device)
    # decoder is moved to device by chunk_decode internally / prior to call

    x_cpu = load_bentheimer_3d_crop(size=crop_size, offset=128)
    F = 2 ** (enc.config.num_resolutions - 1)
    L = crop_size // F  # 16
    porosity_truth = 1.0 - float(x_cpu.mean())  # solid=1, pore=0 -> phi = 1 - mean
    print(f"  crop {crop_size}^3, latent {L}^3")
    print(f"  ground-truth porosity (1 - mean) = {porosity_truth:.4f}")

    # ---- Encode -----------------------------------------------------------
    with torch.inference_mode():
        x_gpu = x_cpu.to(device)
        clear_all_norm_caches(enc)
        set_all_norms_mode(enc, 'cache')
        _ = enc(x_gpu)
        set_all_norms_mode(enc, 'use_cached')
        del x_gpu
        torch.cuda.empty_cache()

        t = time.time()
        z = chunk_encode_3d(
            enc, x_cpu, chunk_latent=enc_chunk_lat,
            device=device, use_cached_norms=False, debug=0,
        )
        torch.cuda.synchronize(device)
        t_enc = time.time() - t
    z_dim = enc.config.z_dim
    z_mean = z[:, :z_dim]  # take mean half (drop logvar)
    print(f"  encoded mean shape: {tuple(z_mean.shape)}")
    print(f"  encode time: {t_enc:.2f}s")

    # ---- Decode -----------------------------------------------------------
    enc.cpu()
    torch.cuda.empty_cache()
    dec.to(device)
    with torch.inference_mode():
        # Calibrate decoder cached norms on the real latent
        clear_all_norm_caches(dec)
        set_all_norms_mode(dec, 'cache')
        _ = dec(z_mean.to(device))
        set_all_norms_mode(dec, 'use_cached')

        t = time.time()
        x_recon = chunk_decode_3d(
            dec, z_mean.to(device),
            list(dec_chunk_lat),
            device=device,
            use_cached_norms=False,
        )
        torch.cuda.synchronize(device)
        t_dec = time.time() - t
    print(f"  reconstruction shape: {tuple(x_recon.shape)}")
    print(f"  decode time: {t_dec:.2f}s")

    # ---- Compare ----------------------------------------------------------
    x_recon_cpu = x_recon[0, 0].float()
    x_orig_cpu = x_cpu[0, 0].float()

    print(f"  recon range: [{float(x_recon_cpu.min()):.3f}, {float(x_recon_cpu.max()):.3f}]")
    print(f"  recon mean:  {float(x_recon_cpu.mean()):.4f}")

    # Float MSE (without thresholding)
    mse = ((x_recon_cpu - x_orig_cpu) ** 2).mean().item()
    print(f"  float MSE (recon vs orig):  {mse:.4e}")

    # Threshold reconstruction at 0.5 -> binary
    x_recon_binary = (x_recon_cpu > 0.5).float()
    porosity_recon = 1.0 - float(x_recon_binary.mean())
    print(f"  reconstructed porosity (after threshold): {porosity_recon:.4f}")

    # Voxel-wise agreement
    agree = (x_recon_binary == x_orig_cpu).float().mean().item()
    print(f"  voxel agreement (binary): {agree:.4f}")

    porosity_err = abs(porosity_recon - porosity_truth)
    print(f"  porosity error |recon - truth|: {porosity_err:.4f}")

    assert torch.isfinite(x_recon_cpu).all(), "reconstruction has non-finite values"
    assert porosity_err < 0.10, (
        f"porosity error {porosity_err:.4f} > 0.10 - reconstruction looks broken"
    )


def main():
    banner("test_08: 3D small roundtrip porosity")
    if not torch.cuda.is_available():
        print("  CUDA unavailable; SKIPPING")
        return
    # Single tile (fastest, sanity)
    _run_roundtrip(
        crop_size=128, enc_chunk_lat=(16, 16, 16), dec_chunk_lat=(16, 16, 16),
        label="single tile, 128^3 -> 16^3 latent",
    )
    # Multi-tile chunk_encode (2x2x2 tiles), single-tile decode
    _run_roundtrip(
        crop_size=128, enc_chunk_lat=(8, 8, 8), dec_chunk_lat=(16, 16, 16),
        label="multi-tile chunk_encode (2x2x2), single-tile decode",
    )
    banner("test_08: ALL PASSED")


if __name__ == '__main__':
    main()

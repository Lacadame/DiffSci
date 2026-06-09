#!/usr/bin/env python
"""
Reconstruction roundtrip on Imperial College stones via the production VAE.

For each Imperial College stone (Bentheimer, Doddington, Estaillades, Ketton):
  1) Load the 1000^3 raw uint8 binary volume.
  2) chunk_encode_3d -> latent (4, 125, 125, 125) on cuda:0.
  3) chunk_decode_3d -> float reconstruction (1000, 1000, 1000) on cuda:0.
  4) Save latent (float32), float reconstruction (float16) and binarised
     reconstruction (uint8 in {0, 1}) to OUTPUT_DIR.

Outputs per stone (in OUTPUT_DIR):
    {stone}_latent.npy        float32 [4, 125, 125, 125]
    {stone}_recon_float.npy   float16 [1000, 1000, 1000]
    {stone}_recon.npy         uint8 (binary {0, 1}) [1000, 1000, 1000]
plus a single summary.json with timing and bulk-porosity comparison.

This uses chunk_encode/chunk_decode WITH first-tile cached norms (the same
strategy 0004d uses for decode), which works because the porous media are
~ergodic.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Make repo importable + locate aux/model_loaders.py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

from model_loaders import load_autoencoder  # noqa: E402
from diffsci2.extra.chunk_encode_2 import (  # noqa: E402
    chunk_encode_3d,
    prepare_encoder_for_cached_encode,
)
from diffsci2.extra.chunk_decode_2 import (  # noqa: E402
    chunk_decode_3d,
    prepare_decoder_for_cached_decode,
)


REFERENCE_PATHS = {
    'Bentheimer':  '/home/ubuntu/repos/DiffSci2/saveddata/raw/imperial_college/Bentheimer_1000c_3p0035um.raw',
    'Doddington':  '/home/ubuntu/repos/DiffSci2/saveddata/raw/imperial_college/Doddington_1000c_2p6929um.raw',
    'Estaillades': '/home/ubuntu/repos/DiffSci2/saveddata/raw/imperial_college/Estaillades_1000c_3p31136um.raw',
    'Ketton':      '/home/ubuntu/repos/DiffSci2/saveddata/raw/imperial_college/Ketton_1000c_3p00006um.raw',
}

OUTPUT_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/stones_reconstruction_test'

VOLUME_SHAPE = (1000, 1000, 1000)  # raw layout
F = 8                              # production VAE downsampling factor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stones', nargs='+', default=list(REFERENCE_PATHS.keys()),
                   choices=list(REFERENCE_PATHS.keys()),
                   help='Subset of stones to process (default: all four)')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--chunk-lat', type=int, default=40,
                   help='Tile size in latent units (default 40, matches 0004d).')
    p.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Threshold for binarisation of float reconstruction.')
    p.add_argument('--debug', type=int, default=1,
                   help='chunk_encode/decode debug verbosity (0/1/2/3).')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip a stone if its _recon.npy already exists.')
    return p.parse_args()


def load_raw_volume(path: str) -> torch.Tensor:
    """Read a 1000^3 uint8 raw file (values in {0, 1}). Returns [1, 1, D, H, W] float32."""
    arr = np.fromfile(path, dtype=np.uint8).reshape(VOLUME_SHAPE)
    return torch.from_numpy(arr.astype(np.float32))[None, None]


def encode_volume(enc, x_cpu, device, chunk_lat, debug):
    """Run chunked encode with first-tile-cache norms. Returns mean latent [1, z_dim, *]."""
    enc.to(device)
    with torch.inference_mode():
        z = chunk_encode_3d(
            enc, x_cpu,
            chunk_latent=(chunk_lat, chunk_lat, chunk_lat),
            device=device,
            use_cached_norms=True,
            debug=debug,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    z_dim = enc.config.z_dim
    return z[:, :z_dim]  # mean half only


def decode_latent(dec, latent, device, chunk_lat, debug):
    """Run chunked decode with first-tile-cache norms. Returns reconstruction [1, 1, *]."""
    dec.to(device)
    with torch.inference_mode():
        recon = chunk_decode_3d(
            dec, latent.to(device),
            [chunk_lat, chunk_lat, chunk_lat],
            device=device,
            use_cached_norms=True,
            debug=debug,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    return recon


def process_stone(stone, args, vae, device, summary):
    """Process a single stone end-to-end. Mutates `summary[stone]` in place."""
    raw_path = REFERENCE_PATHS[stone]
    out_dir = args.output_dir
    s = stone.lower()

    latent_path = os.path.join(out_dir, f'{s}_latent.npy')
    recon_float_path = os.path.join(out_dir, f'{s}_recon_float.npy')
    recon_path = os.path.join(out_dir, f'{s}_recon.npy')

    if args.skip_existing and os.path.exists(recon_path):
        print(f"\n=== {stone} (SKIP — {recon_path} already exists) ===", flush=True)
        return

    print(f"\n=== {stone} ===", flush=True)
    print(f"  raw: {raw_path}", flush=True)

    t_total = time.time()

    # --- Load raw volume ---
    t = time.time()
    x_cpu = load_raw_volume(raw_path)
    porosity_truth = 1.0 - float(x_cpu.mean())
    t_load = time.time() - t
    print(f"  loaded volume {tuple(x_cpu.shape)} in {t_load:.1f}s, "
          f"porosity (1 - mean) = {porosity_truth:.4f}",
          flush=True)

    # --- Encode ---
    enc = vae.encoder
    dec = vae.decoder
    dec.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f"  encoding...", flush=True)
    t = time.time()
    latent = encode_volume(enc, x_cpu, device, args.chunk_lat, args.debug)
    t_enc = time.time() - t
    print(f"  encoded -> latent shape {tuple(latent.shape)} in {t_enc:.1f}s", flush=True)

    # Save latent
    latent_arr = latent[0].detach().cpu().numpy().astype(np.float32)
    np.save(latent_path, latent_arr)
    print(f"  saved {latent_path} ({latent_arr.nbytes / 1e6:.1f} MB)", flush=True)

    # Free encoder, free input volume
    enc.cpu()
    del x_cpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # --- Decode ---
    print(f"  decoding...", flush=True)
    t = time.time()
    recon = decode_latent(dec, latent, device, args.chunk_lat, args.debug)
    t_dec = time.time() - t
    recon_np = recon[0, 0].detach().cpu().numpy()  # [1000, 1000, 1000] float32
    print(f"  decoded -> reconstruction shape {tuple(recon_np.shape)} in {t_dec:.1f}s",
          flush=True)
    print(f"    recon range: [{recon_np.min():.3f}, {recon_np.max():.3f}], "
          f"mean: {recon_np.mean():.4f}", flush=True)
    del recon, latent
    dec.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Save float reconstruction (float16 to halve disk usage)
    recon_f16 = recon_np.astype(np.float16)
    np.save(recon_float_path, recon_f16)
    print(f"  saved {recon_float_path} ({recon_f16.nbytes / 1e9:.2f} GB)", flush=True)
    del recon_f16

    # Binarise + save
    recon_binary = (recon_np > args.threshold).astype(np.uint8)
    np.save(recon_path, recon_binary)
    porosity_recon = 1.0 - float(recon_binary.mean())
    print(f"  saved {recon_path} ({recon_binary.nbytes / 1e9:.2f} GB)", flush=True)
    print(f"  reconstructed porosity (after threshold {args.threshold}): "
          f"{porosity_recon:.4f}",
          flush=True)
    print(f"  porosity error |truth - recon| = "
          f"{abs(porosity_recon - porosity_truth):.4f}",
          flush=True)

    summary[stone] = {
        'raw_path': raw_path,
        'latent_path': latent_path,
        'recon_float_path': recon_float_path,
        'recon_path': recon_path,
        'porosity_truth': porosity_truth,
        'porosity_recon': porosity_recon,
        'porosity_error': abs(porosity_recon - porosity_truth),
        'recon_float_min': float(recon_np.min()),
        'recon_float_max': float(recon_np.max()),
        'recon_float_mean': float(recon_np.mean()),
        'time_load_s': t_load,
        'time_encode_s': t_enc,
        'time_decode_s': t_dec,
        'time_total_s': time.time() - t_total,
        'chunk_lat': args.chunk_lat,
        'threshold': args.threshold,
    }
    print(f"  total time for {stone}: {summary[stone]['time_total_s']:.1f}s", flush=True)
    del recon_np, recon_binary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(device)}, "
              f"memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB",
              flush=True)

    print(f"output dir: {args.output_dir}", flush=True)
    print(f"chunk_lat: {args.chunk_lat}", flush=True)
    print(f"stones to process: {args.stones}", flush=True)

    # Load production VAE (encoder + decoder share weights from converted_vaenet.ckpt)
    print("\nloading production VAE...", flush=True)
    t = time.time()
    vae = load_autoencoder()
    vae.eval()
    prepare_encoder_for_cached_encode(vae.encoder, inplace=True)
    prepare_decoder_for_cached_decode(vae.decoder, inplace=True)
    print(f"  loaded in {time.time() - t:.1f}s", flush=True)

    summary = {
        'output_dir': args.output_dir,
        'chunk_lat': args.chunk_lat,
        'threshold': args.threshold,
        'device': args.device,
        'stones_processed': [],
    }
    summary_path = os.path.join(args.output_dir, 'summary.json')

    for stone in args.stones:
        try:
            process_stone(stone, args, vae, device, summary)
            summary['stones_processed'].append(stone)
        except Exception as e:
            import traceback
            print(f"\n  *** ERROR on {stone}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            summary[f'{stone}_error'] = f"{type(e).__name__}: {e}"

        # Save partial summary after each stone (so a crash mid-way leaves data)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  partial summary saved to {summary_path}", flush=True)

    print("\n=== DONE ===", flush=True)
    print(f"summary: {summary_path}", flush=True)
    for stone in args.stones:
        if stone in summary:
            s = summary[stone]
            print(f"  {stone}: porosity truth={s['porosity_truth']:.4f}, "
                  f"recon={s['porosity_recon']:.4f}, "
                  f"err={s['porosity_error']:.4f}, "
                  f"total time={s['time_total_s']:.1f}s",
                  flush=True)


if __name__ == '__main__':
    main()

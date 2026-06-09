#!/usr/bin/env python
"""
Volume generator with porosity field conditioning from REAL training data.

Unlike 0004-porosity-field-generator.py which samples porosity from a fitted GP,
this script uses the ACTUAL calculated porosity field from the training data.

For sizes <= 1000 (the experimental volume size):
  - Extract a random subvolume from the actual porosity field

For sizes > 1000:
  - Extract a 1000^3 subvolume and pad with 'reflect' mode to reach target size

Usage:
    python scripts/0006-porosity-field-generator-from-training.py \
        --checkpoint /path/to/model.ckpt \
        --stone Estaillades \
        --output-dir ./generated_data/ \
        --volume-sizes 256,512,1024,1280 \
        --volume-samples 64,8,2,1 \
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Add the aux directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
from model_loaders import load_flow_model, load_autoencoder
from diffsci2.extra import chunk_decode_2, punetg_converters


AVAILABLE_STONES = ['Bentheimer', 'Doddington', 'Estaillades', 'Ketton']

# Constants
LATENT_TO_PIXEL_FACTOR = 8  # pixel_size = latent_size * 8
MIN_LATENT_MULTIPLE = 16    # latent_size must be multiple of 16
DOWNSAMPLE_FACTOR = 8       # Same as LATENT_TO_PIXEL_FACTOR

# Porosity field volumes (precomputed local mean porosity, 1000x1000x1000)
POROSITY_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/gpdata2/'
POROSITY_PATHS = {
    'Bentheimer': POROSITY_DIR + 'bentheimer/Bentheimer_1000c_3p0035um_porosity_field_full.npy',
    'Doddington': POROSITY_DIR + 'doddington/Doddington_1000c_2p6929um_porosity_field_full.npy',
    'Estaillades': POROSITY_DIR + 'estaillades/Estaillades_1000c_3p31136um_porosity_field_full.npy',
    'Ketton': POROSITY_DIR + 'ketton/Ketton_1000c_3p00006um_porosity_field_full.npy',
}

# Size of the experimental porosity field
POROSITY_FIELD_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate volumes with real porosity field conditioning from training data'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--stone', type=str, required=True, choices=AVAILABLE_STONES,
        help='Stone type for porosity field'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for generated volumes'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for computation'
    )
    parser.add_argument(
        '--nsteps', type=int, default=21,
        help='Number of sampling steps (default: 21)'
    )
    parser.add_argument(
        '--guidance', type=float, default=1.0,
        help='Classifier-free guidance scale (default: 1.0, no guidance)'
    )
    parser.add_argument(
        '--volume-sizes', type=str, default='256,512,1024',
        help='Comma-separated list of volume sizes to generate (must be multiples of 128)'
    )
    parser.add_argument(
        '--volume-samples', type=str, default='64,8,1',
        help='Comma-separated list of number of samples per size'
    )
    parser.add_argument(
        '--save-porosity', action='store_true',
        help='Save the input porosity field alongside each volume'
    )
    parser.add_argument(
        '--periodic', action='store_true',
        help='Generate periodic (circular) volumes'
    )
    parser.add_argument(
        '--no-binarize', action='store_true',
        help='Save raw float values instead of binarized (bool) volumes'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def parse_int_list(s):
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(',')]


def validate_volume_size(pixel_size):
    """Validate that pixel size corresponds to valid latent size."""
    if pixel_size % (LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE) != 0:
        raise ValueError(
            f"Volume size {pixel_size} must be a multiple of "
            f"{LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE} (128). "
            f"Valid examples: 256, 384, 512, 640, 768, 896, 1024, 1152, ..."
        )
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    return latent_size


def load_porosity_field(stone: str):
    """Load the precomputed porosity field for a stone type."""
    path = POROSITY_PATHS[stone]
    print(f"Loading porosity field from {path}...")
    porosity_field = np.load(path)
    print(f"  Shape: {porosity_field.shape}")
    print(f"  Range: [{porosity_field.min():.4f}, {porosity_field.max():.4f}]")
    return porosity_field


def sample_porosity_subvolume(porosity_field, pixel_size, rng):
    """
    Sample a porosity subvolume from the real porosity field.

    For sizes <= POROSITY_FIELD_SIZE (1000):
        Extract a random subvolume directly.

    For sizes > POROSITY_FIELD_SIZE:
        Extract the full field and pad with 'reflect' to reach target size.

    Parameters
    ----------
    porosity_field : ndarray
        Full porosity field of shape (1000, 1000, 1000).
    pixel_size : int
        Target pixel size for the generated volume.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    porosity_subvolume : ndarray
        Porosity field at pixel resolution (pixel_size, pixel_size, pixel_size).
    """
    field_size = porosity_field.shape[0]  # Should be 1000

    if pixel_size <= field_size:
        # Extract a random subvolume
        max_start = field_size - pixel_size
        start_d = rng.integers(0, max_start + 1) if max_start > 0 else 0
        start_h = rng.integers(0, max_start + 1) if max_start > 0 else 0
        start_w = rng.integers(0, max_start + 1) if max_start > 0 else 0

        porosity_subvolume = porosity_field[
            start_d:start_d + pixel_size,
            start_h:start_h + pixel_size,
            start_w:start_w + pixel_size
        ].copy()
    else:
        # Need to pad: extract full field and pad with reflect
        pad_total = pixel_size - field_size
        # Pad evenly on both sides, with extra on the end if odd
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        # Use numpy pad with reflect mode
        porosity_subvolume = np.pad(
            porosity_field,
            pad_width=((pad_before, pad_after), (pad_before, pad_after), (pad_before, pad_after)),
            mode='reflect'
        )

    return porosity_subvolume.astype(np.float32)


def downsample_porosity_to_latent(porosity_pixel, downsample_factor=DOWNSAMPLE_FACTOR):
    """
    Downsample porosity field from pixel resolution to latent resolution.

    Uses avg_pool3d to match the VAE compression.

    Parameters
    ----------
    porosity_pixel : ndarray
        Porosity field at pixel resolution [D, H, W].
    downsample_factor : int
        Factor to downsample (default: 8 to match VAE).

    Returns
    -------
    porosity_latent : ndarray
        Porosity field at latent resolution [D//factor, H//factor, W//factor].
    """
    # Convert to tensor with batch and channel dims: [1, 1, D, H, W]
    porosity_tensor = torch.from_numpy(porosity_pixel).float()
    porosity_tensor = porosity_tensor.unsqueeze(0).unsqueeze(0)

    # Apply 3D average pooling
    porosity_downsampled = F.avg_pool3d(
        porosity_tensor,
        kernel_size=downsample_factor,
        stride=downsample_factor
    )

    # Remove batch and channel dims: [D', H', W']
    porosity_latent = porosity_downsampled.squeeze(0).squeeze(0).numpy()

    return porosity_latent


def load_models(checkpoint_path, device, periodic=False):
    """Load flow model and autoencoder."""
    flowmodel = load_flow_model(checkpoint_path, custom_checkpoint_path=True)
    vaemodule = load_autoencoder()

    flowmoduleconfig = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        initial_norm=20.0,
        loss_formulation='denoiser'
    )
    flowmodule = diffsci2.models.SIModule(
        config=flowmoduleconfig,
        model=flowmodel,
        autoencoder=vaemodule
    )

    # Convert to circular convolutions for periodic generation
    if periodic:
        flowmodule.model = punetg_converters.convert_conv_to_circular(
            flowmodule.model, [0, 1, 2], True
        )

    flowmodule.to(device)
    return flowmodule, vaemodule


def generate_volume(flowmodule, vaemodule, porosity_latent, pixel_size, nsteps, device, guidance=1.0, periodic=False):
    """
    Generate a single volume conditioned on the given porosity field.

    Parameters
    ----------
    flowmodule : SIModule
        Flow model module.
    vaemodule : VAEModule
        Autoencoder module.
    porosity_latent : ndarray
        Porosity field at latent resolution [latent_size, latent_size, latent_size].
    pixel_size : int
        Target volume size in pixels.
    nsteps : int
        Number of diffusion steps.
    device : str
        Device for computation.
    guidance : float
        Classifier-free guidance scale.
    periodic : bool
        Whether to use periodic (circular) generation.

    Returns
    -------
    volume : ndarray
        Generated volume of shape (pixel_size, pixel_size, pixel_size).
    """
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR

    # Prepare conditioning
    porosity_tensor = torch.tensor(porosity_latent, dtype=torch.float32)
    y = {'porosity': porosity_tensor}

    # For 256 (latent 32), we can sample directly without chunk decode
    # For larger sizes, we need chunk decode
    use_chunk_decode = (pixel_size > 256)

    # Set periodicity for chunk decode
    periodicity = [True, True, True] if periodic else [False, False, False]

    if use_chunk_decode:
        # Sample in latent space
        x_latent = flowmodule.sample(
            1, shape=[4, latent_size, latent_size, latent_size],
            y=y, nsteps=nsteps,
            is_latent_shape=True, return_latents=True, guidance=guidance
        )

        # Chunk decode
        chunk_decode_2.prepare_decoder_for_cached_decode(vaemodule.decoder)
        vaemodule.decoder.to(device)

        chunk_size = [40, 40, 40]

        x = chunk_decode_2.chunk_decode_3d(
            vaemodule.decoder,
            x_latent,
            chunk_size,
            device=device,
            periodicity=periodicity,
            use_cached_norms=True
        )
        x = x[0][0].cpu().numpy()
    else:
        # Direct sampling at pixel resolution (for 256)
        x = flowmodule.sample(1, shape=[1, pixel_size, pixel_size, pixel_size], y=y, nsteps=nsteps, guidance=guidance)
        x = x[0][0].cpu().numpy()

    torch.cuda.empty_cache()
    return x


def main():
    args = parse_args()

    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Parse volume sizes and samples
    volume_sizes = parse_int_list(args.volume_sizes)
    volume_samples = parse_int_list(args.volume_samples)

    if len(volume_sizes) != len(volume_samples):
        raise ValueError(
            f"volume-sizes ({len(volume_sizes)}) and volume-samples ({len(volume_samples)}) "
            f"must have the same length"
        )

    # Validate all volume sizes
    for size in volume_sizes:
        validate_volume_size(size)

    # Create output directories
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Timing data
    timing = {
        'checkpoint': args.checkpoint,
        'stone': args.stone,
        'device': args.device,
        'nsteps': args.nsteps,
        'guidance': args.guidance,
        'periodic': args.periodic,
        'binarize': not args.no_binarize,
        'seed': args.seed,
        'volume_sizes': volume_sizes,
        'volume_samples': volume_samples,
        'samples': []
    }

    # Load porosity field
    porosity_field = load_porosity_field(args.stone)

    # Load models
    print(f"\nLoading models from {args.checkpoint}...")
    if args.periodic:
        print("  Using periodic (circular) convolutions")
    t_start = time.time()
    flowmodule, vaemodule = load_models(args.checkpoint, args.device, periodic=args.periodic)
    timing['model_load_time'] = time.time() - t_start
    print(f"  Model load time: {timing['model_load_time']:.2f}s")

    # Generate volumes for each size
    for pixel_size, n_samples in zip(volume_sizes, volume_samples):
        if n_samples <= 0:
            continue

        latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
        print(f"\nGenerating {n_samples} x {pixel_size}^3 samples (latent: {latent_size}^3, guidance={args.guidance})...")

        if pixel_size > POROSITY_FIELD_SIZE:
            print(f"  Note: size > {POROSITY_FIELD_SIZE}, will use reflect padding")

        for i in range(n_samples):
            print(f"  Generating {pixel_size}_{i}...")
            t_start = time.time()

            # Sample porosity subvolume from real data
            porosity_pixel = sample_porosity_subvolume(porosity_field, pixel_size, rng)

            # Downsample to latent resolution
            porosity_latent = downsample_porosity_to_latent(porosity_pixel)

            # Generate volume
            x = generate_volume(
                flowmodule, vaemodule, porosity_latent,
                pixel_size, args.nsteps, args.device, args.guidance,
                periodic=args.periodic
            )

            elapsed = time.time() - t_start
            timing['samples'].append({'size': pixel_size, 'index': i, 'time': elapsed})

            # Binarize by default (threshold at mean)
            if not args.no_binarize:
                x = (x > x.mean()).astype(bool)

            output_path = os.path.join(data_dir, f'{pixel_size}_{i}.npy')
            np.save(output_path, x)
            print(f"    Saved to {output_path} ({elapsed:.2f}s, {'float' if args.no_binarize else 'bool'})")

            if args.save_porosity:
                # Save the latent-resolution porosity (what the model sees)
                porosity_path = os.path.join(data_dir, f'{pixel_size}_{i}.porosity.npy')
                np.save(porosity_path, porosity_latent)
                print(f"    Saved porosity to {porosity_path}")

    # Compute summary statistics per size
    for size in volume_sizes:
        times = [s['time'] for s in timing['samples'] if s['size'] == size]
        if times:
            timing[f'mean_time_{size}'] = float(np.mean(times))
            timing[f'std_time_{size}'] = float(np.std(times))
            timing[f'total_time_{size}'] = float(np.sum(times))
            timing[f'count_{size}'] = len(times)

    timing['total_generation_time'] = sum(s['time'] for s in timing['samples'])

    # Save timing data
    timing_path = os.path.join(args.output_dir, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)
    print(f"\nTiming data saved to {timing_path}")

    # Print summary
    print("\n=== Timing Summary ===")
    for size in volume_sizes:
        if f'mean_time_{size}' in timing:
            print(f"  {size}^3: {timing[f'count_{size}']} samples, "
                  f"mean={timing[f'mean_time_{size}']:.2f}s, "
                  f"std={timing[f'std_time_{size}']:.2f}s, "
                  f"total={timing[f'total_time_{size}']:.2f}s")
    print(f"  Total generation time: {timing['total_generation_time']:.2f}s")

    print("\nDone!")


if __name__ == '__main__':
    main()

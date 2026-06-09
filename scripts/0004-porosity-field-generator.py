#!/usr/bin/env python
"""
Multi-scale volume generator with porosity field conditioning (updated version).

Generates volumes at arbitrary scales (multiples of 128) using GP-sampled porosity
fields as conditioning.

Usage:
    python porosity_generators_updated.py \
        --checkpoint /path/to/model.ckpt \
        --stone Estaillades \
        --output-dir ./generated_data/ \
        --volume-sizes 256,512,1024,1152 \
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

# Add the aux directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
from model_loaders import load_flow_model, load_autoencoder
from diffsci2.extra import chunk_decode_2, punetg_converters
from diffsci2.extra.matern_gaussian_process import MaternFieldSampler, PeriodicMaternFieldSampler
from diffsci2.nets.enhanced_conditioning import wrap_model_with_enhanced_conditioning


AVAILABLE_STONES = ['Bentheimer', 'Doddington', 'Estaillades', 'Ketton']

# Constants
LATENT_TO_PIXEL_FACTOR = 8  # pixel_size = latent_size * 8
MIN_LATENT_MULTIPLE = 16    # latent_size must be multiple of 16
DILATION_FACTOR = 1         # Working in latent space (no dilation needed)

# Paths to GP analysis data (fitted in latent space with voxel_size=1.0)
GPDATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'data', 'gpdata3c')
GPDATA_PATHS = {
    'Bentheimer': os.path.join(GPDATA_DIR, 'bentheimer', 'bentheimer_porosity_analysis.npz'),
    'Doddington': os.path.join(GPDATA_DIR, 'doddington', 'doddington_porosity_analysis.npz'),
    'Estaillades': os.path.join(GPDATA_DIR, 'estaillades', 'estaillades_porosity_analysis.npz'),
    'Ketton': os.path.join(GPDATA_DIR, 'ketton', 'ketton_porosity_analysis.npz'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate multi-scale volumes with porosity conditioning'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--stone', type=str, required=True, choices=AVAILABLE_STONES,
        help='Stone type for GP parameters'
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
        '--coarse-n', type=int, default=32,
        help='Coarse grid size for GP sampling (default: 16)'
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
        help='Generate periodic (circular) volumes with periodic porosity conditioning'
    )
    parser.add_argument(
        '--no-binarize', action='store_true',
        help='Save raw float values instead of binarized (bool) volumes'
    )
    parser.add_argument(
        '--nfields', type=int, default=None,
        help='Number of porosity fields to sample (enables variance test mode)'
    )
    parser.add_argument(
        '--nsamples-per-field', type=int, default=None,
        help='Number of volume samples per porosity field (enables variance test mode)'
    )
    # Enhanced conditioning options
    parser.add_argument(
        '--enhanced', action='store_true',
        help='Use enhanced conditioning architecture (FiLM + multi-scale)'
    )
    parser.add_argument(
        '--condition-embed-dim', type=int, default=64,
        help='Embedding dimension for enhanced conditioning (must match training)'
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


def load_gpdata(stone: str):
    """Load the GP analysis data for a stone type."""
    path = GPDATA_PATHS[stone]
    return np.load(path)


def create_porosity_sampler(stone: str, coarse_n: int = 16):
    """Create a MaternFieldSampler initialized for the given stone."""
    gpdata = load_gpdata(stone)

    sampler = MaternFieldSampler(
        mean_val=float(gpdata['mean_logit']),
        sigma_sq=float(gpdata['matern_sigma_sq']),
        nu=float(gpdata['matern_nu']),
        length_scale=float(gpdata['matern_length_scale'])
    )

    return sampler, gpdata


def create_periodic_porosity_sampler(stone: str, coarse_n: int = 16):
    """Create a periodic Matern GP sampler for porosity field."""
    gpdata = load_gpdata(stone)

    sampler = PeriodicMaternFieldSampler(
        mean_val=float(gpdata['mean_logit']),
        sigma_sq=float(gpdata['matern_sigma_sq']),
        nu=float(gpdata['matern_nu']),
        length_scale=float(gpdata['matern_length_scale'])
    )

    return sampler, gpdata


def sample_porosity_field(sampler, latent_shape, coarse_n: int = 16):
    """
    Sample a porosity field for conditioning the latent diffusion model.

    The field is sampled in latent space coordinates (DILATION_FACTOR=1).
    GP parameters are fitted directly in latent space, so no dilation is needed.
    """
    import scipy.special

    L_x, L_y, L_z = latent_shape

    # Pixel-space dimensions (dilated coordinates)
    pixel_x = L_x * DILATION_FACTOR
    pixel_y = L_y * DILATION_FACTOR
    pixel_z = L_z * DILATION_FACTOR

    # Coarse grid for GP sampling (in pixel coordinates)
    x_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_x - 0.5) * DILATION_FACTOR, coarse_n)
    y_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_y - 0.5) * DILATION_FACTOR, coarse_n)
    z_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_z - 0.5) * DILATION_FACTOR, coarse_n)

    # Initialize coarse grid
    sampler.initialize_field_from_grid(x_coarse, y_coarse, z_coarse)

    # Target (fine) grid at latent resolution
    x_fine = np.linspace(0.5 * DILATION_FACTOR, (L_x - 0.5) * DILATION_FACTOR, L_x)
    y_fine = np.linspace(0.5 * DILATION_FACTOR, (L_y - 0.5) * DILATION_FACTOR, L_y)
    z_fine = np.linspace(0.5 * DILATION_FACTOR, (L_z - 0.5) * DILATION_FACTOR, L_z)

    # Sample and interpolate to fine grid
    logit_field = sampler.sample_grid_interpolated(1, x_fine, y_fine, z_fine)[0]

    # Sigmoid to convert logit to porosity
    porosity_field = scipy.special.expit(logit_field)

    return porosity_field.astype(np.float32)


def sample_periodic_porosity_field(sampler, shape, coarse_n):
    """Sample a periodic porosity field at the given shape."""
    import scipy.special

    # Create coordinate grid matching the shape
    axes = [np.linspace(0, s, s) for s in shape]
    sampler.initialize_field_from_grid(*axes)

    # Sample the field
    field = sampler.sample_grid(1)[0]

    # Convert from logit to porosity using sigmoid
    porosity = scipy.special.expit(field)

    return porosity.astype(np.float32)


def load_models(checkpoint_path, device, periodic=False, enhanced=False, condition_embed_dim=64):
    """Load flow model and autoencoder.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for computation
        periodic: Whether to use circular convolutions
        enhanced: Whether to use enhanced conditioning wrapper
        condition_embed_dim: Embedding dimension for enhanced conditioning
    """
    # Load base flow model
    flowmodel = load_flow_model(checkpoint_path, custom_checkpoint_path=True)
    vaemodule = load_autoencoder()

    # Wrap with enhanced conditioning if requested
    if enhanced:
        print("  Using enhanced conditioning architecture")
        # Create the wrapper (for inference, cond_drop_p=0 since we don't need dropout)
        flowmodel = wrap_model_with_enhanced_conditioning(
            model=flowmodel,
            condition_embed_dim=condition_embed_dim,
            use_film=True,
            use_multiscale=True,
            use_gradient=True,
            cond_drop_p=0.0,  # No dropout at inference
        )

        # Load enhanced weights from checkpoint
        # The checkpoint contains the full wrapper state
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Lightning saves with 'model.' prefix
            model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            flowmodel.load_state_dict(model_state, strict=False)
            print(f"  Loaded enhanced conditioning weights")

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
        if enhanced:
            # For enhanced model, convert the base_model inside the wrapper
            flowmodule.model.base_model = punetg_converters.convert_conv_to_circular(
                flowmodule.model.base_model, [0, 1, 2], True
            )
        else:
            flowmodule.model = punetg_converters.convert_conv_to_circular(
                flowmodule.model, [0, 1, 2], True
            )

    flowmodule.to(device)
    return flowmodule, vaemodule


def sample_new_porosity_field(sampler, pixel_size, coarse_n, periodic=False):
    """Sample a new porosity field for the given pixel size."""
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    if periodic:
        return sample_periodic_porosity_field(sampler, [latent_size, latent_size, latent_size], coarse_n)
    else:
        return sample_porosity_field(sampler, [latent_size, latent_size, latent_size], coarse_n)


def generate_volume_from_field(flowmodule, vaemodule, porosity_field, pixel_size, nsteps, device, guidance=1.0, periodic=False):
    """
    Generate a single volume conditioned on a given porosity field.

    Parameters
    ----------
    flowmodule : SIModule
        Flow model module.
    vaemodule : VAEModule
        Autoencoder module.
    porosity_field : ndarray
        Porosity field at latent resolution.
    pixel_size : int
        Target volume size in pixels (must be multiple of 128).
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

    porosity_tensor = torch.tensor(porosity_field, dtype=torch.float32)
    y = {'porosity': porosity_tensor}

    use_chunk_decode = (pixel_size > 256)
    periodicity = [True, True, True] if periodic else [False, False, False]

    if use_chunk_decode:
        x_latent = flowmodule.sample(
            1, shape=[4, latent_size, latent_size, latent_size],
            y=y, nsteps=nsteps,
            is_latent_shape=True, return_latents=True, guidance=guidance
        )

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
        x = flowmodule.sample(1, shape=[1, pixel_size, pixel_size, pixel_size], y=y, nsteps=nsteps, guidance=guidance)
        x = x[0][0].cpu().numpy()

    torch.cuda.empty_cache()
    return x


def generate_volume(flowmodule, vaemodule, sampler, pixel_size, coarse_n, nsteps, device, guidance=1.0, periodic=False):
    """Generate a single volume: sample a new porosity field and generate from it."""
    porosity_field = sample_new_porosity_field(sampler, pixel_size, coarse_n, periodic)
    x = generate_volume_from_field(flowmodule, vaemodule, porosity_field, pixel_size, nsteps, device, guidance, periodic)
    return x, porosity_field


def main():
    args = parse_args()

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
        'coarse_n': args.coarse_n,
        'guidance': args.guidance,
        'periodic': args.periodic,
        'binarize': not args.no_binarize,
        'volume_sizes': volume_sizes,
        'volume_samples': volume_samples,
        'samples': []
    }

    # Load models
    print(f"Loading models from {args.checkpoint}...")
    if args.periodic:
        print("  Using periodic (circular) convolutions")
    if args.enhanced:
        print("  Using enhanced conditioning (FiLM + multi-scale)")
    t_start = time.time()
    flowmodule, vaemodule = load_models(
        args.checkpoint,
        args.device,
        periodic=args.periodic,
        enhanced=args.enhanced,
        condition_embed_dim=args.condition_embed_dim
    )
    timing['model_load_time'] = time.time() - t_start
    timing['enhanced'] = args.enhanced
    print(f"  Model load time: {timing['model_load_time']:.2f}s")

    # Create porosity sampler
    print(f"Creating porosity sampler for {args.stone}...")
    if args.periodic:
        sampler, gpdata = create_periodic_porosity_sampler(args.stone, args.coarse_n)
        print("  Using periodic porosity sampler")
    else:
        sampler, gpdata = create_porosity_sampler(args.stone, args.coarse_n)
    print(f"  Mean logit: {gpdata['mean_logit']:.4f}")
    print(f"  Matern params: sigma^2={gpdata['matern_sigma_sq']:.4f}, "
          f"nu={gpdata['matern_nu']:.4f}, l={gpdata['matern_length_scale']:.4f}")

    # Determine mode: variance test or standard
    variance_test = args.nfields is not None and args.nsamples_per_field is not None

    # Generate volumes for each size
    for pixel_size, n_samples in zip(volume_sizes, volume_samples):
        if n_samples <= 0:
            continue

        latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR

        if variance_test:
            nfields = args.nfields
            nsamples_per_field = args.nsamples_per_field
            total = nfields * nsamples_per_field
            print(f"\nVariance test: {nfields} fields x {nsamples_per_field} samples/field = {total} total")
            print(f"  {pixel_size}^3 (latent: {latent_size}^3, guidance={args.guidance})")

            for f in range(nfields):
                # Sample one field for this batch
                porosity_field = sample_new_porosity_field(
                    sampler, pixel_size, args.coarse_n, periodic=args.periodic
                )
                print(f"\n  Field {f}: porosity range [{porosity_field.min():.4f}, {porosity_field.max():.4f}]")

                if args.save_porosity:
                    porosity_path = os.path.join(data_dir, f'{pixel_size}_field_{f}.porosity.npy')
                    np.save(porosity_path, porosity_field)
                    print(f"    Saved porosity to {porosity_path}")

                for s in range(nsamples_per_field):
                    print(f"    Generating {pixel_size}_{s}_{f}...")
                    t_start = time.time()

                    x = generate_volume_from_field(
                        flowmodule, vaemodule, porosity_field,
                        pixel_size, args.nsteps, args.device, args.guidance,
                        periodic=args.periodic
                    )

                    elapsed = time.time() - t_start
                    timing['samples'].append({'size': pixel_size, 'field': f, 'sample': s, 'time': elapsed})

                    if not args.no_binarize:
                        x = (x > x.mean()).astype(bool)

                    output_path = os.path.join(data_dir, f'{pixel_size}_{s}_{f}.npy')
                    np.save(output_path, x)
                    print(f"      Saved to {output_path} ({elapsed:.2f}s, {'float' if args.no_binarize else 'bool'})")

        else:
            # Standard mode: each sample gets its own field
            print(f"\nGenerating {n_samples} x {pixel_size}^3 samples (latent: {latent_size}^3, guidance={args.guidance})...")

            for i in range(n_samples):
                print(f"  Generating {pixel_size}_{i}...")
                t_start = time.time()

                x, porosity_field = generate_volume(
                    flowmodule, vaemodule, sampler,
                    pixel_size, args.coarse_n, args.nsteps, args.device, args.guidance,
                    periodic=args.periodic
                )

                elapsed = time.time() - t_start
                timing['samples'].append({'size': pixel_size, 'index': i, 'time': elapsed})

                if not args.no_binarize:
                    x = (x > x.mean()).astype(bool)

                output_path = os.path.join(data_dir, f'{pixel_size}_{i}.npy')
                np.save(output_path, x)
                print(f"    Saved to {output_path} ({elapsed:.2f}s, {'float' if args.no_binarize else 'bool'})")

                if args.save_porosity:
                    porosity_path = os.path.join(data_dir, f'{pixel_size}_{i}.porosity.npy')
                    np.save(porosity_path, porosity_field)
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

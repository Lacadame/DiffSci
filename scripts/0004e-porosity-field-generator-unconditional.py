#!/usr/bin/env python
"""
Unconditional volume generator.

Generates 3D porous media volumes from an unconditional latent diffusion
checkpoint. No conditioning (porosity field, scalar, etc.) is used.

Usage:
    python scripts/0004e-porosity-field-generator-unconditional.py \
        --checkpoint /path/to/model.ckpt \
        --output-dir ./generated_data/ \
        --volume-sizes 256,512 \
        --volume-samples 64,8 \
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
from model_loaders import load_model_from_module, load_autoencoder
from diffsci2.extra import chunk_decode_2, punetg_converters

LATENT_TO_PIXEL_FACTOR = 8
MIN_LATENT_MULTIPLE = 16


def parse_args():
    parser = argparse.ArgumentParser(description='Unconditional volume generator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to unconditional model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for generated volumes')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for computation')
    parser.add_argument('--nsteps', type=int, default=21,
                        help='Number of sampling steps (default: 21)')
    parser.add_argument('--volume-sizes', type=str, default='256',
                        help='Comma-separated list of volume sizes (multiples of 128)')
    parser.add_argument('--volume-samples', type=str, default='64',
                        help='Comma-separated list of number of samples per size')
    parser.add_argument('--periodic', action='store_true',
                        help='Use periodic (circular) convolutions for generation')
    parser.add_argument('--already-periodic', action='store_true',
                        help='Checkpoint was trained with periodic convolutions. '
                             'Sets up circular convs BEFORE loading weights (implies --periodic).')
    parser.add_argument('--no-binarize', action='store_true',
                        help='Save raw float values instead of binarized (bool) volumes')
    return parser.parse_args()


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(',')]


def validate_volume_size(pixel_size):
    if pixel_size % (LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE) != 0:
        raise ValueError(
            f"Volume size {pixel_size} must be a multiple of "
            f"{LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE} (128). "
            f"Valid examples: 256, 384, 512, 640, 768, 896, 1024, ..."
        )
    return pixel_size // LATENT_TO_PIXEL_FACTOR


def load_unconditional_flow_model(checkpoint_path, already_periodic=False):
    """Load an unconditional flow model (no conditional embedding).

    Args:
        checkpoint_path: Path to checkpoint file.
        already_periodic: If True, convert convolutions to circular BEFORE
            loading weights (for checkpoints trained with --periodic).
    """
    weights = load_model_from_module(checkpoint_path)
    config = diffsci2.nets.PUNetGConfig(
        input_channels=4,
        output_channels=4,
        dimension=3,
        model_channels=64,
        channel_expansion=[2, 4],
        number_resnet_downward_block=2,
        number_resnet_upward_block=2,
        number_resnet_attn_block=0,
        number_resnet_before_attn_block=3,
        number_resnet_after_attn_block=3,
        kernel_size=3,
        in_out_kernel_size=3,
        in_embedding=False,
        time_projection_scale=10.0,
        input_projection_scale=1.0,
        transition_scale_factor=2,
        transition_kernel_size=3,
        dropout=0.1,
        cond_dropout=0.0,
        first_resblock_norm="GroupLN",
        second_resblock_norm="GroupRMS",
        affine_norm=True,
        convolution_type="default",
        num_groups=1,
        attn_residual=False,
        attn_type="default",
        bias=True,
    )
    model = diffsci2.nets.PUNetG(config, conditional_embedding=None)

    if already_periodic:
        model = punetg_converters.convert_conv_to_circular(model, [0, 1, 2], True)

    model.load_state_dict(weights)
    return model


def load_models(checkpoint_path, device, periodic=False, already_periodic=False):
    """Load unconditional flow module and autoencoder.

    Args:
        periodic: Convert to circular convs AFTER loading (post-hoc periodization).
        already_periodic: Convert to circular convs BEFORE loading (checkpoint
            was trained periodic). Implies periodic for generation.
    """
    flowmodel = load_unconditional_flow_model(checkpoint_path, already_periodic=already_periodic)
    vaemodule = load_autoencoder()

    flowmoduleconfig = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        initial_norm=20.0,
        loss_formulation='denoiser',
    )
    flowmodule = diffsci2.models.SIModule(
        config=flowmoduleconfig,
        model=flowmodel,
        autoencoder=vaemodule,
    )

    if periodic and not already_periodic:
        flowmodule.model = punetg_converters.convert_conv_to_circular(
            flowmodule.model, [0, 1, 2], True
        )

    flowmodule.to(device)
    return flowmodule, vaemodule


def generate_volume(flowmodule, vaemodule, pixel_size, nsteps, device, periodic=False):
    """Generate a single unconditional volume."""
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    use_chunk_decode = (pixel_size > 256)
    periodicity = [True, True, True] if periodic else [False, False, False]

    if use_chunk_decode:
        x_latent = flowmodule.sample(
            1, shape=[4, latent_size, latent_size, latent_size],
            y=None, nsteps=nsteps,
            is_latent_shape=True, return_latents=True, guidance=1.0,
        )

        chunk_decode_2.prepare_decoder_for_cached_decode(vaemodule.decoder)
        vaemodule.decoder.to(device)

        x = chunk_decode_2.chunk_decode_3d(
            vaemodule.decoder,
            x_latent,
            [40, 40, 40],
            device=device,
            periodicity=periodicity,
            use_cached_norms=True,
        )
        x = x[0][0].cpu().numpy()
    else:
        x = flowmodule.sample(
            1, shape=[1, pixel_size, pixel_size, pixel_size],
            y=None, nsteps=nsteps, guidance=1.0,
        )
        x = x[0][0].cpu().numpy()

    torch.cuda.empty_cache()
    return x


def main():
    args = parse_args()

    volume_sizes = parse_int_list(args.volume_sizes)
    volume_samples = parse_int_list(args.volume_samples)

    if len(volume_sizes) != len(volume_samples):
        raise ValueError(
            f"volume-sizes ({len(volume_sizes)}) and volume-samples ({len(volume_samples)}) "
            f"must have the same length"
        )

    for size in volume_sizes:
        validate_volume_size(size)

    # --already-periodic implies --periodic
    if args.already_periodic:
        args.periodic = True

    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    timing = {
        'checkpoint': args.checkpoint,
        'device': args.device,
        'nsteps': args.nsteps,
        'periodic': args.periodic,
        'already_periodic': args.already_periodic,
        'binarize': not args.no_binarize,
        'volume_sizes': volume_sizes,
        'volume_samples': volume_samples,
        'samples': [],
    }

    print("=" * 60)
    print("Unconditional Volume Generator")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Steps: {args.nsteps}")
    print(f"  Periodic: {args.periodic}" + (" (already periodic checkpoint)" if args.already_periodic else ""))
    print(f"  Binarize: {not args.no_binarize}")
    print()

    print("Loading models...")
    t_start = time.time()
    flowmodule, vaemodule = load_models(
        args.checkpoint, args.device,
        periodic=args.periodic, already_periodic=args.already_periodic,
    )
    timing['model_load_time'] = time.time() - t_start
    print(f"  Loaded in {timing['model_load_time']:.2f}s")

    for pixel_size, n_samples in zip(volume_sizes, volume_samples):
        if n_samples <= 0:
            continue

        latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
        print(f"\nGenerating {n_samples} x {pixel_size}^3 (latent {latent_size}^3)...")

        for i in range(n_samples):
            print(f"  [{i+1}/{n_samples}] {pixel_size}_{i}...", end=" ", flush=True)
            t_start = time.time()

            x = generate_volume(flowmodule, vaemodule, pixel_size, args.nsteps, args.device, args.periodic)

            elapsed = time.time() - t_start
            timing['samples'].append({'size': pixel_size, 'index': i, 'time': elapsed})

            if not args.no_binarize:
                x = (x > x.mean()).astype(bool)

            output_path = os.path.join(data_dir, f'{pixel_size}_{i}.npy')
            np.save(output_path, x)
            print(f"saved ({elapsed:.2f}s, {'float' if args.no_binarize else 'bool'})")

    # Summary
    for size in volume_sizes:
        times = [s['time'] for s in timing['samples'] if s['size'] == size]
        if times:
            timing[f'mean_time_{size}'] = float(np.mean(times))
            timing[f'std_time_{size}'] = float(np.std(times))
            timing[f'total_time_{size}'] = float(np.sum(times))
            timing[f'count_{size}'] = len(times)

    timing['total_generation_time'] = sum(s['time'] for s in timing['samples'])

    timing_path = os.path.join(args.output_dir, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)

    print("\n=== Summary ===")
    for size in volume_sizes:
        if f'mean_time_{size}' in timing:
            print(f"  {size}^3: {timing[f'count_{size}']} samples, "
                  f"mean={timing[f'mean_time_{size}']:.2f}s, "
                  f"std={timing[f'std_time_{size}']:.2f}s")
    print(f"  Total: {timing['total_generation_time']:.2f}s")
    print(f"  Timing saved to {timing_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()

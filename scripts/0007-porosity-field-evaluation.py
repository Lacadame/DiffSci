#!/usr/bin/env python
"""
Calculate and save local porosity fields for generated stone volumes.

Processes generated stone volumes, calculates local porosity fields using FFT
convolution, and saves them with compression (average pooling) to avoid memory issues.

Usage:
    # Process a single path (skips already-calculated files by default)
    python 0007-porosity-field-evaluation.py --path /path/to/stone/data

    # Force recalculation of all porosity fields
    python 0007-porosity-field-evaluation.py --path /path/to/stone/data --recalculate-porosity
"""

import argparse
import os

import numpy as np
import scipy.signal
import torch


SIZE = 1280
KERNEL_SIZE = 256
POOL_KERNEL = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate and save local porosity fields for generated stone volumes'
    )
    parser.add_argument(
        '--path', type=str, required=True,
        help='Path to the data directory containing generated .npy volumes'
    )
    parser.add_argument(
        '--recalculate-porosity', action='store_true', default=False,
        help='If set, recalculate all porosity fields even if they already exist'
    )
    parser.add_argument(
        '--size', type=int, default=SIZE,
        help=f'Volume size prefix to look for (default: {SIZE})'
    )
    parser.add_argument(
        '--kernel-size', type=int, default=KERNEL_SIZE,
        help=f'Kernel size for FFT convolution (default: {KERNEL_SIZE})'
    )
    parser.add_argument(
        '--pool-kernel', type=int, default=POOL_KERNEL,
        help=f'Average pooling kernel for compression (default: {POOL_KERNEL})'
    )
    parser.add_argument(
        '--suffix-kernel-size', action='store_true', default=False,
        help='If set, include kernel size in the output filename (e.g. .calculated_porosity_256.npy)'
    )
    return parser.parse_args()


def load_generated_data(data_dir, size, return_names=False, prepattern=None):
    """Load all generated volumes of a given size or return their names."""
    print(data_dir)
    if prepattern is None:
        def filter_pattern(f):
            return f.endswith('.npy') and f.count('.') == 1
    else:
        def filter_pattern(f):
            return f.endswith(f'.{prepattern}.npy')

    files = os.listdir(data_dir)
    print(files)
    chosen_volumes = [f for f in files if f.startswith(f'{size}_')
                      and filter_pattern(f)]
    return chosen_volumes


def calculate_porosity_field_full(data, kernel_size):
    """
    Calculate the local porosity field using FFT convolution (same mode, full size).

    Uses proper normalization that accounts for boundary effects: at each position,
    divides by the number of actual (non-padded) voxels that contributed.
    """
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    pore_data = 1 - data.astype(np.float32)

    pore_sum = scipy.signal.fftconvolve(pore_data, kernel, mode='same')
    normalization = scipy.signal.fftconvolve(
        np.ones_like(pore_data), kernel, mode='same'
    )
    porosity_field = pore_sum / normalization
    return porosity_field


def average_volume(volume, kernel_size):
    """Downsample volume using 3D average pooling."""
    avgpool = torch.nn.AvgPool3d(kernel_size)
    volume = torch.from_numpy(volume).float()
    volume = volume.unsqueeze(0).unsqueeze(0)
    avg_volume = avgpool(volume)
    avg_volume = avg_volume.squeeze().numpy()
    return avg_volume


def main():
    args = parse_args()

    data_dir = args.path
    size = args.size
    kernel_size = args.kernel_size
    pool_kernel = args.pool_kernel
    recalculate = args.recalculate_porosity
    suffix_kernel_size = args.suffix_kernel_size

    porosity_tag = f'calculated_porosity_{kernel_size}' if suffix_kernel_size else 'calculated_porosity'

    print(f"Path: {data_dir}")
    print(f"Size: {size}")
    print(f"Kernel size: {kernel_size}")
    print(f"Pool kernel: {pool_kernel}")
    print(f"Recalculate: {recalculate}")
    print(f"Suffix kernel size: {suffix_kernel_size}")
    print(f"Porosity tag: {porosity_tag}")

    # Get list of all volume files
    names = load_generated_data(data_dir, size)
    print(f"\nFound {len(names)} volume files")

    if not recalculate:
        calculated_porosity_names = load_generated_data(
            data_dir, size, prepattern=porosity_tag
        )
        print(f"Found {len(calculated_porosity_names)} already calculated porosity fields")
    else:
        calculated_porosity_names = []

    # Process each volume
    for i, name in enumerate(sorted(names)):
        expected_output = name.split('.')[0] + f'.{porosity_tag}.npy'

        if not recalculate and expected_output in calculated_porosity_names:
            print(f"[{i+1}/{len(names)}] {name} already calculated, skipping")
            continue

        print(f"[{i+1}/{len(names)}] Calculating {name}")

        volume = np.load(os.path.join(data_dir, name))
        print(f"  Volume shape: {volume.shape}")

        field = calculate_porosity_field_full(volume, kernel_size)
        print(f"  Porosity field shape: {field.shape}")

        field = average_volume(field, pool_kernel)
        print(f"  Compressed field shape: {field.shape}")

        savename = name.replace('.npy', f'.{porosity_tag}.npy')
        save_path = os.path.join(data_dir, savename)
        np.save(save_path, field)
        print(f"  Saved to {savename}")

    print("\nDone!")


if __name__ == '__main__':
    main()

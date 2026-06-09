#!/usr/bin/env python
"""
Metrics evaluator for generated porous media volumes.

Computes porosity and permeability metrics for generated volumes and compares
against reference (validation) data. Saves results to npz for later analysis.

Usage:
    # Simple usage with stone name (auto-resolves paths)
    python metrics_evaluator.py --stone Estaillades --volume-sizes 1024,1152

    # With border cropping (crops 128 voxels from each side)
    python metrics_evaluator.py --stone Estaillades --volume-sizes 1024,1152 --border-crop 128

    # Custom paths
    python metrics_evaluator.py \
        --generated-dir ./generated_data/data/ \
        --reference-path /path/to/reference.raw \
        --output metrics_results.npz \
        --voxel-length 3.3116e-6
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import poregen.features


# Voxel lengths for different stones (in meters)
VOXEL_LENGTHS = {
    'Bentheimer': 3.0035e-6,
    'Doddington': 2.6929e-6,
    'Estaillades': 3.31136e-6,
    'Ketton': 3.00006e-6,
}

# Reference volume paths
DATA_DIR = '/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/'
REFERENCE_PATHS = {
    'Bentheimer': DATA_DIR + 'Bentheimer_1000c_3p0035um.raw',
    'Doddington': DATA_DIR + 'Doddington_1000c_2p6929um.raw',
    'Estaillades': DATA_DIR + 'Estaillades_1000c_3p31136um.raw',
    'Ketton': DATA_DIR + 'Ketton_1000c_3p00006um.raw',
}

# Generated data directories
GENERATED_DATA_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/'
GENERATED_PATHS = {
    'Bentheimer': GENERATED_DATA_DIR + 'bentheimer_pfield_gen/data',
    'Doddington': GENERATED_DATA_DIR + 'doddington_pfield_gen/data',
    'Estaillades': GENERATED_DATA_DIR + 'estaillades_pfield_gen/data',
    'Ketton': GENERATED_DATA_DIR + 'ketton_pfield_gen/data',
}

AVAILABLE_STONES = list(VOXEL_LENGTHS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate metrics for generated porous media volumes'
    )
    parser.add_argument(
        '--stone', type=str, default=None, choices=AVAILABLE_STONES,
        help='Stone type (sets voxel length, reference path, and generated dir automatically)'
    )
    parser.add_argument(
        '--generated-dir', type=str, default=None,
        help='Directory containing generated volumes (overrides --stone default)'
    )
    parser.add_argument(
        '--reference-path', type=str, default=None,
        help='Path to reference .raw volume file (overrides --stone default)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output .npz file path for results (default: metrics_{stone}.npz)'
    )
    parser.add_argument(
        '--voxel-length', type=float, default=None,
        help='Voxel length in meters (overrides --stone)'
    )
    parser.add_argument(
        '--reference-shape', type=int, nargs=3, default=[1000, 1000, 1000],
        help='Shape of reference volume (default: 1000 1000 1000)'
    )
    parser.add_argument(
        '--subvolume-size', type=int, default=256,
        help='Subvolume size for metric extraction (default: 256)'
    )
    parser.add_argument(
        '--skip-permeability', action='store_true',
        help='Skip permeability calculation (much faster)'
    )
    parser.add_argument(
        '--border-crop', type=int, default=0,
        help='Crop this many voxels from each border before extracting subvolumes (default: 0)'
    )
    parser.add_argument(
        '--volume-sizes', type=str, default='256,512,1024',
        help='Comma-separated list of volume sizes to process (default: 256,512,1024)'
    )
    parser.add_argument(
        '--full-volume', action='store_true',
        help='Compute metrics on full (cropped) volume instead of extracting subvolumes'
    )
    parser.add_argument(
        '--subvolumes-per-side', type=int, default=None,
        help='Force this many subvolumes per side (e.g., 2 means 2x2x2=8 subvolumes with striding)'
    )
    return parser.parse_args()


def extract_strided_subvolumes(volume, subvolume_size, n_per_side=None):
    """
    Extract subvolumes from a cubic volume with strided extraction.

    Parameters
    ----------
    volume : ndarray
        3D volume of shape [L, L, L]
    subvolume_size : int
        Size of each subvolume (l)
    n_per_side : int, optional
        If specified, force this many subvolumes per side with uniform striding.
        If None, calculates automatically based on size.

    Returns
    -------
    subvolumes : list
        List of subvolumes, each of shape [l, l, l]
    n : int
        Number of subvolumes per side
    """
    L = volume.shape[0]
    l = subvolume_size

    assert len(volume.shape) == 3, f"Volume must be 3D, got shape {volume.shape}"
    assert L >= l, f"Volume size {L} must be >= subvolume_size {l}"

    # Calculate number of subvolumes per dimension
    if n_per_side is not None:
        n = n_per_side
        if n == 1:
            stride = 0
        else:
            stride = (L - l) / (n - 1)
    elif L % l == 0:
        n = L // l
        stride = l
    else:
        n = L // l + 1
        stride = (L - l) / (n - 1) if n > 1 else 0

    subvolumes = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                start_i = int(i * stride)
                start_j = int(j * stride)
                start_k = int(k * stride)

                subvolume = volume[
                    start_i:start_i + l,
                    start_j:start_j + l,
                    start_k:start_k + l
                ]
                subvolumes.append(subvolume)

    return subvolumes, n


def get_generated_volume_paths(data_dir, size):
    """Get paths to all generated volumes of a given size (lazy loading)."""
    files = os.listdir(data_dir)
    paths = []
    for f in sorted(files):
        if f.startswith(f'{size}_') and f.endswith('.npy') and f.count('.') == 1:
            paths.append(os.path.join(data_dir, f))
    return paths


def center_crop(volume, border):
    """Crop border voxels from all sides of the volume."""
    if border <= 0:
        return volume
    return volume[border:-border, border:-border, border:-border]


def extract_metrics(subvolumes, extractor):
    """Extract metrics from a list of subvolumes."""
    porosities = []
    permeabilities = []

    for subvolume in subvolumes:
        # Binarize if needed (threshold at mean)
        if subvolume.dtype != np.uint8 and subvolume.dtype != bool:
            subvolume = subvolume > subvolume.mean()

        tensor = torch.tensor(subvolume).unsqueeze(0)
        result = extractor(tensor)

        # Extract porosity
        porosity = result['porosity']
        if hasattr(porosity, 'item'):
            porosity = porosity.item()
        porosities.append(float(porosity))

        # Extract permeability if available
        if 'permeability' in result:
            permeability = result['permeability']
            if torch.any(torch.isnan(permeability)):
                permeabilities.append(np.nan)
            else:
                permeabilities.append(permeability[0].item())

    return {
        'porosity': np.array(porosities),
        'permeability': np.array(permeabilities) if permeabilities else None
    }


def main():
    args = parse_args()

    # Resolve paths from stone if not explicitly given
    if args.stone is not None:
        stone = args.stone
        generated_dir = args.generated_dir or GENERATED_PATHS[stone]
        reference_path = args.reference_path or REFERENCE_PATHS[stone]
        voxel_length = args.voxel_length or VOXEL_LENGTHS[stone]
        output_path = args.output or f'metrics_{stone.lower()}.npz'
    else:
        # Require explicit paths if no stone specified
        if args.generated_dir is None or args.reference_path is None:
            raise ValueError("Must specify --stone or both --generated-dir and --reference-path")
        if args.voxel_length is None:
            raise ValueError("Must specify --stone or --voxel-length")
        if args.output is None:
            raise ValueError("Must specify --stone or --output")
        stone = None
        generated_dir = args.generated_dir
        reference_path = args.reference_path
        voxel_length = args.voxel_length
        output_path = args.output

    print(f"Stone: {stone or 'custom'}")
    print(f"Voxel length: {voxel_length:.6e} m")
    print(f"Generated dir: {generated_dir}")
    print(f"Reference path: {reference_path}")
    print(f"Output: {output_path}")
    print(f"Subvolume size: {args.subvolume_size}" + (" (full volume mode)" if args.full_volume else ""))
    if args.border_crop > 0:
        print(f"Border crop: {args.border_crop}")

    # Create extractor
    porosity_extractor = poregen.features.feature_extractors.PorosityExtractor()
    if args.skip_permeability:
        extractor = porosity_extractor
    else:
        permeability_extractor = poregen.features.feature_extractors.PermeabilityExtractor(
            voxel_length=voxel_length, type_pnm=2
        )
        extractor = poregen.features.feature_extractors.CompositeExtractor(
            [porosity_extractor, permeability_extractor]
        )

    # Timing
    timing = {}

    # Load and process reference data
    print(f"\nLoading reference data from {reference_path}...")
    t_start = time.time()
    reference_data = np.fromfile(reference_path, dtype=np.uint8).reshape(args.reference_shape)
    print(f"  Reference shape: {reference_data.shape}")

    if args.full_volume:
        print(f"  Using full volume for reference metrics")
        reference_subvolumes = [reference_data]
        reference_n_per_side = 1
    else:
        print(f"  Extracting {args.subvolume_size}^3 subvolumes...")
        reference_subvolumes, reference_n_per_side = extract_strided_subvolumes(
            reference_data, args.subvolume_size, args.subvolumes_per_side
        )
        print(f"  Extracted {len(reference_subvolumes)} subvolumes ({reference_n_per_side}^3)")

    print(f"  Computing metrics...")
    reference_metrics = extract_metrics(reference_subvolumes, extractor)
    timing['reference'] = time.time() - t_start
    print(f"  Reference metrics computed in {timing['reference']:.2f}s")
    print(f"    Porosity: mean={reference_metrics['porosity'].mean():.4f}, "
          f"std={reference_metrics['porosity'].std():.4f}")
    if reference_metrics['permeability'] is not None:
        valid_perm = reference_metrics['permeability'][~np.isnan(reference_metrics['permeability'])]
        if len(valid_perm) > 0:
            print(f"    Permeability: mean={valid_perm.mean():.4e}, std={valid_perm.std():.4e}")

    # Results storage
    results = {
        'stone': stone or 'custom',
        'reference_porosity': reference_metrics['porosity'],
        'reference_permeability': reference_metrics['permeability'],
        'subvolume_size': args.subvolume_size,
        'subvolumes_per_side': args.subvolumes_per_side,
        'full_volume': args.full_volume,
        'voxel_length': voxel_length,
        'border_crop': args.border_crop,
    }

    # Parse volume sizes
    volume_sizes = [int(x.strip()) for x in args.volume_sizes.split(',')]

    # Process generated data for each size
    for size in volume_sizes:
        print(f"\nProcessing generated {size}^3 volumes...")
        t_start = time.time()

        volume_paths = get_generated_volume_paths(generated_dir, size)
        if len(volume_paths) == 0:
            print(f"  No {size}^3 volumes found, skipping")
            continue

        print(f"  Found {len(volume_paths)} volumes")

        # Compute effective size after crop
        effective_size = size - 2 * args.border_crop if args.border_crop > 0 else size
        if args.border_crop > 0:
            print(f"  Will apply border crop of {args.border_crop} -> {effective_size}^3")

        # Process volumes one at a time to save memory
        all_subvolumes = []
        n_per_side = None
        for vol_idx, vol_path in enumerate(volume_paths):
            print(f"  Loading volume {vol_idx + 1}/{len(volume_paths)}: {os.path.basename(vol_path)}")
            volume = np.load(vol_path)

            # Apply border crop if specified
            if args.border_crop > 0:
                volume = center_crop(volume, args.border_crop)

            # Extract subvolumes if needed
            if args.full_volume:
                subvolumes = [volume]
                n_per_side = 1
            elif effective_size == args.subvolume_size and args.subvolumes_per_side is None:
                subvolumes = [volume]
                n_per_side = 1
            else:
                subvolumes, n_per_side = extract_strided_subvolumes(volume, args.subvolume_size, args.subvolumes_per_side)

            all_subvolumes.extend(subvolumes)

            # Free memory
            del volume

        if args.full_volume:
            print(f"  Using {len(all_subvolumes)} full volumes")
        else:
            print(f"  Total subvolumes: {len(all_subvolumes)} ({n_per_side}^3 per volume)")

        # Compute metrics
        print(f"  Computing metrics...")
        metrics = extract_metrics(all_subvolumes, extractor)
        timing[f'generated_{size}'] = time.time() - t_start

        print(f"  Metrics computed in {timing[f'generated_{size}']:.2f}s")
        print(f"    Porosity: mean={metrics['porosity'].mean():.4f}, "
              f"std={metrics['porosity'].std():.4f}")
        if metrics['permeability'] is not None:
            valid_perm = metrics['permeability'][~np.isnan(metrics['permeability'])]
            if len(valid_perm) > 0:
                print(f"    Permeability: mean={valid_perm.mean():.4e}, std={valid_perm.std():.4e}")

        # Store results
        results[f'generated_{size}_porosity'] = metrics['porosity']
        results[f'generated_{size}_permeability'] = metrics['permeability']
        results[f'generated_{size}_n_volumes'] = len(volume_paths)
        results[f'generated_{size}_n_subvolumes'] = len(all_subvolumes)

    # Add timing and metadata
    results['timing'] = timing
    results['total_time'] = sum(timing.values())

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"\nSaving results to {output_path}...")

    # Convert timing dict to arrays for npz compatibility
    timing_keys = list(timing.keys())
    timing_values = [timing[k] for k in timing_keys]

    # Prepare save dict (npz doesn't like nested dicts)
    save_dict = {}
    for k, v in results.items():
        if k == 'timing':
            save_dict['timing_keys'] = timing_keys
            save_dict['timing_values'] = timing_values
        elif v is not None:
            save_dict[k] = v

    np.savez(output_path, **save_dict)

    print(f"\n=== Summary ===")
    print(f"Total processing time: {results['total_time']:.2f}s")
    print(f"Results saved to: {output_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()

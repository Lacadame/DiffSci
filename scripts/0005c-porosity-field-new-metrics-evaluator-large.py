#!/usr/bin/env python
"""
Large-volume metrics evaluator for generated porous media.

Takes a SINGLE large volume (2304^3), crops to 2048^3, splits into 8 subcubes
of 1024^3, and computes porosity and two-phase flow properties for each subcube.
The subcube statistics are compared to the reference stone, just like 0005b does
with many generated 1024 volumes.

Memory-conscious: loads the full volume, extracts one subcube at a time, and
frees memory between subcubes.

Usage:
    # Evaluate a single large volume for Bentheimer
    python 0005c-porosity-field-new-metrics-evaluator-large.py \
        --stone Bentheimer \
        --volume-path /home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/bentheimer_pfield_case_5_2304/data/2304_0.npy

    # Skip reference processing
    python 0005c-porosity-field-new-metrics-evaluator-large.py \
        --stone Bentheimer \
        --volume-path /path/to/2304_0.npy \
        --skip-reference
"""

import argparse
import os
import time

import numpy as np

from poregen.features.snow2 import snow2
from diffsci2.extra.pore.permeability_from_pnm import PoreNetworkPermeability


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

AVAILABLE_STONES = list(VOXEL_LENGTHS.keys())

# Large volume constants
FULL_SIZE = 2304
CROP_SIZE = 2048
SUBCUBE_SIZE = 1024
BORDER_CROP = (FULL_SIZE - CROP_SIZE) // 2  # 128


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate large-volume metrics: split one 2304 volume into 8 subcubes of 1024'
    )
    parser.add_argument(
        '--stone', type=str, required=True, choices=AVAILABLE_STONES,
        help='Stone type (sets voxel length and reference path)'
    )
    parser.add_argument(
        '--volume-path', type=str, required=True,
        help='Path to the single large .npy volume (2304^3)'
    )
    parser.add_argument(
        '--reference-path', type=str, default=None,
        help='Path to reference .raw volume file (overrides --stone default)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output .npz file path (default: saved in same folder as volume)'
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
        '--skip-reference', action='store_true',
        help='Skip processing reference volume'
    )
    parser.add_argument(
        '--contact-angle', type=float, default=140.0,
        help='Contact angle for drainage simulation [degrees] (default: 140)'
    )
    parser.add_argument(
        '--surface-tension', type=float, default=0.48,
        help='Surface tension for drainage simulation [N/m] (default: 0.48)'
    )
    return parser.parse_args()


def center_crop(volume, border):
    """Crop border voxels from all sides of the volume."""
    if border <= 0:
        return volume
    return volume[border:-border, border:-border, border:-border]


def get_subcube_slices():
    """
    Return the 8 subcube slicing tuples for a 2048^3 volume split into 1024^3.

    Yields (i, j, k, slice_x, slice_y, slice_z) for each subcube.
    """
    n = SUBCUBE_SIZE
    subcubes = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                sx = slice(i * n, (i + 1) * n)
                sy = slice(j * n, (j + 1) * n)
                sz = slice(k * n, (k + 1) * n)
                subcubes.append((i, j, k, sx, sy, sz))
    return subcubes


def extract_porespy_network(binary_volume):
    """
    Extract pore network from binary volume using SNOW algorithm.

    Returns the raw porespy network dictionary (before openpnm conversion).
    """
    # SNOW expects pore space = 1, solid = 0
    # Our convention is pore space = 0, solid = 1
    pore_space = 1 - binary_volume
    partitioning = snow2(pore_space, voxel_size=1.0)
    return partitioning.network


def compute_two_phase_metrics(binary_volume, porespy_network, voxel_size, contact_angle, surface_tension):
    """
    Compute porosity and two-phase flow properties from a binary volume
    and its already-extracted porespy network.

    Args:
        binary_volume: 3D binary array (0=pore, 1=solid)
        porespy_network: Pre-extracted porespy network dict (from SNOW2)
        voxel_size: Physical voxel size in meters
        contact_angle: Contact angle for drainage [degrees]
        surface_tension: Surface tension for drainage [N/m]

    Returns:
        dict with porosity, absolute permeability, relative permeabilities, Sw, Pc
    """
    # Porosity: fraction of void space (pore space = 0, solid = 1)
    porosity = (1 - binary_volume).mean()

    # Create pore network wrapper from the already-extracted network
    pn_wrapper = PoreNetworkPermeability.from_porespy_network(
        porespy_network,
        volume_length=binary_volume.shape[0],
        voxel_size=voxel_size,
    )

    # Calculate absolute permeability
    abs_perm = pn_wrapper.calculate_absolute_permeability()

    # Run drainage simulation
    _ = pn_wrapper.run_drainage_simulation(
        contact_angle=contact_angle,
        surface_tension=surface_tension,
    )

    # Calculate relative permeability curves
    rel_perm = pn_wrapper.calculate_relative_permeability_curves()

    return {
        'porosity': porosity,
        # Absolute permeability
        'K_abs_x': abs_perm.K_x,
        'K_abs_y': abs_perm.K_y,
        'K_abs_z': abs_perm.K_z,
        'K_abs_mean': abs_perm.K_mean,
        'K_abs_x_physical': abs_perm.K_x_physical,
        'K_abs_y_physical': abs_perm.K_y_physical,
        'K_abs_z_physical': abs_perm.K_z_physical,
        'K_abs_mean_physical': abs_perm.K_mean_physical,
        # Saturation and capillary pressure
        'Sw': rel_perm.Sw,
        'Snw': rel_perm.Snwp,
        'Pc': rel_perm.Pc,
        # Relative permeabilities (shape: n_saturations x 3 for x,y,z)
        'kr_wetting': rel_perm.kr_wetting,
        'kr_nonwetting': rel_perm.kr_nonwetting,
        'kr_wetting_mean': rel_perm.kr_wetting_mean,
        'kr_nonwetting_mean': rel_perm.kr_nonwetting_mean,
    }


def save_network(network_dict, save_path):
    """Save porespy network dictionary to npz file."""
    save_dict = {}
    for key, value in network_dict.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, (int, float, str, bool)):
            save_dict[key] = np.array(value)
        elif isinstance(value, list):
            save_dict[key] = np.array(value)
        else:
            print(f"  Warning: skipping network key '{key}' of type {type(value)}")

    np.savez(save_path, **save_dict)


def main():
    args = parse_args()

    stone = args.stone
    voxel_length = args.voxel_length or VOXEL_LENGTHS[stone]
    reference_path = args.reference_path or REFERENCE_PATHS[stone]

    # Output saved in the same folder as the input volume
    volume_dir = os.path.dirname(os.path.abspath(args.volume_path))
    volume_basename = os.path.splitext(os.path.basename(args.volume_path))[0]
    output_path = args.output or os.path.join(
        volume_dir, f'metrics_{stone.lower()}_{volume_basename}_large_twophase.npz'
    )

    print(f"Stone: {stone}")
    print(f"Voxel length: {voxel_length:.6e} m")
    print(f"Volume path: {args.volume_path}")
    print(f"Reference path: {reference_path}")
    print(f"Output: {output_path}")
    print(f"Scheme: {FULL_SIZE}^3 -> crop {BORDER_CROP} -> {CROP_SIZE}^3 -> 8 x {SUBCUBE_SIZE}^3")
    print(f"Contact angle: {args.contact_angle} deg")
    print(f"Surface tension: {args.surface_tension} N/m")

    # Timing
    timing = {}

    # Results storage
    results = {
        'stone': stone,
        'voxel_length': voxel_length,
        'full_size': FULL_SIZE,
        'crop_size': CROP_SIZE,
        'subcube_size': SUBCUBE_SIZE,
        'border_crop': BORDER_CROP,
        'contact_angle': args.contact_angle,
        'surface_tension': args.surface_tension,
        'volume_path': args.volume_path,
    }

    # -------------------------------------------------------------------------
    # Process reference data
    # -------------------------------------------------------------------------
    if not args.skip_reference and reference_path is not None and os.path.exists(reference_path):
        print(f"\nProcessing reference data from {reference_path}...")
        t_start = time.time()

        reference_data = np.fromfile(reference_path, dtype=np.uint8).reshape(args.reference_shape)
        print(f"  Reference shape: {reference_data.shape}")

        # Extract and save network
        print(f"  Extracting pore network...")
        network = extract_porespy_network(reference_data)
        network_path = reference_path.replace('.raw', '.network.npz')
        save_network(network, network_path)
        print(f"  Saved network to {network_path}")

        # Compute metrics
        print(f"  Computing two-phase flow metrics...")
        ref_metrics = compute_two_phase_metrics(
            reference_data, network, voxel_length,
            args.contact_angle, args.surface_tension
        )

        timing['reference'] = time.time() - t_start
        print(f"  Reference metrics computed in {timing['reference']:.2f}s")
        print(f"    Porosity: {ref_metrics['porosity']:.4f}")
        print(f"    K_abs (mean): {ref_metrics['K_abs_mean_physical'] * 1e15:.2f} mD")

        # Store reference results
        for key, value in ref_metrics.items():
            results[f'reference_{key}'] = value

        del reference_data, network

    # -------------------------------------------------------------------------
    # Process the large volume: load, crop, split into 8 subcubes
    # -------------------------------------------------------------------------
    print(f"\nProcessing large volume: {args.volume_path}")
    print(f"  Loading full {FULL_SIZE}^3 volume...")

    subcube_slices = get_subcube_slices()
    all_metrics = []

    # Load the full volume once, extract all subcubes, then free it.
    # We load -> crop -> extract subcubes to separate arrays -> free full volume.
    # This way we only hold the cropped volume temporarily.
    t_start = time.time()

    full_volume = np.load(args.volume_path)
    print(f"  Loaded shape: {full_volume.shape}")

    if full_volume.shape != (FULL_SIZE, FULL_SIZE, FULL_SIZE):
        raise ValueError(
            f"Expected volume of shape ({FULL_SIZE}, {FULL_SIZE}, {FULL_SIZE}), "
            f"got {full_volume.shape}"
        )

    # Crop to 2048^3
    print(f"  Cropping {BORDER_CROP} voxels from each side -> {CROP_SIZE}^3...")
    cropped = center_crop(full_volume, BORDER_CROP)
    del full_volume
    print(f"  Cropped shape: {cropped.shape}")

    # Extract each subcube, process it, free it
    for sub_idx, (i, j, k, sx, sy, sz) in enumerate(subcube_slices):
        label = f"subcube_{i}{j}{k}"
        print(f"\n  [{sub_idx + 1}/8] {label} (indices [{i}:{i+1}, {j}:{j+1}, {k}:{k+1}] x {SUBCUBE_SIZE})")

        # Extract subcube (this creates a copy since slicing + .copy())
        subcube = cropped[sx, sy, sz].copy()
        print(f"    Shape: {subcube.shape}")

        # Extract and save network
        print(f"    Extracting pore network...")
        network = extract_porespy_network(subcube)
        network_save_path = os.path.join(
            volume_dir,
            f'{volume_basename}_{label}.network.npz'
        )
        save_network(network, network_save_path)
        print(f"    Saved network to {os.path.basename(network_save_path)}")

        # Compute two-phase flow metrics
        try:
            metrics = compute_two_phase_metrics(
                subcube, network, voxel_length,
                args.contact_angle, args.surface_tension
            )
            all_metrics.append(metrics)
            print(f"    Porosity: {metrics['porosity']:.4f}, "
                  f"K_abs: {metrics['K_abs_mean_physical'] * 1e15:.2f} mD")
        except Exception as e:
            print(f"    Error computing metrics: {e}")

        # Free memory
        del subcube, network

    # Free the cropped volume
    del cropped

    timing['subcubes'] = time.time() - t_start
    print(f"\n  All subcubes processed in {timing['subcubes']:.2f}s")

    if len(all_metrics) == 0:
        print("  No valid metrics computed for any subcube!")
    else:
        # Aggregate results (same format as 0005b for generated volumes)
        size_key = f'generated_{SUBCUBE_SIZE}'
        results[f'{size_key}_n_volumes'] = len(all_metrics)
        results[f'{size_key}_porosity'] = np.array([m['porosity'] for m in all_metrics])
        results[f'{size_key}_K_abs_mean'] = np.array([m['K_abs_mean'] for m in all_metrics])
        results[f'{size_key}_K_abs_mean_physical'] = np.array([m['K_abs_mean_physical'] for m in all_metrics])
        results[f'{size_key}_K_abs_x'] = np.array([m['K_abs_x'] for m in all_metrics])
        results[f'{size_key}_K_abs_y'] = np.array([m['K_abs_y'] for m in all_metrics])
        results[f'{size_key}_K_abs_z'] = np.array([m['K_abs_z'] for m in all_metrics])

        results[f'{size_key}_Sw'] = np.array([m['Sw'] for m in all_metrics], dtype=object)
        results[f'{size_key}_Snw'] = np.array([m['Snw'] for m in all_metrics], dtype=object)
        results[f'{size_key}_Pc'] = np.array([m['Pc'] for m in all_metrics], dtype=object)
        results[f'{size_key}_kr_wetting'] = np.array([m['kr_wetting'] for m in all_metrics], dtype=object)
        results[f'{size_key}_kr_nonwetting'] = np.array([m['kr_nonwetting'] for m in all_metrics], dtype=object)
        results[f'{size_key}_kr_wetting_mean'] = np.array([m['kr_wetting_mean'] for m in all_metrics], dtype=object)
        results[f'{size_key}_kr_nonwetting_mean'] = np.array([m['kr_nonwetting_mean'] for m in all_metrics], dtype=object)

        # Print summary statistics
        porosities = results[f'{size_key}_porosity']
        perms = results[f'{size_key}_K_abs_mean_physical'] * 1e15
        print(f"\n  Summary (8 subcubes of {SUBCUBE_SIZE}^3):")
        print(f"    Porosity: mean={porosities.mean():.4f}, std={porosities.std():.4f}")
        print(f"    K_abs (mD): mean={perms.mean():.2f}, std={perms.std():.2f}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    results['timing'] = timing
    results['total_time'] = sum(timing.values())

    print(f"\nSaving results to {output_path}...")

    save_dict = {}
    for k, v in results.items():
        if k == 'timing':
            save_dict['timing_keys'] = list(timing.keys())
            save_dict['timing_values'] = list(timing.values())
        elif v is not None:
            save_dict[k] = v

    np.savez(output_path, **save_dict)

    print(f"\n=== Summary ===")
    print(f"Total processing time: {results['total_time']:.2f}s")
    print(f"Results saved to: {output_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()

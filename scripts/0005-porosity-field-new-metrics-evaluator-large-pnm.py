#!/usr/bin/env python
"""
Large-volume PNM extractor for generated porous media.

Takes a directory of large volumes (.npy), crops 128 voxels from each side
(border crop), and extracts the pore network (SNOW2) for the cropped volume.

Only the pore network is extracted — no flow simulations are run. The saved
networks can later be used to compute subnetwork flows.

Volumes are discovered automatically: all .npy files without compound
extensions (e.g. 0.npy is included, 0.porosity.npy is excluded).

Usage:
    python 0005-porosity-field-new-metrics-evaluator-large-pnm.py \
        --stone Bentheimer \
        --volume-dir /home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/bentheimer_pfield_gen_case_10_large/data/
"""

import argparse
import os
import time

import numpy as np

from poregen.features.snow2 import snow2


# Voxel lengths for different stones (in meters)
VOXEL_LENGTHS = {
    'Bentheimer': 3.0035e-6,
    'Doddington': 2.6929e-6,
    'Estaillades': 3.31136e-6,
    'Ketton': 3.00006e-6,
}

AVAILABLE_STONES = list(VOXEL_LENGTHS.keys())

# Border crop: always remove 128 voxels from each side
BORDER_CROP = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract pore networks (SNOW2) from large cropped volumes'
    )
    parser.add_argument(
        '--stone', type=str, required=True, choices=AVAILABLE_STONES,
        help='Stone type (used for output naming)'
    )
    parser.add_argument(
        '--volume-dir', type=str, required=True,
        help='Directory containing .npy volume files'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for network files (default: same as volume-dir)'
    )
    return parser.parse_args()


def find_volumes(directory):
    """
    Find all .npy files without compound extensions in a directory.

    Includes:  0.npy, 2304_0.npy
    Excludes:  0.porosity.npy, 2304_0.porosity.npy  (compound .X.npy)
    """
    volumes = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.npy'):
            continue
        # Check for compound extension: strip .npy, then check if remainder has a dot
        stem = fname[:-4]  # remove .npy
        if '.' in stem:
            continue
        volumes.append(os.path.join(directory, fname))
    return volumes


def center_crop(volume, border):
    """Crop border voxels from all sides of the volume."""
    if border <= 0:
        return volume
    return volume[border:-border, border:-border, border:-border]


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
    volume_dir = os.path.abspath(args.volume_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else volume_dir

    # Discover volumes
    volumes = find_volumes(volume_dir)
    if not volumes:
        print(f"No .npy volumes found in {volume_dir}")
        return

    print(f"Stone: {stone}")
    print(f"Volume dir: {volume_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(volumes)} volume(s): {[os.path.basename(v) for v in volumes]}")
    print(f"Scheme: load -> crop {BORDER_CROP} voxels from each side -> SNOW2 network")

    os.makedirs(output_dir, exist_ok=True)
    t_total_start = time.time()

    for vol_idx, vol_path in enumerate(volumes):
        vol_basename = os.path.splitext(os.path.basename(vol_path))[0]
        print(f"\n{'='*60}")
        print(f"[{vol_idx + 1}/{len(volumes)}] Processing {os.path.basename(vol_path)}")
        print(f"{'='*60}")

        t_vol_start = time.time()

        # Load volume
        full_volume = np.load(vol_path)
        print(f"  Loaded shape: {full_volume.shape}")

        # Crop 128 voxels from each side
        print(f"  Cropping {BORDER_CROP} voxels from each side...")
        volume = center_crop(full_volume, BORDER_CROP)
        del full_volume
        print(f"  Cropped shape: {volume.shape}")

        porosity = (1 - volume).mean()
        print(f"  Porosity: {porosity:.4f}")

        # Extract pore network for the whole cropped volume
        print(f"  Extracting pore network (SNOW2)...")
        t_net = time.time()
        network = extract_porespy_network(volume)
        dt_net = time.time() - t_net
        print(f"  Network extracted in {dt_net:.1f}s")

        del volume

        # Save network
        network_save_path = os.path.join(
            output_dir,
            f'{vol_basename}.network.npz'
        )
        save_network(network, network_save_path)
        print(f"  Saved: {os.path.basename(network_save_path)}")

        del network

        dt_vol = time.time() - t_vol_start
        print(f"  Volume {vol_basename} done in {dt_vol:.1f}s")

    dt_total = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"All done. {len(volumes)} volume(s) processed in {dt_total:.1f}s")
    print(f"Networks saved to: {output_dir}")


if __name__ == '__main__':
    main()

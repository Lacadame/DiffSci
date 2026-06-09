#!/usr/bin/env python
"""
Unfold a generated porous media volume by cropping and mirroring.

Takes a binary .npy volume, crops 128 voxels from each side in all three
directions, then mirrors the result first along x, then along y, producing
a volume 2x larger in x and y. Also extracts a pore network (SNOW2) from
the unfolded volume and saves it.

Given input path/to/0.npy, produces:
    path/to/0.unfolded.npy
    path/to/0.unfolded.network.npz

Usage:
    python 0012-unfolding.py /path/to/0.npy

    # With a specific stone type (sets voxel length for the pore network)
    python 0012-unfolding.py /path/to/0.npy --stone Estaillades
"""

import argparse
import os
import resource
import time
import tracemalloc

import numpy as np
import psutil

from poregen.features.snow2 import snow2


BORDER_CROP = 128

VOXEL_LENGTHS = {
    'Bentheimer': 3.0035e-6,
    'Doddington': 2.6929e-6,
    'Estaillades': 3.31136e-6,
    'Ketton': 3.00006e-6,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop and mirror-unfold a binary porous media volume, then extract pore network.'
    )
    parser.add_argument(
        'volume_path', type=str,
        help='Path to the input .npy binary volume'
    )
    parser.add_argument(
        '--stone', type=str, default=None,
        choices=list(VOXEL_LENGTHS.keys()),
        help='Stone type (sets voxel length for SNOW2). If not given, voxel_size=1.0 is used.'
    )
    parser.add_argument(
        '--border-crop', type=int, default=BORDER_CROP,
        help=f'Number of voxels to crop from each side (default: {BORDER_CROP})'
    )
    return parser.parse_args()


def crop(volume, border):
    """Crop border voxels from all sides."""
    return volume[border:-border, border:-border, border:-border]


def mirror_unfold(volume):
    """
    Mirror-unfold: concatenate with x-flipped copy along x,
    then concatenate with y-flipped copy along y.

    (Nx, Ny, Nz) -> (2*Nx, Ny, Nz) -> (2*Nx, 2*Ny, Nz)
    """
    # Mirror in x
    x_flipped = volume[::-1, :, :]
    unfolded_x = np.concatenate([volume, x_flipped], axis=0)
    del x_flipped

    # Mirror in y
    y_flipped = unfolded_x[:, ::-1, :]
    unfolded_xy = np.concatenate([unfolded_x, y_flipped], axis=1)
    del y_flipped, unfolded_x

    return unfolded_xy


def extract_porespy_network(binary_volume, voxel_size=1.0):
    """Extract pore network using SNOW2 (pore=0, solid=1 convention)."""
    pore_space = 1 - binary_volume
    partitioning = snow2(pore_space, voxel_size=voxel_size)
    return partitioning.network


def save_network(network_dict, save_path):
    """Save porespy network dictionary to npz."""
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


def mem_report(label):
    """Print current memory usage with label."""
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    vms_gb = proc.memory_info().vms / (1024 ** 3)
    sys_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # KB -> GB
    print(f"  [MEM {label}] RSS={rss_gb:.2f} GB | VMS={vms_gb:.2f} GB | "
          f"Peak RSS={peak_gb:.2f} GB | System available={sys_avail_gb:.1f} GB",
          flush=True)


def main():
    args = parse_args()

    tracemalloc.start()

    volume_path = os.path.abspath(args.volume_path)
    volume_dir = os.path.dirname(volume_path)
    basename = os.path.basename(volume_path)

    # Derive output names: 0.npy -> 0.unfolded.npy, 0.unfolded.network.npz
    stem = basename.rsplit('.npy', 1)[0]
    unfolded_npy_path = os.path.join(volume_dir, f'{stem}.unfolded.npy')
    unfolded_network_path = os.path.join(volume_dir, f'{stem}.unfolded.network.npz')

    voxel_size = VOXEL_LENGTHS[args.stone] if args.stone else 1.0

    print(f"Input: {volume_path}")
    print(f"Border crop: {args.border_crop}")
    if args.stone:
        print(f"Stone: {args.stone}, voxel size: {voxel_size:.6e} m")
    print(f"System total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    mem_report("start")

    # --- Load and crop ---
    print("Loading volume...")
    volume = np.load(volume_path)
    print(f"  Original shape: {volume.shape}, dtype: {volume.dtype}")
    mem_report("after load")

    print(f"Cropping {args.border_crop} voxels from each side...")
    cropped = crop(volume, args.border_crop)
    del volume
    print(f"  Cropped shape: {cropped.shape}")
    mem_report("after crop")

    # --- Mirror-unfold ---
    print("Mirror-unfolding (x then y)...")
    unfolded = mirror_unfold(cropped)
    del cropped
    print(f"  Unfolded shape: {unfolded.shape}")
    n_voxels = np.prod(unfolded.shape)
    print(f"  Total voxels: {n_voxels:,} ({n_voxels * 1 / (1024**3):.2f} GB as bool, "
          f"{n_voxels * 8 / (1024**3):.1f} GB as float64)")
    mem_report("after unfold")

    # --- Save unfolded volume ---
    print(f"Saving unfolded volume to {unfolded_npy_path}...")
    np.save(unfolded_npy_path, unfolded)
    print("  Done.")
    mem_report("after save npy")

    # --- Extract and save pore network ---
    print("Extracting pore network (SNOW2)... this will take a while.")
    print(f"  Estimated SNOW2 memory: ~{n_voxels * 8 / (1024**3) * 3:.0f}-"
          f"{n_voxels * 8 / (1024**3) * 5:.0f} GB "
          f"(3-5 float64 arrays of {n_voxels:,} voxels)")
    mem_report("before SNOW2")

    t0 = time.time()
    network = extract_porespy_network(unfolded, voxel_size=voxel_size)
    elapsed = time.time() - t0
    print(f"  Network extracted in {elapsed:.1f}s")
    mem_report("after SNOW2")

    # tracemalloc snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("\n  [TRACEMALLOC] Top 10 Python allocations:")
    for stat in top_stats[:10]:
        print(f"    {stat}")

    print(f"Saving network to {unfolded_network_path}...")
    save_network(network, unfolded_network_path)
    print("  Done.")

    del unfolded, network
    mem_report("final")
    print("\nFinished.")


if __name__ == '__main__':
    main()

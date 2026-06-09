#!/usr/bin/env python
"""
Calculate porosity and permeability on strided sub-blocks of porous media volumes.

For each volume (reference + generated), divides into a grid of sub-blocks and
computes porosity (and optionally absolute permeability) per sub-block, saving
as spatial arrays.

Usage:
    # Porosity only (fast)
    python 0010-diversity-calculation.py --path /path/to/data --stone Bentheimer --divisions 4 --stride full --porosity-only

    # Full (porosity + permeability)
    python 0010-diversity-calculation.py --path /path/to/data --stone Bentheimer --divisions 4 --stride full
"""

import argparse
import os
import time

import numpy as np

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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate strided sub-block porosity and permeability'
    )
    parser.add_argument(
        '--path', type=str, required=True,
        help='Path to the data directory containing generated .npy volumes'
    )
    parser.add_argument(
        '--stone', type=str, required=True, choices=AVAILABLE_STONES,
        help='Stone type (sets voxel length and reference path)'
    )
    parser.add_argument(
        '--divisions', type=int, required=True,
        help='Number of divisions per side (e.g., 4 = quarter block size)'
    )
    parser.add_argument(
        '--stride', type=str, required=True, choices=['full', 'half'],
        help='Stride mode: full (non-overlapping) or half (50%% overlap)'
    )
    parser.add_argument(
        '--porosity-only', action='store_true', default=False,
        help='Only compute porosity (skip permeability, much faster)'
    )
    parser.add_argument(
        '--size', type=int, default=1280,
        help='Volume size prefix to look for (default: 1280)'
    )
    parser.add_argument(
        '--border-crop', type=int, default=128,
        help='Crop this many voxels from each border of generated volumes (default: 128)'
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
        '--recalculate', action='store_true', default=False,
        help='Recalculate even if output files already exist'
    )
    parser.add_argument(
        '--max-volumes', type=int, default=None,
        help='Maximum number of generated volumes to process'
    )
    parser.add_argument(
        '--recalculate-absolute-permeability', action='store_true',
        help='Only recalculate K from cached networks (no SNOW2). '
             'Requires networks to have been saved from a previous run.'
    )
    return parser.parse_args()


def get_generated_volume_paths(data_dir, size):
    """Get paths to all generated volumes of a given size."""
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


def compute_stride_grid(volume_side, divisions, stride_mode):
    """Compute sub-block parameters.

    Returns:
        block_size: size of each sub-block
        stride: step between sub-blocks
        n_blocks: number of blocks per side
    """
    block_size = volume_side // divisions
    if stride_mode == 'full':
        stride = block_size
    else:  # half
        stride = block_size // 2

    n_blocks = (volume_side - block_size) // stride + 1
    return block_size, stride, n_blocks


def compute_subblock_porosity(volume, block_size, stride, n_blocks):
    """Compute porosity for each sub-block.

    Returns array of shape [n_blocks, n_blocks, n_blocks, 1].
    """
    result = np.zeros((n_blocks, n_blocks, n_blocks, 1), dtype=np.float32)
    for i in range(n_blocks):
        for j in range(n_blocks):
            for k in range(n_blocks):
                x0 = i * stride
                y0 = j * stride
                z0 = k * stride
                subvol = volume[x0:x0+block_size, y0:y0+block_size, z0:z0+block_size]
                # Convention: pore=0, solid=1 -> porosity = mean of (1-volume)
                porosity = (1 - subvol).mean()
                result[i, j, k, 0] = porosity
    return result


def _network_cache_path(base_path, tag, i, j, k):
    """Path for a cached sub-block network."""
    return f"{base_path}.subblock_{i}_{j}_{k}_{tag}.network.npz"


def _save_network(network_dict, save_path):
    """Save porespy network dictionary to npz file."""
    save_dict = {}
    for key, value in network_dict.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, (int, float, str, bool)):
            save_dict[key] = np.array(value)
        elif isinstance(value, list):
            save_dict[key] = np.array(value)
    np.savez(save_path, **save_dict)


def compute_subblock_permeability(volume, block_size, stride, n_blocks, voxel_size,
                                  cache_base_path=None, tag=None,
                                  use_cached_networks=False):
    """Compute absolute permeability (K_x, K_y, K_z) for each sub-block.

    Returns array of shape [n_blocks, n_blocks, n_blocks, 3].
    """
    from diffsci2.extra.pore.permeability_from_pnm import PoreNetworkPermeability

    result = np.full((n_blocks, n_blocks, n_blocks, 3), np.nan, dtype=np.float64)
    total = n_blocks ** 3
    count = 0

    for i in range(n_blocks):
        for j in range(n_blocks):
            for k in range(n_blocks):
                count += 1
                print(f"    Sub-block ({i},{j},{k}) [{count}/{total}]", end="")
                t0 = time.time()

                try:
                    net_path = None
                    if cache_base_path and tag:
                        net_path = _network_cache_path(cache_base_path, tag, i, j, k)

                    # Try loading cached network
                    if use_cached_networks and net_path and os.path.exists(net_path):
                        network = dict(np.load(net_path))
                    else:
                        from poregen.features.snow2 import snow2
                        x0 = i * stride
                        y0 = j * stride
                        z0 = k * stride
                        subvol = volume[x0:x0+block_size, y0:y0+block_size, z0:z0+block_size]
                        pore_space = 1 - subvol
                        partitioning = snow2(pore_space, voxel_size=1.0)
                        network = partitioning.network
                        # Save network for future reuse
                        if net_path:
                            _save_network(network, net_path)

                    # Compute absolute permeability only
                    pn_wrapper = PoreNetworkPermeability.from_porespy_network(
                        network,
                        volume_length=block_size,
                        voxel_size=voxel_size,
                    )
                    abs_perm = pn_wrapper.calculate_absolute_permeability()

                    result[i, j, k, 0] = abs_perm.K_x_physical
                    result[i, j, k, 1] = abs_perm.K_y_physical
                    result[i, j, k, 2] = abs_perm.K_z_physical

                    dt = time.time() - t0
                    K_mean = abs_perm.K_mean_physical
                    print(f" K_mean={K_mean*1e15:.2f} mD ({dt:.1f}s)")

                except Exception as e:
                    dt = time.time() - t0
                    print(f" ERROR: {e} ({dt:.1f}s)")

    return result


def output_tag(divisions, stride_mode):
    return f"strides_{divisions}_{stride_mode}"


def process_volume(volume, volume_side, divisions, stride_mode, voxel_size, porosity_only,
                    cache_base_path=None, tag=None, use_cached_networks=False):
    """Process a single volume: compute porosity and optionally permeability on strided sub-blocks."""
    block_size, stride, n_blocks = compute_stride_grid(volume_side, divisions, stride_mode)
    print(f"  Volume side: {volume_side}, block_size: {block_size}, "
          f"stride: {stride}, n_blocks: {n_blocks} ({n_blocks**3} total)")

    # Porosity
    if volume is not None:
        print(f"  Computing porosity...")
        t0 = time.time()
        porosity_array = compute_subblock_porosity(volume, block_size, stride, n_blocks)
        print(f"  Porosity done in {time.time()-t0:.1f}s")
        print(f"  Porosity range: [{porosity_array.min():.4f}, {porosity_array.max():.4f}], "
              f"mean: {porosity_array.mean():.4f}")
    else:
        porosity_array = None

    # Permeability
    perm_array = None
    if not porosity_only:
        print(f"  Computing permeability...")
        t0 = time.time()
        perm_array = compute_subblock_permeability(
            volume, block_size, stride, n_blocks, voxel_size,
            cache_base_path=cache_base_path, tag=tag,
            use_cached_networks=use_cached_networks,
        )
        print(f"  Permeability done in {time.time()-t0:.1f}s")
        valid = ~np.isnan(perm_array)
        if valid.any():
            print(f"  Permeability range: [{perm_array[valid].min()*1e15:.2f}, "
                  f"{perm_array[valid].max()*1e15:.2f}] mD")

    return porosity_array, perm_array


def main():
    args = parse_args()

    stone = args.stone
    data_dir = args.path
    divisions = args.divisions
    stride_mode = args.stride
    porosity_only = args.porosity_only
    voxel_size = VOXEL_LENGTHS[stone]

    tag = output_tag(divisions, stride_mode)

    print(f"Stone: {stone}")
    print(f"Path: {data_dir}")
    print(f"Divisions: {divisions}")
    print(f"Stride: {stride_mode}")
    print(f"Porosity only: {porosity_only}")
    print(f"Voxel size: {voxel_size:.6e} m")
    print(f"Output tag: {tag}")

    # --- Reference volume ---
    if not args.skip_reference:
        ref_path = REFERENCE_PATHS[stone]
        ref_porosity_path = os.path.join(data_dir, f"reference.calculated_porosity_{tag}.npy")
        ref_perm_path = os.path.join(data_dir, f"reference.calculated_permeability_{tag}.npy")

        skip_ref = False
        if not args.recalculate:
            if os.path.exists(ref_porosity_path):
                if porosity_only or os.path.exists(ref_perm_path):
                    print(f"\nReference already calculated, skipping")
                    skip_ref = True

        if not skip_ref and os.path.exists(ref_path):
            recalc_only = args.recalculate_absolute_permeability
            ref_cache_base = os.path.join(data_dir, "reference")

            if recalc_only:
                print(f"\nRecalculating reference K from cached networks...")
                ref_side = args.reference_shape[0]
                porosity_arr, perm_arr = process_volume(
                    None, ref_side, divisions, stride_mode, voxel_size,
                    porosity_only=False,
                    cache_base_path=ref_cache_base, tag=tag,
                    use_cached_networks=True,
                )
            else:
                print(f"\nProcessing reference: {ref_path}")
                ref_shape = tuple(args.reference_shape)
                ref_volume = np.fromfile(ref_path, dtype=np.uint8).reshape(ref_shape)
                print(f"  Reference shape: {ref_volume.shape}")
                ref_side = ref_shape[0]
                porosity_arr, perm_arr = process_volume(
                    ref_volume, ref_side, divisions, stride_mode, voxel_size, porosity_only,
                    cache_base_path=ref_cache_base, tag=tag,
                )

            if porosity_arr is not None:
                np.save(ref_porosity_path, porosity_arr)
                print(f"  Saved: {os.path.basename(ref_porosity_path)}")

            if perm_arr is not None:
                np.save(ref_perm_path, perm_arr)
                print(f"  Saved: {os.path.basename(ref_perm_path)}")
        elif not skip_ref:
            print(f"\nReference file not found: {ref_path}, skipping")

    # --- Generated volumes ---
    print(f"\nLooking for generated volumes in {data_dir}...")
    volume_paths = get_generated_volume_paths(data_dir, args.size)

    if args.max_volumes is not None:
        volume_paths = volume_paths[:args.max_volumes]

    print(f"Found {len(volume_paths)} generated volumes")

    effective_size = args.size - 2 * args.border_crop if args.border_crop > 0 else args.size
    if args.border_crop > 0:
        print(f"Border crop: {args.border_crop} -> effective size: {effective_size}")

    for vol_idx, vol_path in enumerate(volume_paths):
        vol_name = os.path.basename(vol_path)
        base_name = vol_name.replace('.npy', '')

        porosity_out = os.path.join(data_dir, f"{base_name}.calculated_porosity_{tag}.npy")
        perm_out = os.path.join(data_dir, f"{base_name}.calculated_permeability_{tag}.npy")

        # Check if already done
        if not args.recalculate:
            if os.path.exists(porosity_out):
                if porosity_only or os.path.exists(perm_out):
                    print(f"\n[{vol_idx+1}/{len(volume_paths)}] {vol_name} already calculated, skipping")
                    continue

        recalc_only = args.recalculate_absolute_permeability
        cache_base = os.path.join(data_dir, base_name)

        if recalc_only:
            print(f"\n[{vol_idx+1}/{len(volume_paths)}] Recalculating K for {vol_name}")
            vol_side = effective_size
            porosity_arr, perm_arr = process_volume(
                None, vol_side, divisions, stride_mode, voxel_size,
                porosity_only=False,
                cache_base_path=cache_base, tag=tag,
                use_cached_networks=True,
            )
        else:
            print(f"\n[{vol_idx+1}/{len(volume_paths)}] Processing {vol_name}")
            volume = np.load(vol_path)

            if args.border_crop > 0:
                volume = center_crop(volume, args.border_crop)
                print(f"  Cropped to {volume.shape}")

            vol_side = volume.shape[0]
            porosity_arr, perm_arr = process_volume(
                volume, vol_side, divisions, stride_mode, voxel_size, porosity_only,
                cache_base_path=cache_base, tag=tag,
            )
            del volume

        if porosity_arr is not None:
            np.save(porosity_out, porosity_arr)
            print(f"  Saved: {os.path.basename(porosity_out)}")

        if perm_arr is not None:
            np.save(perm_out, perm_arr)
            print(f"  Saved: {os.path.basename(perm_out)}")

    print("\nDone!")


if __name__ == '__main__':
    main()

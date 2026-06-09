#!/usr/bin/env python
"""
Large-volume metrics evaluator: subvolume-stride approach.

Takes a single large volume (1280x1280x4352), crops 128 voxels from each side
to get (1024x1024x4096), then extracts subvolumes of 1024^3 along z with a
configurable stride (default 1024 = non-overlapping, 512 = half-overlap).

For each subvolume: runs SNOW2, computes porosity and two-phase flow properties.
This is the "real" evaluation path (not the subnetwork trim shortcut).

Usage:
    # Full stride (4 non-overlapping subvolumes) for Bentheimer
    python 0005d-porosity-field-new-metrics-evaluator-large-subvol.py \
        --stone Bentheimer \
        --volume-path /home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/bentheimer_pfield_gen_case_10_large/data/0.npy \
        --stride 1024

    # Half stride (7 overlapping subvolumes)
    python 0005d-porosity-field-new-metrics-evaluator-large-subvol.py \
        --stone Bentheimer \
        --volume-path .../0.npy \
        --stride 512
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

# Volume constants
BORDER_CROP = 128
SUBCUBE_SIZE = 1024


def compute_abs_perm_from_network(network_path, voxel_size):
    """Load a cached .network.npz and compute only absolute permeability."""
    net = np.load(network_path)
    pnp = PoreNetworkPermeability.from_binary_volume(
        network=net, voxel_size=voxel_size,
    )
    K = pnp.calculate_absolute_permeability()
    return {
        'K_abs_x': K.K_x, 'K_abs_y': K.K_y, 'K_abs_z': K.K_z,
        'K_abs_mean': K.K_mean,
        'K_abs_x_physical': K.K_x_physical,
        'K_abs_y_physical': K.K_y_physical,
        'K_abs_z_physical': K.K_z_physical,
        'K_abs_mean_physical': K.K_mean_physical,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate large-volume metrics: crop 128, stride subvolumes along z'
    )
    parser.add_argument(
        '--stone', type=str, required=True, choices=AVAILABLE_STONES,
        help='Stone type (sets voxel length and reference path)'
    )
    parser.add_argument(
        '--volume-path', type=str, required=True,
        help='Path to the large .npy volume (1280x1280x4352)'
    )
    parser.add_argument(
        '--stride', type=int, default=1024,
        help='Stride along z in voxels (1024=full, 512=half-overlap). Default: 1024'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/metrics',
        help='Directory to save output .npz file'
    )
    parser.add_argument(
        '--reference-path', type=str, default=None,
        help='Path to reference .raw volume file (overrides --stone default)'
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
    parser.add_argument(
        '--use-cached-network', action='store_true',
        help='Skip SNOW2 extraction, load cached .network.npz files instead'
    )
    parser.add_argument(
        '--recalculate-absolute-permeability', action='store_true',
        help='Only recalculate K_abs from cached networks and patch existing results. '
             'Does not load volumes or re-run drainage/kr.'
    )
    return parser.parse_args()


def center_crop(volume, border):
    """Crop border voxels from all sides of the volume."""
    if border <= 0:
        return volume
    return volume[border:-border, border:-border, border:-border]


def get_subvolume_slices(total_z, subcube_size, stride):
    """
    Return subvolume slices along z.

    For stride=1024: 4 non-overlapping subvolumes
    For stride=512:  7 half-overlapping subvolumes
    """
    slices = []
    z = 0
    idx = 0
    while z + subcube_size <= total_z:
        slices.append((idx, z, z + subcube_size))
        z += stride
        idx += 1
    return slices


def extract_porespy_network(binary_volume):
    """
    Extract pore network from binary volume using SNOW algorithm.
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
    """
    porosity = (1 - binary_volume).mean()

    pn_wrapper = PoreNetworkPermeability.from_porespy_network(
        porespy_network,
        volume_length=binary_volume.shape[0],
        voxel_size=voxel_size,
    )

    abs_perm = pn_wrapper.calculate_absolute_permeability()

    _ = pn_wrapper.run_drainage_simulation(
        contact_angle=contact_angle,
        surface_tension=surface_tension,
    )

    rel_perm = pn_wrapper.calculate_relative_permeability_curves()

    return {
        'porosity': porosity,
        'K_abs_x': abs_perm.K_x,
        'K_abs_y': abs_perm.K_y,
        'K_abs_z': abs_perm.K_z,
        'K_abs_mean': abs_perm.K_mean,
        'K_abs_x_physical': abs_perm.K_x_physical,
        'K_abs_y_physical': abs_perm.K_y_physical,
        'K_abs_z_physical': abs_perm.K_z_physical,
        'K_abs_mean_physical': abs_perm.K_mean_physical,
        'Sw': rel_perm.Sw,
        'Snw': rel_perm.Snwp,
        'Pc': rel_perm.Pc,
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
    stride = args.stride
    voxel_length = VOXEL_LENGTHS[stone]
    reference_path = args.reference_path or REFERENCE_PATHS[stone]

    # Output path
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f'metrics_{stone.lower()}_largescale_stride{stride}_twophase.npz'
    )

    # Network save dir (same as volume)
    volume_dir = os.path.dirname(os.path.abspath(args.volume_path))

    print(f"Stone: {stone}")
    print(f"Voxel length: {voxel_length:.6e} m")
    print(f"Volume path: {args.volume_path}")
    print(f"Stride: {stride} voxels")
    print(f"Contact angle: {args.contact_angle} deg")
    print(f"Surface tension: {args.surface_tension} N/m")
    print(f"Output: {output_path}")

    # --- Fast path: recalculate only K_abs from cached networks ---
    if args.recalculate_absolute_permeability:
        print("\n*** Recalculating absolute permeability only (patching existing results) ***")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Existing results not found: {output_path}")

        existing = dict(np.load(output_path, allow_pickle=True))
        patched = dict(existing)

        # Patch reference
        ref_net_path = reference_path.replace('.raw', '.network.npz')
        if f'reference_K_abs_mean' in existing and os.path.exists(ref_net_path):
            print(f"\nRecalculating reference K_abs from {ref_net_path}...")
            K = compute_abs_perm_from_network(ref_net_path, voxel_length)
            for key, val in K.items():
                patched[f'reference_{key}'] = val
            print(f"  K_abs (mean): {K['K_abs_mean_physical'] * 1e15:.2f} mD")

        # Patch subvolumes
        size_key = f'generated_{SUBCUBE_SIZE}'
        n_key = f'{size_key}_n_volumes'
        if n_key in existing:
            n_vols = int(existing[n_key])
            vol_basename = os.path.splitext(os.path.basename(args.volume_path))[0]
            total_z = 1024 * 4  # after crop: 4096
            subvol_slices = get_subvolume_slices(total_z, SUBCUBE_SIZE, stride)

            K_lists = {k: [] for k in [
                'K_abs_x', 'K_abs_y', 'K_abs_z', 'K_abs_mean',
                'K_abs_x_physical', 'K_abs_y_physical', 'K_abs_z_physical',
                'K_abs_mean_physical',
            ]}

            print(f"\nRecalculating K_abs for {n_vols} subvolumes...")
            for sub_idx, z_lo, z_hi in subvol_slices[:n_vols]:
                label = f"subvol_{sub_idx:02d}_z{z_lo}-{z_hi}"
                net_path = os.path.join(volume_dir, f'{vol_basename}.{label}.network.npz')
                if not os.path.exists(net_path):
                    print(f"  [{sub_idx+1}/{n_vols}] MISSING: {os.path.basename(net_path)}")
                    continue
                K = compute_abs_perm_from_network(net_path, voxel_length)
                for key in K_lists:
                    K_lists[key].append(K[key])
                print(f"  [{sub_idx+1}/{n_vols}] {label}: K_abs={K['K_abs_mean_physical']*1e15:.2f} mD")

            for key, vals in K_lists.items():
                if len(vals) > 0:
                    patched[f'{size_key}_{key}'] = np.array(vals)

            if len(K_lists['K_abs_mean_physical']) > 0:
                perms = np.array(K_lists['K_abs_mean_physical']) * 1e15
                print(f"  Summary: K_abs (mD): mean={perms.mean():.2f}, std={perms.std():.2f}")

        # Save
        print(f"\nSaving patched results to {output_path}...")
        save_dict = {}
        for k, v in patched.items():
            if v is not None:
                save_dict[k] = v
        np.savez(output_path, **save_dict)
        print("Done!")
        return

    timing = {}
    results = {
        'stone': stone,
        'voxel_length': voxel_length,
        'border_crop': BORDER_CROP,
        'subcube_size': SUBCUBE_SIZE,
        'stride': stride,
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

        ref_net_path = reference_path.replace('.raw', '.network.npz')
        if args.use_cached_network and os.path.exists(ref_net_path):
            print(f"  Loading cached network from {ref_net_path}...")
            network = dict(np.load(ref_net_path))
        else:
            print(f"  Extracting pore network...")
            network = extract_porespy_network(reference_data)

        print(f"  Computing two-phase flow metrics...")
        ref_metrics = compute_two_phase_metrics(
            reference_data, network, voxel_length,
            args.contact_angle, args.surface_tension
        )

        timing['reference'] = time.time() - t_start
        print(f"  Reference metrics computed in {timing['reference']:.2f}s")
        print(f"    Porosity: {ref_metrics['porosity']:.4f}")
        print(f"    K_abs (mean): {ref_metrics['K_abs_mean_physical'] * 1e15:.2f} mD")

        for key, value in ref_metrics.items():
            results[f'reference_{key}'] = value

        del reference_data, network

    # -------------------------------------------------------------------------
    # Load, crop, and process subvolumes
    # -------------------------------------------------------------------------
    print(f"\nLoading volume: {args.volume_path}")
    full_volume = np.load(args.volume_path)
    print(f"  Loaded shape: {full_volume.shape}")

    print(f"  Cropping {BORDER_CROP} voxels from each side...")
    cropped = center_crop(full_volume, BORDER_CROP)
    del full_volume
    print(f"  Cropped shape: {cropped.shape}")

    total_z = cropped.shape[2]
    subvol_slices = get_subvolume_slices(total_z, SUBCUBE_SIZE, stride)
    n_subs = len(subvol_slices)
    print(f"  {n_subs} subvolumes of {SUBCUBE_SIZE}^3 with stride {stride}")

    all_metrics = []
    t_start = time.time()

    for sub_idx, z_lo, z_hi in subvol_slices:
        label = f"subvol_{sub_idx:02d}_z{z_lo}-{z_hi}"
        print(f"\n  [{sub_idx + 1}/{n_subs}] {label}")

        subcube = cropped[:, :, z_lo:z_hi].copy()
        print(f"    Shape: {subcube.shape}, porosity (voxel): {(1 - subcube).mean():.4f}")

        # Extract or load pore network
        vol_basename = os.path.splitext(os.path.basename(args.volume_path))[0]
        network_save_path = os.path.join(
            volume_dir,
            f'{vol_basename}.{label}.network.npz'
        )
        if args.use_cached_network and os.path.exists(network_save_path):
            print(f"    Loading cached network from {os.path.basename(network_save_path)}")
            network = dict(np.load(network_save_path))
        else:
            print(f"    Extracting pore network (SNOW2)...")
            t_net = time.time()
            network = extract_porespy_network(subcube)
            dt_net = time.time() - t_net
            print(f"    Network extracted in {dt_net:.1f}s")
            save_network(network, network_save_path)
            print(f"    Saved: {os.path.basename(network_save_path)}")

        # Compute two-phase metrics
        try:
            t_met = time.time()
            metrics = compute_two_phase_metrics(
                subcube, network, voxel_length,
                args.contact_angle, args.surface_tension
            )
            dt_met = time.time() - t_met
            all_metrics.append(metrics)
            print(f"    Porosity: {metrics['porosity']:.4f}, "
                  f"K_abs: {metrics['K_abs_mean_physical'] * 1e15:.2f} mD "
                  f"(metrics in {dt_met:.1f}s)")
        except Exception as e:
            print(f"    Error computing metrics: {e}")

        del subcube, network

    del cropped

    timing['subvolumes'] = time.time() - t_start
    print(f"\n  All subvolumes processed in {timing['subvolumes']:.2f}s")

    if len(all_metrics) == 0:
        print("  No valid metrics computed!")
        return

    # -------------------------------------------------------------------------
    # Aggregate and save
    # -------------------------------------------------------------------------
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

    # Print summary
    porosities = results[f'{size_key}_porosity']
    perms = results[f'{size_key}_K_abs_mean_physical'] * 1e15
    print(f"\n  Summary ({n_subs} subvolumes of {SUBCUBE_SIZE}^3, stride {stride}):")
    print(f"    Porosity: mean={porosities.mean():.4f}, std={porosities.std():.4f}")
    print(f"    K_abs (mD): mean={perms.mean():.2f}, std={perms.std():.2f}")

    # Save
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
    print("Done!")


if __name__ == '__main__':
    main()

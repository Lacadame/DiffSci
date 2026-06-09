#!/usr/bin/env python
"""
Buckley-Leverett evaluator for generated porous media volumes.

Takes pre-extracted pore networks (.network.npz files) — no binary volumes
needed — and computes oil/water two-phase flow properties + Buckley-Leverett
production metrics.

Key differences from 0005b:
- Works from saved .network.npz files, skipping SNOW extraction entirely
- Uses oil/water drainage parameters (not mercury injection)
- Fits Corey relative permeability models
- Runs Buckley-Leverett solver (Welge construction)
- Reports production metrics: breakthrough PVI, recovery curves

Usage:
    # Simple usage with stone name
    python 0005d-porosity-field-buckley-leverett.py --stone Bentheimer

    # Different generation case
    python 0005d-porosity-field-buckley-leverett.py --stone Bentheimer --pattern _pfield_gen_case_8

    # Custom viscosity ratio
    python 0005d-porosity-field-buckley-leverett.py --stone Bentheimer --mu-oil 10e-3
"""

import argparse
import os
import time

import numpy as np

from diffsci2.extra.pore.permeability_from_pnm import PoreNetworkPermeability
from diffsci2.extra.pore.corey_model import fit_corey_model
from diffsci2.extra.pore.buckley_leverett import BuckleyLeverettSolver


# ============================================================================
# Stone-specific constants
# ============================================================================

VOXEL_LENGTHS = {
    'Bentheimer': 3.0035e-6,
    'Doddington': 2.6929e-6,
    'Estaillades': 3.31136e-6,
    'Ketton': 3.00006e-6,
}

# Approximate literature porosities (used for BL when volume is not available)
POROSITIES = {
    'Bentheimer': 0.22,
    'Doddington': 0.19,
    'Estaillades': 0.12,
    'Ketton': 0.12,
}

# Reference network paths (saved by 0005b alongside the .raw files)
DATA_DIR = '/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/'
REFERENCE_NETWORK_PATHS = {
    'Bentheimer': DATA_DIR + 'Bentheimer_1000c_3p0035um.network.npz',
    'Doddington': DATA_DIR + 'Doddington_1000c_2p6929um.network.npz',
    'Estaillades': DATA_DIR + 'Estaillades_1000c_3p31136um.network.npz',
    'Ketton': DATA_DIR + 'Ketton_1000c_3p00006um.network.npz',
}

GENERATED_DATA_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/'
DEFAULT_PATTERN = '_pfield_gen_3'

AVAILABLE_STONES = list(VOXEL_LENGTHS.keys())

# Standard PVI values for reporting recovery curves
STANDARD_PVI = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])


# ============================================================================
# Helpers
# ============================================================================

def get_generated_path(stone, pattern=DEFAULT_PATTERN):
    return GENERATED_DATA_DIR + stone.lower() + pattern + '/data'


def get_network_paths(data_dir, size):
    """Get paths to all .network.npz files for a given volume size."""
    files = os.listdir(data_dir)
    paths = []
    for f in sorted(files):
        if f.startswith(f'{size}_') and f.endswith('.network.npz'):
            paths.append(os.path.join(data_dir, f))
    return paths


def compute_buckley_leverett_metrics(
    network_path, voxel_size, porosity,
    contact_angle, surface_tension,
    mu_w, mu_nw,
):
    """
    From a single .network.npz file, compute:
    - Absolute permeability
    - kr curves (via drainage)
    - Corey fit
    - Buckley-Leverett solution

    Returns dict of scalar and array metrics.
    """
    net = np.load(network_path)

    # Build PoreNetworkPermeability from saved network (no volume needed)
    pn_wrapper = PoreNetworkPermeability.from_binary_volume(
        network=net, voxel_size=voxel_size,
    )

    # Absolute permeability
    abs_perm = pn_wrapper.calculate_absolute_permeability()

    # Drainage with oil/water parameters
    pn_wrapper.run_drainage_simulation(
        contact_angle=contact_angle,
        surface_tension=surface_tension,
    )

    # Relative permeability curves
    rel_perm = pn_wrapper.calculate_relative_permeability_curves()

    # Fit Corey model
    corey = fit_corey_model(rel_perm, direction='mean')

    # Buckley-Leverett (dimensionless — use unit geometry, report in PVI)
    solver = BuckleyLeverettSolver(
        kr_model=corey,
        mu_w=mu_w,
        mu_nw=mu_nw,
        porosity=porosity,
        total_rate=1.0,
        area=1.0,
        length=1.0,
    )
    bl_result = solver.welge_construction()

    # PVI at breakthrough = 1 / xD_shock
    pvi_bt = 1.0 / bl_result.xD_shock if bl_result.xD_shock > 0 else np.inf

    # Recovery at standard PVI values
    # PVI = qt * t / (phi * A * L) => t = PVI * phi * A * L / qt
    # With unit geometry: t = PVI * porosity
    recovery_at_pvi = np.array([
        solver.oil_recovery(pvi * porosity, bl_result) for pvi in STANDARD_PVI
    ])

    return {
        # Absolute permeability
        'K_abs_x': abs_perm.K_x,
        'K_abs_y': abs_perm.K_y,
        'K_abs_z': abs_perm.K_z,
        'K_abs_mean': abs_perm.K_mean,
        'K_abs_mean_physical': abs_perm.K_mean_physical,
        # Corey parameters
        'corey_Swr': corey.Swr,
        'corey_Snwr': corey.Snwr,
        'corey_kr_w0': corey.kr_w0,
        'corey_kr_nw0': corey.kr_nw0,
        'corey_nw': corey.nw,
        'corey_nnw': corey.nnw,
        # BL results
        'Sw_shock': bl_result.Sw_shock,
        'fw_shock': bl_result.fw_shock,
        'xD_shock': bl_result.xD_shock,
        'pvi_breakthrough': pvi_bt,
        'recovery_at_breakthrough': solver.oil_recovery(
            pvi_bt * porosity, bl_result
        ) if np.isfinite(pvi_bt) else 0.0,
        'recovery_at_pvi': recovery_at_pvi,
        # Raw kr data (for reference)
        'Sw': rel_perm.Sw,
        'Pc': rel_perm.Pc,
        'kr_wetting_mean': rel_perm.kr_wetting_mean,
        'kr_nonwetting_mean': rel_perm.kr_nonwetting_mean,
        # fw curve from the solver
        'bl_Sw_array': bl_result.Sw_array,
        'bl_fw_array': bl_result.fw_array,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Buckley-Leverett production metrics from pre-extracted pore networks'
    )
    parser.add_argument(
        '--stone', type=str, default=None, choices=AVAILABLE_STONES,
        help='Stone type (sets voxel length, porosity, paths automatically)'
    )
    parser.add_argument(
        '--pattern', type=str, default=DEFAULT_PATTERN,
        help=f'Generation pattern suffix (default: {DEFAULT_PATTERN})'
    )
    parser.add_argument(
        '--generated-dir', type=str, default=None,
        help='Directory containing .network.npz files (overrides --stone/--pattern)'
    )
    parser.add_argument(
        '--reference-network', type=str, default=None,
        help='Path to reference .network.npz (overrides --stone default)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output .npz path (default: metrics_{stone}_{pattern}_buckley.npz)'
    )
    parser.add_argument(
        '--voxel-length', type=float, default=None,
        help='Voxel length in meters (overrides --stone)'
    )
    parser.add_argument(
        '--porosity', type=float, default=None,
        help='Porosity for BL calculation (overrides --stone default)'
    )
    parser.add_argument(
        '--volume-sizes', type=str, default='1280',
        help='Comma-separated volume sizes (default: 1280)'
    )
    parser.add_argument(
        '--skip-reference', action='store_true',
        help='Skip processing reference network'
    )
    parser.add_argument(
        '--max-volumes', type=int, default=None,
        help='Max networks to process per size (default: all)'
    )
    # Oil/water parameters
    parser.add_argument(
        '--contact-angle', type=float, default=140.0,
        help='Oil/water contact angle [degrees] (default: 140, water-wet)'
    )
    parser.add_argument(
        '--surface-tension', type=float, default=0.03,
        help='Oil/water interfacial tension [N/m] (default: 0.03)'
    )
    parser.add_argument(
        '--mu-water', type=float, default=1e-3,
        help='Water viscosity [Pa.s] (default: 1e-3 = 1 cP)'
    )
    parser.add_argument(
        '--mu-oil', type=float, default=5e-3,
        help='Oil viscosity [Pa.s] (default: 5e-3 = 5 cP, light crude)'
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Resolve stone-based defaults
    if args.stone is not None:
        stone = args.stone
        generated_dir = args.generated_dir or get_generated_path(stone, args.pattern)
        ref_network = args.reference_network or REFERENCE_NETWORK_PATHS[stone]
        voxel_length = args.voxel_length or VOXEL_LENGTHS[stone]
        porosity = args.porosity or POROSITIES[stone]
        pattern_suffix = args.pattern.replace('_', '-').strip('-')
        output_path = args.output or f'metrics_{stone.lower()}_{pattern_suffix}_buckley.npz'
    else:
        if args.generated_dir is None:
            raise ValueError("Must specify --stone or --generated-dir")
        if args.voxel_length is None:
            raise ValueError("Must specify --stone or --voxel-length")
        if args.porosity is None:
            raise ValueError("Must specify --stone or --porosity")
        stone = None
        generated_dir = args.generated_dir
        ref_network = args.reference_network
        voxel_length = args.voxel_length
        porosity = args.porosity
        output_path = args.output or 'metrics_buckley.npz'

    print(f"Stone: {stone or 'custom'}")
    print(f"Pattern: {args.pattern}")
    print(f"Voxel length: {voxel_length:.6e} m")
    print(f"Porosity (for BL): {porosity:.3f}")
    print(f"Generated dir: {generated_dir}")
    print(f"Reference network: {ref_network}")
    print(f"Output: {output_path}")
    print(f"Oil/water: theta={args.contact_angle} deg, sigma={args.surface_tension} N/m")
    print(f"Viscosities: mu_w={args.mu_water*1e3:.1f} cP, mu_o={args.mu_oil*1e3:.1f} cP")
    print(f"Mobility ratio M ~ {args.mu_oil/args.mu_water:.1f}")

    timing = {}
    results = {
        'stone': stone or 'custom',
        'pattern': args.pattern,
        'voxel_length': voxel_length,
        'porosity': porosity,
        'contact_angle': args.contact_angle,
        'surface_tension': args.surface_tension,
        'mu_water': args.mu_water,
        'mu_oil': args.mu_oil,
        'standard_pvi': STANDARD_PVI,
    }

    flow_kwargs = dict(
        voxel_size=voxel_length,
        porosity=porosity,
        contact_angle=args.contact_angle,
        surface_tension=args.surface_tension,
        mu_w=args.mu_water,
        mu_nw=args.mu_oil,
    )

    # --- Reference network ---
    if not args.skip_reference and ref_network and os.path.exists(ref_network):
        print(f"\nProcessing reference network: {ref_network}")
        t_start = time.time()

        try:
            ref_metrics = compute_buckley_leverett_metrics(ref_network, **flow_kwargs)
            timing['reference'] = time.time() - t_start

            print(f"  K_abs: {ref_metrics['K_abs_mean_physical'] * 1e12:.3f} Darcy")
            print(f"  Corey: Swr={ref_metrics['corey_Swr']:.3f}, "
                  f"Snwr={ref_metrics['corey_Snwr']:.3f}, "
                  f"nw={ref_metrics['corey_nw']:.2f}, nnw={ref_metrics['corey_nnw']:.2f}")
            print(f"  BL: Sw_shock={ref_metrics['Sw_shock']:.3f}, "
                  f"PVI_bt={ref_metrics['pvi_breakthrough']:.3f}, "
                  f"Recovery_bt={ref_metrics['recovery_at_breakthrough']:.1%}")
            print(f"  Time: {timing['reference']:.1f}s")

            for key, value in ref_metrics.items():
                results[f'reference_{key}'] = value
        except Exception as e:
            print(f"  Error: {e}")
            timing['reference'] = time.time() - t_start

    # --- Generated networks ---
    volume_sizes = [int(x.strip()) for x in args.volume_sizes.split(',')]

    for size in volume_sizes:
        print(f"\nProcessing generated {size}^3 networks...")
        t_start = time.time()

        net_paths = get_network_paths(generated_dir, size)
        if len(net_paths) == 0:
            print(f"  No {size}^3 networks found in {generated_dir}")
            continue

        if args.max_volumes is not None:
            net_paths = net_paths[:args.max_volumes]

        print(f"  Found {len(net_paths)} networks")

        all_metrics = []
        for idx, net_path in enumerate(net_paths):
            name = os.path.basename(net_path)
            print(f"  [{idx+1}/{len(net_paths)}] {name}", end='')

            try:
                m = compute_buckley_leverett_metrics(net_path, **flow_kwargs)
                all_metrics.append(m)
                print(f"  K={m['K_abs_mean_physical']*1e12:.3f}D, "
                      f"PVI_bt={m['pvi_breakthrough']:.3f}, "
                      f"Rec_bt={m['recovery_at_breakthrough']:.1%}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        timing[f'generated_{size}'] = time.time() - t_start
        print(f"  Done in {timing[f'generated_{size}']:.1f}s")

        if len(all_metrics) == 0:
            continue

        n = len(all_metrics)
        prefix = f'generated_{size}'
        results[f'{prefix}_n_volumes'] = n

        # Scalar metrics -> arrays
        for key in [
            'K_abs_mean', 'K_abs_mean_physical', 'K_abs_x', 'K_abs_y', 'K_abs_z',
            'corey_Swr', 'corey_Snwr', 'corey_kr_w0', 'corey_kr_nw0',
            'corey_nw', 'corey_nnw',
            'Sw_shock', 'fw_shock', 'xD_shock',
            'pvi_breakthrough', 'recovery_at_breakthrough',
        ]:
            results[f'{prefix}_{key}'] = np.array([m[key] for m in all_metrics])

        # Recovery curves (fixed-length, same PVI grid)
        results[f'{prefix}_recovery_at_pvi'] = np.array(
            [m['recovery_at_pvi'] for m in all_metrics]
        )  # shape: (n_volumes, len(STANDARD_PVI))

        # Variable-length arrays (kr data per sample)
        for key in ['Sw', 'Pc', 'kr_wetting_mean', 'kr_nonwetting_mean',
                     'bl_Sw_array', 'bl_fw_array']:
            results[f'{prefix}_{key}'] = np.array(
                [m[key] for m in all_metrics], dtype=object
            )

        # Print summary
        pvi_bt = results[f'{prefix}_pvi_breakthrough']
        rec_bt = results[f'{prefix}_recovery_at_breakthrough']
        perms = results[f'{prefix}_K_abs_mean_physical'] * 1e12
        print(f"  Summary:")
        print(f"    K_abs (Darcy): {perms.mean():.3f} +/- {perms.std():.3f}")
        print(f"    PVI at BT: {pvi_bt.mean():.3f} +/- {pvi_bt.std():.3f}")
        print(f"    Recovery at BT: {rec_bt.mean():.1%} +/- {rec_bt.std():.1%}")

    # --- Save ---
    results['timing_keys'] = list(timing.keys())
    results['timing_values'] = list(timing.values())
    results['total_time'] = sum(timing.values())

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True) \
        if os.path.dirname(output_path) else None

    save_dict = {k: v for k, v in results.items() if v is not None}
    np.savez(output_path, **save_dict)

    print(f"\n=== Summary ===")
    print(f"Total time: {results['total_time']:.1f}s")
    print(f"Saved to: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()

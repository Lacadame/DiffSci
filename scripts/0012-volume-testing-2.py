#!/usr/bin/env python
"""
Evaluate physical-property preservation through the chunked encode/decode
roundtrip produced by 0012-volume-testing-1.py.

For each Imperial College stone, compute the same metrics as
0005b-porosity-field-new-metrics-evaluator.py on:
    * the REFERENCE volume (the original 1000^3 raw)
    * the RECONSTRUCTION  ({stone}_recon.npy from script 1)

Metrics computed (per volume):
    porosity, K_abs (x/y/z + mean, lattice & physical),
    Sw, Snw, Pc, kr_wetting, kr_nonwetting (per-direction and mean).

Each comparison saves its own .npz so a slow stone doesn't block the others.
A summary JSON is also written with porosity / K_abs comparison.

By default skips a stone if its results .npz already exists (use --force to
re-run). Cached SNOW2 networks are reused if found alongside each volume.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Make repo importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from poregen.features.snow2 import snow2  # noqa: E402
from diffsci2.extra.pore.permeability_from_pnm import PoreNetworkPermeability  # noqa: E402


# ---------------------------------------------------------------------------
# Stone references
# ---------------------------------------------------------------------------

VOXEL_LENGTHS = {
    'Bentheimer': 3.0035e-6,
    'Doddington': 2.6929e-6,
    'Estaillades': 3.31136e-6,
    'Ketton': 3.00006e-6,
}
DATA_DIR = '/home/ubuntu/repos/DiffSci2/saveddata/raw/imperial_college'
REFERENCE_PATHS = {
    'Bentheimer':  os.path.join(DATA_DIR, 'Bentheimer_1000c_3p0035um.raw'),
    'Doddington':  os.path.join(DATA_DIR, 'Doddington_1000c_2p6929um.raw'),
    'Estaillades': os.path.join(DATA_DIR, 'Estaillades_1000c_3p31136um.raw'),
    'Ketton':      os.path.join(DATA_DIR, 'Ketton_1000c_3p00006um.raw'),
}
RECON_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/stones_reconstruction_test'
VOLUME_SHAPE = (1000, 1000, 1000)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stones', nargs='+', default=list(VOXEL_LENGTHS.keys()),
                   choices=list(VOXEL_LENGTHS.keys()))
    p.add_argument('--recon-dir', default=RECON_DIR)
    p.add_argument('--output-dir', default=RECON_DIR,
                   help='Where to write {stone}_metrics.npz files (default: same as recon-dir).')
    p.add_argument('--border-crop', type=int, default=0,
                   help='Crop this many voxels from each border before evaluation '
                        '(applied to BOTH reference and reconstruction). 0 = no crop.')
    p.add_argument('--contact-angle', type=float, default=140.0)
    p.add_argument('--surface-tension', type=float, default=0.48)
    p.add_argument('--use-cached-network', action='store_true',
                   help='Reuse {volume}.network.npz files if they exist.')
    p.add_argument('--force', action='store_true',
                   help='Re-run even if {stone}_metrics.npz already exists.')
    p.add_argument('--reference-only', action='store_true',
                   help='Skip the reconstruction; only compute reference metrics.')
    p.add_argument('--reconstruction-only', action='store_true',
                   help='Skip the reference; only compute reconstruction metrics.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics (same logic as 0005b)
# ---------------------------------------------------------------------------

def center_crop(volume, border):
    if border <= 0:
        return volume
    return volume[border:-border, border:-border, border:-border]


def extract_porespy_network(binary_volume):
    """SNOW2 expects pore=1, solid=0; our convention is pore=0, solid=1."""
    pore_space = 1 - binary_volume
    return snow2(pore_space, voxel_size=1.0).network


def save_network(network_dict, save_path):
    save_dict = {}
    for key, value in network_dict.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, (int, float, str, bool)):
            save_dict[key] = np.array(value)
        elif isinstance(value, list):
            save_dict[key] = np.array(value)
        else:
            print(f"    Warning: skipping network key '{key}' of type {type(value)}")
    np.savez(save_path, **save_dict)


def get_or_extract_network(binary_volume, network_path, use_cached):
    if use_cached and os.path.exists(network_path):
        print(f"    loading cached network {os.path.basename(network_path)}", flush=True)
        return dict(np.load(network_path))
    print(f"    extracting SNOW2 network...", flush=True)
    t = time.time()
    network = extract_porespy_network(binary_volume)
    print(f"    SNOW2 extraction done in {time.time() - t:.1f}s", flush=True)
    save_network(network, network_path)
    print(f"    saved network to {os.path.basename(network_path)}", flush=True)
    return network


def compute_two_phase_metrics(binary_volume, porespy_network, voxel_size,
                               contact_angle, surface_tension):
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
        'porosity': float(porosity),
        'K_abs_x': abs_perm.K_x, 'K_abs_y': abs_perm.K_y, 'K_abs_z': abs_perm.K_z,
        'K_abs_mean': abs_perm.K_mean,
        'K_abs_x_physical': abs_perm.K_x_physical,
        'K_abs_y_physical': abs_perm.K_y_physical,
        'K_abs_z_physical': abs_perm.K_z_physical,
        'K_abs_mean_physical': abs_perm.K_mean_physical,
        'Sw': rel_perm.Sw, 'Snw': rel_perm.Snwp, 'Pc': rel_perm.Pc,
        'kr_wetting': rel_perm.kr_wetting,
        'kr_nonwetting': rel_perm.kr_nonwetting,
        'kr_wetting_mean': rel_perm.kr_wetting_mean,
        'kr_nonwetting_mean': rel_perm.kr_nonwetting_mean,
    }


# ---------------------------------------------------------------------------
# Per-stone driver
# ---------------------------------------------------------------------------

def load_reference(stone):
    arr = np.fromfile(REFERENCE_PATHS[stone], dtype=np.uint8).reshape(VOLUME_SHAPE)
    return arr  # uint8 in {0, 1}


def load_reconstruction(stone, recon_dir):
    path = os.path.join(recon_dir, f'{stone.lower()}_recon.npy')
    arr = np.load(path)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    return arr


def evaluate_one_volume(label, binary_volume, voxel_length, network_path,
                         use_cached_network, contact_angle, surface_tension):
    print(f"  [{label}] shape {binary_volume.shape}, "
          f"porosity {1.0 - binary_volume.mean():.4f}",
          flush=True)
    network = get_or_extract_network(binary_volume, network_path, use_cached_network)
    print(f"  [{label}] computing two-phase metrics...", flush=True)
    t = time.time()
    metrics = compute_two_phase_metrics(
        binary_volume, network, voxel_length, contact_angle, surface_tension,
    )
    print(f"  [{label}] metrics done in {time.time() - t:.1f}s. "
          f"porosity={metrics['porosity']:.4f}, "
          f"K_abs (mD)={metrics['K_abs_mean_physical']*1e15:.2f}",
          flush=True)
    return metrics


def process_stone(stone, args, summary):
    print(f"\n=== {stone} ===", flush=True)
    voxel_length = VOXEL_LENGTHS[stone]
    print(f"  voxel size: {voxel_length:.4e} m", flush=True)

    out_path = os.path.join(args.output_dir, f'{stone.lower()}_metrics.npz')
    if (not args.force) and os.path.exists(out_path):
        print(f"  {out_path} exists; SKIPPING (pass --force to re-run)", flush=True)
        return

    results = {
        'stone': stone,
        'voxel_length': voxel_length,
        'border_crop': args.border_crop,
        'contact_angle': args.contact_angle,
        'surface_tension': args.surface_tension,
    }
    timing = {}

    # ---- REFERENCE ----
    if not args.reconstruction_only:
        t = time.time()
        ref = load_reference(stone)
        ref = center_crop(ref, args.border_crop)
        ref_net_path = REFERENCE_PATHS[stone].replace('.raw', '.network.npz')
        try:
            ref_metrics = evaluate_one_volume(
                'REFERENCE', ref, voxel_length, ref_net_path,
                args.use_cached_network, args.contact_angle, args.surface_tension,
            )
            for k, v in ref_metrics.items():
                results[f'reference_{k}'] = v
        except Exception as e:
            import traceback
            print(f"  *** REFERENCE ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results['reference_error'] = f"{type(e).__name__}: {e}"
        timing['reference'] = time.time() - t
        del ref

    # ---- RECONSTRUCTION ----
    if not args.reference_only:
        t = time.time()
        try:
            recon = load_reconstruction(stone, args.recon_dir)
            recon = center_crop(recon, args.border_crop)
        except FileNotFoundError as e:
            print(f"  reconstruction not found: {e} — did you run 0012-volume-testing-1.py?",
                  flush=True)
            recon = None

        if recon is not None:
            recon_net_path = os.path.join(
                args.recon_dir, f'{stone.lower()}_recon.network.npz'
            )
            try:
                recon_metrics = evaluate_one_volume(
                    'RECONSTRUCTION', recon, voxel_length, recon_net_path,
                    args.use_cached_network, args.contact_angle, args.surface_tension,
                )
                for k, v in recon_metrics.items():
                    results[f'reconstruction_{k}'] = v
            except Exception as e:
                import traceback
                print(f"  *** RECONSTRUCTION ERROR: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                results['reconstruction_error'] = f"{type(e).__name__}: {e}"
            del recon
        timing['reconstruction'] = time.time() - t

    # ---- Save -----------------------------------------------------------
    results['timing_keys'] = list(timing.keys())
    results['timing_values'] = list(timing.values())
    save_dict = {k: v for k, v in results.items() if v is not None}
    np.savez(out_path, **save_dict)
    print(f"  saved {out_path}", flush=True)

    # quick comparison summary into the json
    cmp = {'voxel_length': voxel_length}
    for key in [
        'porosity',
        'K_abs_mean_physical',
        'K_abs_x_physical', 'K_abs_y_physical', 'K_abs_z_physical',
    ]:
        ref = results.get(f'reference_{key}')
        rec = results.get(f'reconstruction_{key}')
        if ref is not None:
            cmp[f'reference_{key}'] = float(ref)
        if rec is not None:
            cmp[f'reconstruction_{key}'] = float(rec)
        if ref is not None and rec is not None:
            try:
                cmp[f'rel_err_{key}'] = float(abs(rec - ref) / max(abs(ref), 1e-30))
            except Exception:
                pass
    summary[stone] = cmp


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"recon dir: {args.recon_dir}", flush=True)
    print(f"output dir: {args.output_dir}", flush=True)
    print(f"stones: {args.stones}", flush=True)
    print(f"border crop: {args.border_crop}", flush=True)
    print(f"contact angle: {args.contact_angle} deg", flush=True)
    print(f"surface tension: {args.surface_tension} N/m", flush=True)

    summary = {
        'recon_dir': args.recon_dir,
        'output_dir': args.output_dir,
        'border_crop': args.border_crop,
        'contact_angle': args.contact_angle,
        'surface_tension': args.surface_tension,
        'per_stone': {},
    }

    for stone in args.stones:
        try:
            process_stone(stone, args, summary['per_stone'])
        except Exception as e:
            import traceback
            print(f"\n*** stone {stone} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            summary['per_stone'][stone] = {'error': f"{type(e).__name__}: {e}"}
        with open(os.path.join(args.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  partial summary saved\n", flush=True)

    print("\n=== METRICS SUMMARY ===", flush=True)
    for stone, s in summary['per_stone'].items():
        print(f"  {stone}:", flush=True)
        for k, v in s.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4e}", flush=True)
            else:
                print(f"    {k}: {v}", flush=True)
    print(f"\nfull summary: {os.path.join(args.output_dir, 'metrics_summary.json')}",
          flush=True)


if __name__ == '__main__':
    main()

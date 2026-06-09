#!/usr/bin/env python
"""
Oil-water longitudinal flow analysis via pore network modeling.

Runs a complete two-phase flow pipeline on a pre-extracted pore network:
  1. Fine REV sweep along z (absolute K_z only)
  2. Broad REV sweep along z (drainage + kr with trapping)
  3. Full network: abs perm, drainage with trapping, kr, Corey fit, Buckley-Leverett

Physical scenario: oil-wet rock, water (non-wetting) invading oil (wetting)
from zmin, oil escapes from zmax. This is drainage in PNM convention.

Saves all results to a single .npz file alongside the input network.

Usage:
    # Estaillades with default parameters
    python 0011-oil-water-flow.py \
        --network /path/to/estaillades/0.network.npz \
        --voxel-size 3.3116

    # Custom physical parameters
    python 0011-oil-water-flow.py \
        --network /path/to/network.npz \
        --voxel-size 3.0 \
        --contact-angle 140 \
        --surface-tension 0.03 \
        --mu-w 1e-3 --mu-nw 5e-3

    # Control sweep resolution
    python 0011-oil-water-flow.py \
        --network /path/to/network.npz \
        --voxel-size 3.3116 \
        --n-steps-fine 20 \
        --n-steps-broad 10
"""

import argparse
import os
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Oil-water longitudinal flow analysis via pore network modeling.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        '--network', type=str, required=True,
        help='Path to pre-extracted .network.npz file',
    )
    parser.add_argument(
        '--voxel-size', type=float, required=True,
        help='Voxel size in micrometers (e.g. 3.3116)',
    )

    # Physical parameters
    parser.add_argument(
        '--contact-angle', type=float, default=140.0,
        help='Contact angle in degrees, measured through water (default: 140, oil-wet)',
    )
    parser.add_argument(
        '--surface-tension', type=float, default=0.03,
        help='Oil-water interfacial tension in N/m (default: 0.03)',
    )
    parser.add_argument(
        '--mu-w', type=float, default=1e-3,
        help='Water viscosity in Pa.s (default: 1e-3 = 1 cP)',
    )
    parser.add_argument(
        '--mu-nw', type=float, default=5e-3,
        help='Oil viscosity in Pa.s (default: 5e-3 = 5 cP)',
    )

    # Sweep parameters
    parser.add_argument(
        '--n-steps', type=int, default=20,
        help='Number of steps for both REV sweeps (default: 20)',
    )
    parser.add_argument(
        '--boundary-tol', type=float, default=5.0,
        help='Tolerance in voxels for face pore identification (default: 5.0)',
    )

    # Buckley-Leverett reservoir parameters
    parser.add_argument(
        '--bl-length', type=float, default=100.0,
        help='Reservoir length in meters for BL solver (default: 100)',
    )
    parser.add_argument(
        '--bl-area', type=float, default=100.0,
        help='Reservoir cross-section area in m^2 for BL solver (default: 100)',
    )
    parser.add_argument(
        '--bl-rate', type=float, default=10.0,
        help='Injection rate in m^3/day for BL solver (default: 10)',
    )

    # Volume for porosity
    parser.add_argument(
        '--volume', type=str, default=None,
        help='Path to binary volume .npy (0=pore, 1=solid). '
             'Default: 0.npy in the same directory as --network.',
    )

    # Output
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output .npz path (default: <network_dir>/oilwater_results.npz)',
    )

    # Quick re-run mode
    parser.add_argument(
        '--recalculate-porosity-only', action='store_true',
        help='Load existing results .npz, recalculate porosity from volume, '
             'update BL with new porosity, and re-save. Skips all PNM.',
    )

    return parser.parse_args()


def compute_porosity_from_volume(volume_path, crop=128):
    """Load binary volume, crop borders, compute porosity (0=pore, 1=solid)."""
    print(f"Loading volume: {volume_path}")
    vol = np.load(volume_path)
    if hasattr(vol, 'files'):  # .npz
        vol = vol[vol.files[0]]
    c = crop
    if c > 0:
        vol = vol[c:-c, c:-c, c:-c]
    porosity = float((vol == 0).sum() / vol.size)
    print(f"  Shape after crop({c}): {vol.shape}")
    print(f"  Porosity (voxel-based): {porosity:.4f}")
    return porosity


def main():
    args = parse_args()

    voxel_size = args.voxel_size * 1e-6  # um -> m

    # Determine volume path
    if args.volume is None:
        volume_path = os.path.join(
            os.path.dirname(args.network), '0.npy'
        )
    else:
        volume_path = args.volume

    # Determine output path
    if args.output is None:
        output_path = os.path.join(
            os.path.dirname(args.network), 'oilwater_results.npz'
        )
    else:
        output_path = args.output

    # ---- Quick mode: recalculate porosity only ----
    if args.recalculate_porosity_only:
        from diffsci2.extra.pore import CoreyModelParameters, BuckleyLeverettSolver, fit_corey_model
        from diffsci2.extra.pore import RelativePermeabilityResult

        print(f"Recalculating porosity only from {volume_path}")
        print(f"Loading existing results: {output_path}")
        results = dict(np.load(output_path, allow_pickle=True))

        # Load volume and recompute porosity for every sweep step
        vol = np.load(volume_path)
        if hasattr(vol, 'files'):
            vol = vol[vol.files[0]]
        crop = 128

        # Sweep porosity (same z-stops as original run)
        z_range_saved = results['z_range']
        n_steps = int(results.get('broad_n_steps', len(results['fine_z_lengths'])))
        z_stops = np.linspace(float(z_range_saved[0]), float(z_range_saved[1]),
                              n_steps + 1)[1:]
        sweep_porosity = []
        for z_stop in z_stops:
            z_idx = min(int(z_stop), vol.shape[2])
            z_lo = crop
            z_hi = max(z_idx - crop, z_lo + 1)
            sub = vol[crop:-crop, crop:-crop, z_lo:z_hi]
            phi = float((sub == 0).sum() / sub.size) if sub.size > 0 else 0.0
            sweep_porosity.append(phi)
        sweep_porosity = np.array(sweep_porosity)

        # Full-volume porosity
        full_porosity = float((vol[crop:-crop, crop:-crop, crop:-crop] == 0).sum()
                              / vol[crop:-crop, crop:-crop, crop:-crop].size)
        del vol

        old_porosity = float(results.get('full_porosity', 0))
        print(f"  Old full porosity: {old_porosity:.4f} -> New: {full_porosity:.4f}")
        print(f"  Sweep porosity range: [{sweep_porosity.min():.4f}, {sweep_porosity.max():.4f}]")

        results['full_porosity'] = full_porosity
        results['fine_porosity'] = sweep_porosity
        results['broad_porosity'] = sweep_porosity

        # Re-run BL for each broad sweep step with new porosity
        mu_w = float(results.get('mu_w', 1e-3))
        mu_nw = float(results.get('mu_nw', 5e-3))
        bl_rate = float(results.get('bl_rate', 10.0))
        bl_area = float(results.get('bl_area', 100.0))
        bl_length = float(results.get('bl_length', 100.0))

        broad_t_bt = []
        for i in range(n_steps):
            key_sw = f'broad_step_{i}_Sw'
            if key_sw not in results:
                broad_t_bt.append(np.nan)
                continue
            try:
                rp = RelativePermeabilityResult(
                    Sw=results[f'broad_step_{i}_Sw'],
                    Pc=results[f'broad_step_{i}_Pc'],
                    kr_wetting=results[f'broad_step_{i}_kr_wetting'],
                    kr_nonwetting=results[f'broad_step_{i}_kr_nonwetting'],
                )
                corey_i = fit_corey_model(rp, direction='z')
                solver_i = BuckleyLeverettSolver(
                    kr_model=corey_i, mu_w=mu_w, mu_nw=mu_nw,
                    porosity=sweep_porosity[i],
                    total_rate=bl_rate, area=bl_area, length=bl_length,
                )
                bl_i = solver_i.welge_construction()
                broad_t_bt.append(solver_i.breakthrough_time(bl_i))
            except Exception:
                broad_t_bt.append(np.nan)
        results['broad_t_breakthrough'] = np.array(broad_t_bt)
        print(f"  Broad BL updated: {np.sum(np.isfinite(broad_t_bt))}/{n_steps} steps")

        # Re-run full-network BL with new porosity
        corey = CoreyModelParameters(
            Swr=float(results['corey_Swr']),
            Snwr=float(results['corey_Snwr']),
            kr_w0=float(results['corey_kr_w0']),
            kr_nw0=float(results['corey_kr_nw0']),
            nw=float(results['corey_nw']),
            nnw=float(results['corey_nnw']),
        )
        solver = BuckleyLeverettSolver(
            kr_model=corey, mu_w=mu_w, mu_nw=mu_nw,
            porosity=full_porosity,
            total_rate=bl_rate, area=bl_area, length=bl_length,
        )
        bl_result = solver.welge_construction()
        t_bt = solver.breakthrough_time(bl_result)
        results['bl_Sw_shock'] = bl_result.Sw_shock
        results['bl_fw_shock'] = bl_result.fw_shock
        results['bl_t_breakthrough'] = t_bt
        print(f"  Full BL updated: t_bt = {t_bt:.2f} days")

        # Re-save BL profiles
        Sw_curve = np.linspace(corey.Swr, 1.0 - corey.Snwr, 500)
        results['bl_Sw_curve'] = Sw_curve
        results['bl_fw_curve'] = corey.fractional_flow(Sw_curve, mu_w, mu_nw)
        x_pos = np.linspace(0, bl_length, 500)
        results['bl_x_pos'] = x_pos
        time_fracs = np.array([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
        results['bl_time_fracs'] = time_fracs
        profiles = []
        for frac in time_fracs:
            t = frac * t_bt
            profiles.append(solver.saturation_profile(x_pos, t, bl_result))
        results['bl_profiles'] = np.array(profiles)
        t_recovery = np.linspace(0.01, 3.0 * t_bt, 200)
        results['bl_t_recovery'] = t_recovery
        results['bl_recovery'] = np.array(
            [solver.oil_recovery(t, bl_result) for t in t_recovery]
        )

        np.savez(output_path, **results)
        print(f"Saved to {output_path}")
        return

    # ---- Full run ----
    # Lazy imports (heavy)
    from diffsci2.extra.pore import (
        PoreNetworkPermeability,
        linear_rev_sweep,
        fit_corey_model,
        BuckleyLeverettSolver,
    )

    print(f"Network: {args.network}")
    print(f"Voxel size: {args.voxel_size} um")
    print(f"Contact angle: {args.contact_angle} deg")
    print(f"Surface tension: {args.surface_tension} N/m")
    print(f"mu_w: {args.mu_w} Pa.s, mu_nw: {args.mu_nw} Pa.s")
    print(f"REV sweep: {args.n_steps} steps (both single-phase and two-phase)")
    print(f"Output: {output_path}")
    print()

    # ---- Load network ----
    t_total = time.time()
    net = np.load(args.network)
    net_dict = dict(net)

    coords = net['pore.coords']
    x_range = (float(coords[:, 0].min()), float(coords[:, 0].max()))
    y_range = (float(coords[:, 1].min()), float(coords[:, 1].max()))
    z_range = (float(coords[:, 2].min()), float(coords[:, 2].max()))
    volume_shape = (
        int(np.ceil(x_range[1])),
        int(np.ceil(y_range[1])),
        int(np.ceil(z_range[1])),
    )

    print(f"Pores: {len(coords):,}, Throats: {len(net['throat.conns']):,}")
    print(f"z range: [{z_range[0]:.0f}, {z_range[1]:.0f}] voxels")
    print()

    results = {}  # will be saved to npz

    # Store metadata
    results['voxel_size'] = voxel_size
    results['contact_angle'] = args.contact_angle
    results['surface_tension'] = args.surface_tension
    results['mu_w'] = args.mu_w
    results['mu_nw'] = args.mu_nw
    results['volume_shape'] = np.array(volume_shape)
    results['x_range'] = np.array(x_range)
    results['y_range'] = np.array(y_range)
    results['z_range'] = np.array(z_range)
    results['n_pores_total'] = len(coords)
    results['n_throats_total'] = len(net['throat.conns'])

    # ---- Precompute volume-based porosity for every sweep step ----
    print("Loading binary volume for porosity calculation...")
    vol = np.load(volume_path)
    if hasattr(vol, 'files'):
        vol = vol[vol.files[0]]
    print(f"  Volume shape: {vol.shape}")

    crop = 128
    z_stops = np.linspace(z_range[0], z_range[1], args.n_steps + 1)[1:]
    sweep_porosity = []
    for z_stop in z_stops:
        z_idx = min(int(z_stop), vol.shape[2])
        z_lo = crop
        z_hi = max(z_idx - crop, z_lo + 1)
        sub = vol[crop:-crop, crop:-crop, z_lo:z_hi]
        phi = float((sub == 0).sum() / sub.size) if sub.size > 0 else 0.0
        sweep_porosity.append(phi)
    sweep_porosity = np.array(sweep_porosity)
    print(f"  Precomputed porosity for {len(sweep_porosity)} sweep steps "
          f"(crop={crop}, range [{sweep_porosity.min():.4f}, {sweep_porosity.max():.4f}])")

    # Full-volume porosity (cropped)
    full_porosity = float((vol[crop:-crop, crop:-crop, crop:-crop] == 0).sum()
                          / vol[crop:-crop, crop:-crop, crop:-crop].size)
    print(f"  Full-volume porosity (cropped): {full_porosity:.4f}")
    del vol  # free memory

    # ================================================================
    # 1. Fine K_z REV sweep (absolute permeability only)
    # ================================================================
    print("=" * 60)
    print(f"PHASE 1: K_z REV sweep (absolute only, {args.n_steps} steps)")
    print("=" * 60)
    t0 = time.time()

    rev_fine = linear_rev_sweep(
        network_dict=net_dict,
        volume_shape=volume_shape,
        sweep_axis=2,
        start=z_range[0],
        end=z_range[1],
        n_steps=args.n_steps,
        voxel_size=voxel_size,
        compute_two_phase=False,
        boundary_tol=args.boundary_tol,
    )

    results['fine_z_lengths'] = np.array([r.volume_dims[2] for r in rev_fine])
    results['fine_Kz'] = np.array([r.abs_perm.K_z for r in rev_fine])
    results['fine_Kz_physical'] = np.array([r.abs_perm.K_z_physical for r in rev_fine])
    results['fine_porosity'] = sweep_porosity  # from binary volume, not network
    results['fine_n_pores'] = np.array([r.n_pores for r in rev_fine])
    results['fine_elapsed_s'] = time.time() - t0

    print(f"\nFine sweep done in {results['fine_elapsed_s']:.1f}s")

    # ================================================================
    # 2. Broad two-phase REV sweep (with trapping)
    # ================================================================
    print()
    print("=" * 60)
    print(f"PHASE 2: Two-phase REV sweep (with trapping, {args.n_steps} steps)")
    print("=" * 60)
    t0 = time.time()

    rev_broad = linear_rev_sweep(
        network_dict=net_dict,
        volume_shape=volume_shape,
        sweep_axis=2,
        start=z_range[0],
        end=z_range[1],
        n_steps=args.n_steps,
        voxel_size=voxel_size,
        contact_angle=args.contact_angle,
        surface_tension=args.surface_tension,
        compute_two_phase=True,
        boundary_tol=args.boundary_tol,
        inlet_face='zmin',
        trapping=True,
    )

    broad_z_lengths = []
    broad_Kz_physical = []
    broad_porosity = []
    broad_n_pores = []
    broad_Sw_min = []
    broad_t_breakthrough = []

    for i, r in enumerate(rev_broad):
        broad_z_lengths.append(r.volume_dims[2])
        broad_Kz_physical.append(r.abs_perm.K_z_physical)
        broad_porosity.append(sweep_porosity[i])  # from binary volume
        broad_n_pores.append(r.n_pores)

        if r.rel_perm is not None:
            broad_Sw_min.append(r.rel_perm.Sw.min())
            results[f'broad_step_{i}_Sw'] = r.rel_perm.Sw
            results[f'broad_step_{i}_Pc'] = r.rel_perm.Pc
            results[f'broad_step_{i}_kr_wetting'] = r.rel_perm.kr_wetting
            results[f'broad_step_{i}_kr_nonwetting'] = r.rel_perm.kr_nonwetting

            # Fit Corey + solve BL for breakthrough time at this sub-volume
            try:
                corey_i = fit_corey_model(r.rel_perm, direction='z')
                solver_i = BuckleyLeverettSolver(
                    kr_model=corey_i,
                    mu_w=args.mu_w,
                    mu_nw=args.mu_nw,
                    porosity=sweep_porosity[i],  # from binary volume
                    total_rate=args.bl_rate,
                    area=args.bl_area,
                    length=args.bl_length,
                )
                bl_i = solver_i.welge_construction()
                t_bt_i = solver_i.breakthrough_time(bl_i)
                broad_t_breakthrough.append(t_bt_i)
                print(f"    BL: Sor={r.rel_perm.Sw.min():.4f}, "
                      f"t_bt={t_bt_i:.2f} days")
            except Exception as e:
                broad_t_breakthrough.append(np.nan)
                print(f"    BL fit failed: {e}")
        else:
            broad_Sw_min.append(np.nan)
            broad_t_breakthrough.append(np.nan)

    results['broad_z_lengths'] = np.array(broad_z_lengths)
    results['broad_Kz_physical'] = np.array(broad_Kz_physical)
    results['broad_porosity'] = np.array(broad_porosity)
    results['broad_n_pores'] = np.array(broad_n_pores)
    results['broad_Sw_min'] = np.array(broad_Sw_min)
    # In an oil-wet system, wetting phase = oil.  Sw_min from PNM is the
    # minimum wetting-phase (oil) saturation, i.e. Sor directly.
    results['broad_Sor'] = np.array(broad_Sw_min)
    results['broad_t_breakthrough'] = np.array(broad_t_breakthrough)
    results['broad_n_steps'] = args.n_steps
    results['broad_elapsed_s'] = time.time() - t0

    print(f"\nBroad sweep done in {results['broad_elapsed_s']:.1f}s")

    # ================================================================
    # 3. Full network analysis
    # ================================================================
    print()
    print("=" * 60)
    print("PHASE 3: Full network analysis")
    print("=" * 60)
    t0 = time.time()

    pnp = PoreNetworkPermeability.from_binary_volume(
        network=net,
        voxel_size=voxel_size,
    )
    print(f"Network loaded: {pnp}")

    # Absolute permeability (all 3 directions)
    abs_perm = pnp.calculate_absolute_permeability()
    results['full_Kx_physical'] = abs_perm.K_x_physical
    results['full_Ky_physical'] = abs_perm.K_y_physical
    results['full_Kz_physical'] = abs_perm.K_z_physical
    results['full_Kmean_physical'] = abs_perm.K_mean_physical
    print(f"K_x={abs_perm.K_x_physical*1e15:.3f} mD, "
          f"K_y={abs_perm.K_y_physical*1e15:.3f} mD, "
          f"K_z={abs_perm.K_z_physical*1e15:.3f} mD")

    # Drainage with trapping
    Pc_values = pnp.run_drainage_simulation(
        inlet_face='zmin',
        contact_angle=args.contact_angle,
        surface_tension=args.surface_tension,
        trapping=True,
        outlet_face='zmax',
    )
    print(f"Drainage: {len(Pc_values)} Pc steps")

    # Relative permeability (z-direction only)
    rel_perm = pnp.calculate_relative_permeability_curves(directions=['z'])
    results['full_Sw'] = rel_perm.Sw
    results['full_Pc'] = rel_perm.Pc
    results['full_kr_wetting'] = rel_perm.kr_wetting
    results['full_kr_nonwetting'] = rel_perm.kr_nonwetting
    results['full_Sw_min'] = rel_perm.Sw.min()
    # Sw_min = residual wetting phase (oil) saturation = Sor
    results['full_Sor'] = rel_perm.Sw.min()
    print(f"Sw range: [{rel_perm.Sw.min():.4f}, {rel_perm.Sw.max():.4f}], "
          f"Sor={results['full_Sor']:.4f}")

    # Porosity from binary volume (precomputed above)
    porosity_net = full_porosity
    results['full_porosity'] = porosity_net

    # Corey model fit
    corey = fit_corey_model(rel_perm, direction='z')
    results['corey_Swr'] = corey.Swr
    results['corey_Snwr'] = corey.Snwr
    results['corey_nw'] = corey.nw
    results['corey_nnw'] = corey.nnw
    results['corey_kr_w0'] = corey.kr_w0
    results['corey_kr_nw0'] = corey.kr_nw0
    print(f"Corey: Swr={corey.Swr:.4f}, Snwr={corey.Snwr:.4f}, "
          f"nw={corey.nw:.3f}, nnw={corey.nnw:.3f}")

    # Buckley-Leverett
    solver = BuckleyLeverettSolver(
        kr_model=corey,
        mu_w=args.mu_w,
        mu_nw=args.mu_nw,
        porosity=porosity_net,
        total_rate=args.bl_rate,
        area=args.bl_area,
        length=args.bl_length,
    )
    bl_result = solver.welge_construction()
    t_bt = solver.breakthrough_time(bl_result)

    results['bl_Sw_shock'] = bl_result.Sw_shock
    results['bl_fw_shock'] = bl_result.fw_shock
    results['bl_t_breakthrough'] = t_bt
    results['bl_length'] = args.bl_length
    results['bl_area'] = args.bl_area
    results['bl_rate'] = args.bl_rate

    # Save BL profiles for plotting
    Sw_curve = np.linspace(corey.Swr, 1.0 - corey.Snwr, 500)
    results['bl_Sw_curve'] = Sw_curve
    results['bl_fw_curve'] = corey.fractional_flow(Sw_curve, args.mu_w, args.mu_nw)

    x_pos = np.linspace(0, args.bl_length, 500)
    results['bl_x_pos'] = x_pos

    time_fracs = np.array([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
    results['bl_time_fracs'] = time_fracs
    profiles = []
    for frac in time_fracs:
        t = frac * t_bt
        profiles.append(solver.saturation_profile(x_pos, t, bl_result))
    results['bl_profiles'] = np.array(profiles)

    t_recovery = np.linspace(0.01, 3.0 * t_bt, 200)
    results['bl_t_recovery'] = t_recovery
    results['bl_recovery'] = np.array(
        [solver.oil_recovery(t, bl_result) for t in t_recovery]
    )

    results['full_elapsed_s'] = time.time() - t0
    print(f"\nFull analysis done in {results['full_elapsed_s']:.1f}s")

    # ---- Save ----
    total_elapsed = time.time() - t_total
    results['total_elapsed_s'] = total_elapsed

    np.savez(output_path, **results)
    fsize_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved to {output_path} ({fsize_mb:.1f} MB)")
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == '__main__':
    main()

"""
Sub-network extraction and property computation for pore networks.

Given a pre-computed large pore network (from SNOW2), extract sub-networks
by spatial bounding box and compute flow properties on them. This avoids
re-running the expensive SNOW2 segmentation for each sub-volume, enabling
efficient REV (Representative Elementary Volume) analysis.

The core idea:
    1. Load a large network once (from .network.npz)
    2. For each sub-volume of interest, trim the network spatially
    3. Re-label boundaries on the trimmed sub-network
    4. Compute properties (porosity, permeability, drainage, kr)

For absolute permeability, the Darcy L/A are computed per-direction to
handle non-cubic sub-volumes correctly.

For relative permeability, we reuse PoreNetworkPermeability internally.
Since kr = K_eff / K_abs and both use the same L/A, the ratio is correct
regardless of the geometry parameter passed to PoreNetworkPermeability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time
import warnings

import numpy as np
import openpnm

from .permeability_from_pnm import (
    AbsolutePermeabilityResult,
    RelativePermeabilityResult,
    PoreNetworkPermeability,
)


@dataclass
class SubnetworkResult:
    """Container for sub-network property computation results."""
    bounds: tuple
    volume_dims: tuple
    n_pores: int
    n_throats: int
    porosity: float
    abs_perm: AbsolutePermeabilityResult
    rel_perm: Optional[RelativePermeabilityResult] = None
    elapsed_s: float = 0.0


def extract_subnetwork_properties(
    network_dict: dict,
    volume_shape: tuple[int, int, int],
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    voxel_size: float = 1.0,
    contact_angle: float = 140.0,
    surface_tension: float = 0.48,
    compute_two_phase: bool = True,
    boundary_tol: float = 5.0,
    inlet_face: str = 'xmin',
    trapping: bool = False,
) -> SubnetworkResult:
    """
    Extract a sub-network by spatial bounds and compute flow properties.

    Parameters
    ----------
    network_dict : dict
        PoreSpy-style network dictionary (e.g. from ``np.load('x.network.npz')``).
        Also accepts NpzFile objects directly.
    volume_shape : (int, int, int)
        Shape of the original segmented volume in voxels (Nx, Ny, Nz).
        Kept as reference metadata; actual geometry derives from bounds.
    bounds : ((lbx, ubx), (lby, uby), (lbz, ubz))
        Coordinate bounds defining the sub-volume, in the same coordinate
        system as ``pore.coords`` in the network.
    voxel_size : float
        Physical voxel size in meters.
    contact_angle : float
        Contact angle for drainage simulation [degrees].
    surface_tension : float
        Surface tension for drainage simulation [N/m].
    compute_two_phase : bool
        If True, also run drainage and compute relative permeability curves.
    boundary_tol : float
        Tolerance in voxels for identifying face pores after trimming.
        Pores within this distance of the sub-network coordinate extremum
        are labeled as face pores and receive boundary conditions.
    inlet_face : str
        Face from which the non-wetting phase invades during drainage.
    trapping : bool
        If True, apply trapping during drainage (defending phase clusters
        disconnected from the outlet face are marked as trapped).

    Returns
    -------
    SubnetworkResult
        Contains porosity, absolute permeability (correct for non-cubic
        geometry), and optionally relative permeability curves.
    """
    t0 = time.time()

    # Handle NpzFile input
    if hasattr(network_dict, 'files'):
        network_dict = dict(network_dict)

    (lbx, ubx), (lby, uby), (lbz, ubz) = bounds
    Lx, Ly, Lz = ubx - lbx, uby - lby, ubz - lbz

    # ---- Build and trim the OpenPNM network ----
    net_copy = {k: v.copy() for k, v in network_dict.items()}
    pn = openpnm.io.network_from_porespy(net_copy)

    coords = pn['pore.coords']
    keep = (
        (coords[:, 0] >= lbx) & (coords[:, 0] <= ubx) &
        (coords[:, 1] >= lby) & (coords[:, 1] <= uby) &
        (coords[:, 2] >= lbz) & (coords[:, 2] <= ubz)
    )
    to_remove = np.where(~keep)[0]
    if len(to_remove) > 0:
        openpnm.topotools.trim(pn, pores=to_remove)

    health = openpnm.utils.check_network_health(pn)
    if len(health['disconnected_pores']) > 0:
        openpnm.topotools.trim(pn, pores=health['disconnected_pores'])

    if pn.Np == 0:
        return SubnetworkResult(
            bounds=bounds, volume_dims=(Lx, Ly, Lz),
            n_pores=0, n_throats=0, porosity=0.0,
            abs_perm=AbsolutePermeabilityResult(0, 0, 0, voxel_size),
            elapsed_s=time.time() - t0,
        )

    # ---- Re-label face pores for the sub-volume ----
    coords = pn['pore.coords']
    for ax_i, ax in enumerate('xyz'):
        c = coords[:, ax_i]
        pn[f'pore.{ax}min'] = c <= c.min() + boundary_tol
        pn[f'pore.{ax}max'] = c >= c.max() - boundary_tol

    # Map PoreSpy names -> OpenPNM names (diameter, spacing, radius)
    PoreNetworkPermeability._setup_network_geometry(pn)

    # ---- Porosity (network-derived) ----
    porosity = pn['pore.volume'].sum() / (Lx * Ly * Lz)

    # ---- Absolute permeability (non-cubic Darcy geometry) ----
    g_base = np.pi * (pn['throat.radius'] ** 4) / (8.0 * pn['throat.spacing'])
    pn['throat.hydraulic_conductance'] = g_base
    phase = openpnm.phase.Phase(network=pn)

    dims_LA = [(Lx, Ly * Lz), (Ly, Lx * Lz), (Lz, Lx * Ly)]
    K_vals = [0.0, 0.0, 0.0]

    for ax_i, ax in enumerate('xyz'):
        L_ax, A_ax = dims_LA[ax_i]
        inlet = pn.pores(f'{ax}min')
        outlet = pn.pores(f'{ax}max')
        if len(inlet) == 0 or len(outlet) == 0:
            continue
        try:
            sf = openpnm.algorithms.StokesFlow(network=pn, phase=phase)
            sf.set_value_BC(pores=inlet, values=1.0)
            sf.set_value_BC(pores=outlet, values=0.0)
            sf.run()
            Q = sf.rate(pores=inlet)[0]
            K_vals[ax_i] = Q * (L_ax / A_ax)
        except Exception as e:
            warnings.warn(f"StokesFlow failed for {ax}-direction: {e}")

    abs_perm = AbsolutePermeabilityResult(
        K_x=K_vals[0], K_y=K_vals[1], K_z=K_vals[2],
        voxel_size=voxel_size,
    )

    # ---- Two-phase (drainage + relative permeability) ----
    rel_perm = None
    if compute_two_phase:
        try:
            pnp = PoreNetworkPermeability.from_openpnm_network(
                pn,
                volume_length=int(max(Lx, Ly, Lz)),
                voxel_size=voxel_size,
                trim_disconnected=False,
                setup_geometry=False,
                volume_dims=(Lx, Ly, Lz),
            )
            pnp.calculate_absolute_permeability()
            pnp.run_drainage_simulation(
                inlet_face=inlet_face,
                contact_angle=contact_angle,
                surface_tension=surface_tension,
                trapping=trapping,
            )
            rel_perm = pnp.calculate_relative_permeability_curves()
        except Exception as e:
            warnings.warn(f"Two-phase computation failed: {e}")

    return SubnetworkResult(
        bounds=bounds,
        volume_dims=(Lx, Ly, Lz),
        n_pores=pn.Np,
        n_throats=pn.Nt,
        porosity=porosity,
        abs_perm=abs_perm,
        rel_perm=rel_perm,
        elapsed_s=time.time() - t0,
    )


def linear_rev_sweep(
    network_dict: dict,
    volume_shape: tuple[int, int, int],
    sweep_axis: int,
    start: float,
    end: float,
    n_steps: int,
    voxel_size: float = 1.0,
    contact_angle: float = 140.0,
    surface_tension: float = 0.48,
    compute_two_phase: bool = True,
    boundary_tol: float = 5.0,
    inlet_face: str = 'xmin',
    trapping: bool = False,
) -> list[SubnetworkResult]:
    """
    Compute flow properties on increasing sub-volumes along one axis.

    One face is fixed at ``start``; the opposite face grows from
    ``start + delta`` to ``end`` in ``n_steps`` equal increments.
    The other two axes are kept at the full network extent.

    Parameters
    ----------
    network_dict : dict
        PoreSpy-style network dictionary (or NpzFile).
    volume_shape : (int, int, int)
        Shape of the original segmented volume in voxels (Nx, Ny, Nz).
    sweep_axis : int
        Axis to sweep: 0 = x, 1 = y, 2 = z.
    start : float
        Fixed face coordinate along the sweep axis.
    end : float
        Maximum coordinate the growing face reaches.
    n_steps : int
        Number of sub-volume sizes to evaluate.
    voxel_size : float
        Physical voxel size [m].
    contact_angle : float
        Contact angle for drainage [degrees].
    surface_tension : float
        Surface tension for drainage [N/m].
    compute_two_phase : bool
        If True, also compute drainage + relative permeability.
    boundary_tol : float
        Tolerance for face pore identification [voxels].
    inlet_face : str
        Face from which the non-wetting phase invades during drainage.
    trapping : bool
        If True, apply trapping during drainage.

    Returns
    -------
    list[SubnetworkResult]
        One result per step, ordered by increasing sub-volume size.
    """
    # Handle NpzFile input
    if hasattr(network_dict, 'files'):
        network_dict = dict(network_dict)

    coords = network_dict['pore.coords']
    full_bounds = [
        (float(coords[:, i].min()), float(coords[:, i].max()))
        for i in range(3)
    ]

    axis_name = 'xyz'[sweep_axis]
    stops = np.linspace(start, end, n_steps + 1)[1:]

    print(f"Linear REV sweep: axis={axis_name}, "
          f"range=[{start:.0f}, {end:.0f}], steps={n_steps}")
    print(f"Full network extent: "
          f"x=[{full_bounds[0][0]:.0f},{full_bounds[0][1]:.0f}], "
          f"y=[{full_bounds[1][0]:.0f},{full_bounds[1][1]:.0f}], "
          f"z=[{full_bounds[2][0]:.0f},{full_bounds[2][1]:.0f}]")
    print(f"Network size: {len(coords)} pores")

    results = []
    for step_i, stop in enumerate(stops):
        bounds_list = list(full_bounds)
        bounds_list[sweep_axis] = (start, float(stop))
        bounds = tuple(tuple(b) for b in bounds_list)

        sub_dim = stop - start
        print(f"\n  [{step_i + 1}/{n_steps}] {axis_name}: "
              f"[{start:.0f}, {stop:.0f}] (L={sub_dim:.0f} voxels)")

        result = extract_subnetwork_properties(
            network_dict=network_dict,
            volume_shape=volume_shape,
            bounds=bounds,
            voxel_size=voxel_size,
            contact_angle=contact_angle,
            surface_tension=surface_tension,
            compute_two_phase=compute_two_phase,
            boundary_tol=boundary_tol,
            inlet_face=inlet_face,
            trapping=trapping,
        )
        results.append(result)

        K_mD = result.abs_perm.K_mean_physical * 1e15
        print(f"    pores={result.n_pores}, throats={result.n_throats}, "
              f"phi={result.porosity:.4f}, K_mean={K_mD:.2f} mD, "
              f"t={result.elapsed_s:.1f}s")
        if result.rel_perm is not None:
            print(f"    Sw range: [{result.rel_perm.Sw.min():.3f}, "
                  f"{result.rel_perm.Sw.max():.3f}]")

    total_t = sum(r.elapsed_s for r in results)
    print(f"\nSweep complete: {n_steps} steps in {total_t:.1f}s total")

    return results

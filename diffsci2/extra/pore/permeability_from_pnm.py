"""
Pore Network Permeability Calculator
=====================================

This module provides a class for computing single-phase and two-phase flow
properties from pore network models using OpenPNM.

The typical workflow is:
    1. Create a PoreNetworkPermeability instance from a binary volume or network
    2. Calculate absolute (single-phase) permeability
    3. Run drainage simulation to invade non-wetting phase
    4. Calculate relative permeabilities at each saturation state

Example usage:
    >>> from diffsci2.extra.pore import PoreNetworkPermeability
    >>>
    >>> # From a binary volume (pore space = 0, solid = 1)
    >>> pnp = PoreNetworkPermeability.from_binary_volume(binary_volume, voxel_size=3e-6)
    >>>
    >>> # Calculate absolute permeability
    >>> K_abs = pnp.calculate_absolute_permeability()
    >>>
    >>> # Run drainage and get relative permeability curves
    >>> results = pnp.run_drainage_simulation()
    >>> rel_perm = pnp.calculate_relative_permeability_curves()
/home/ubuntu/repos/DiffSci2/diffsci2/extra/pore
Author: Generated from notebook 0014-two-phase-flow-network-metrics.ipynb
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Sequence
import warnings

import numpy as np
import openpnm


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class AbsolutePermeabilityResult:
    """
    Container for absolute permeability results.

    Stores both dimensionless values (for internal calculations like kr)
    and physical values in m^2.

    Attributes:
        K_x: Permeability in x-direction [dimensionless]
        K_y: Permeability in y-direction [dimensionless]
        K_z: Permeability in z-direction [dimensionless]
        voxel_size: Physical size of each voxel [m], used for unit conversion
    """
    K_x: float
    K_y: float
    K_z: float
    voxel_size: float = 1.0

    @property
    def K_mean(self) -> float:
        """Mean dimensionless permeability across all three directions."""
        return np.mean([self.K_x, self.K_y, self.K_z])

    @property
    def K_array(self) -> np.ndarray:
        """Dimensionless permeability as array [K_x, K_y, K_z]."""
        return np.array([self.K_x, self.K_y, self.K_z])

    # Physical units (m^2)
    @property
    def K_x_physical(self) -> float:
        """Permeability in x-direction [m^2]."""
        return self.K_x * (self.voxel_size ** 2)

    @property
    def K_y_physical(self) -> float:
        """Permeability in y-direction [m^2]."""
        return self.K_y * (self.voxel_size ** 2)

    @property
    def K_z_physical(self) -> float:
        """Permeability in z-direction [m^2]."""
        return self.K_z * (self.voxel_size ** 2)

    @property
    def K_mean_physical(self) -> float:
        """Mean permeability in physical units [m^2]."""
        return self.K_mean * (self.voxel_size ** 2)

    @property
    def K_array_physical(self) -> np.ndarray:
        """Physical permeability as array [K_x, K_y, K_z] in m^2."""
        return self.K_array * (self.voxel_size ** 2)


@dataclass
class RelativePermeabilityResult:
    """
    Container for relative permeability results at multiple saturation states.

    This dataclass contains both the drainage curve (Pc vs Sw) and the
    relative permeability curves (kr vs Sw) computed step-by-step from
    invaded pore volumes.

    Attributes:
        Sw: Array of wetting phase saturations (computed from pore volumes)
        Pc: Array of capillary pressures [Pa]
        kr_wetting: Array of shape (n_saturations, 3) for kr_w in [x, y, z]
        kr_nonwetting: Array of shape (n_saturations, 3) for kr_nw in [x, y, z]
    """
    Sw: np.ndarray
    Pc: np.ndarray
    kr_wetting: np.ndarray      # shape: (n_saturations, 3) for x, y, z
    kr_nonwetting: np.ndarray   # shape: (n_saturations, 3) for x, y, z

    @property
    def Snwp(self) -> np.ndarray:
        """Non-wetting phase saturation (1 - Sw)."""
        return 1.0 - self.Sw

    @property
    def kr_wetting_mean(self) -> np.ndarray:
        """Mean wetting phase relative permeability across directions."""
        return np.mean(self.kr_wetting, axis=1)

    @property
    def kr_nonwetting_mean(self) -> np.ndarray:
        """Mean non-wetting phase relative permeability across directions."""
        return np.mean(self.kr_nonwetting, axis=1)

    # --- Interpolation methods ---

    def _make_interpolator(self, x: np.ndarray, y: np.ndarray):
        """Build a linear interpolator, sorting by x first."""
        from scipy.interpolate import interp1d
        order = np.argsort(x)
        return interp1d(
            x[order], y[order],
            kind='linear', bounds_error=False, fill_value=np.nan,
        )

    def Sw_from_Pc(self, Pc_query):
        """Interpolate wetting saturation from capillary pressure."""
        return self._make_interpolator(self.Pc, self.Sw)(np.asarray(Pc_query))

    def Pc_from_Sw(self, Sw_query):
        """Interpolate capillary pressure from wetting saturation."""
        return self._make_interpolator(self.Sw, self.Pc)(np.asarray(Sw_query))

    def _kr_column(self, kr_array: np.ndarray, direction: str) -> np.ndarray:
        """Select a kr column by direction name."""
        if direction == 'mean':
            return np.mean(kr_array, axis=1)
        idx = {'x': 0, 'y': 1, 'z': 2}[direction]
        return kr_array[:, idx]

    def kr_wetting_from_Sw(self, Sw_query, direction: str = 'mean'):
        """Interpolate wetting relative permeability from saturation."""
        y = self._kr_column(self.kr_wetting, direction)
        return self._make_interpolator(self.Sw, y)(np.asarray(Sw_query))

    def kr_nonwetting_from_Sw(self, Sw_query, direction: str = 'mean'):
        """Interpolate non-wetting relative permeability from saturation."""
        y = self._kr_column(self.kr_nonwetting, direction)
        return self._make_interpolator(self.Sw, y)(np.asarray(Sw_query))


# =============================================================================
# Main Class
# =============================================================================

class PoreNetworkPermeability:
    """
    A class for computing permeability and relative permeability from pore networks.

    This class wraps OpenPNM functionality to provide a streamlined workflow for:
    - Single-phase (absolute) permeability calculation
    - Two-phase drainage simulation
    - Relative permeability curve generation

    The network can be created from:
    - A 3D binary volume (using SNOW algorithm via poregen)
    - An existing OpenPNM network object
    - A PoreSpy network dictionary (from SNOW or other extraction methods)

    Physical Parameters:
        The class uses dimensionless calculations internally. Physical units
        are only needed for capillary pressure calculations, which require
        the voxel_size (physical length per voxel).

    Attributes:
        pn: The OpenPNM network object
        volume_length: The characteristic length of the sample (voxels)
        voxel_size: Physical size of each voxel [m]
        _abs_perm_result: Cached absolute permeability result
        _drainage_algorithm: The drainage algorithm after running simulation
        _g_base: Base hydraulic conductance array (for reuse in rel perm)
    """

    def __init__(
        self,
        network: openpnm.network.Network,
        volume_length: int,
        voxel_size: float = 1.0,
        volume_dims: Optional[tuple] = None,
    ):
        """
        Initialize with an existing OpenPNM network.

        For most use cases, prefer the factory methods:
        - from_binary_volume()
        - from_porespy_network()

        Args:
            network: An OpenPNM Network object (already processed)
            volume_length: The edge length of the cubic sample in voxels.
                           Used as fallback when volume_dims is not given.
            voxel_size: Physical size of each voxel in meters (default=1.0 for
                        dimensionless calculations)
            volume_dims: Tuple (Lx, Ly, Lz) giving the extent of the sample
                         in each direction (in voxels). When provided, this is
                         used for per-direction L/A in Darcy's law instead of
                         assuming a cube with side volume_length.
        """
        self.pn = network
        self.volume_length = volume_length
        self.voxel_size = voxel_size
        if volume_dims is not None:
            self.volume_dims = tuple(float(d) for d in volume_dims)
        else:
            self.volume_dims = (float(volume_length),) * 3

        # Internal state (populated by various methods)
        self._abs_perm_result: Optional[AbsolutePermeabilityResult] = None
        self._drainage_algorithm = None
        self._g_base: Optional[np.ndarray] = None

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_binary_volume(
        cls,
        binary_volume: Optional[np.ndarray] = None,
        voxel_size: float = 1.0,
        trim_disconnected: bool = True,
        network=None,
        volume_length: Optional[int] = None,
    ) -> "PoreNetworkPermeability":
        """
        Create a PoreNetworkPermeability from a 3D binary volume or a
        pre-extracted network.

        If ``network`` is provided (e.g. loaded from a ``.npz`` file), the
        binary volume is not needed and SNOW extraction is skipped entirely.

        Uses the SNOW algorithm (via poregen.features.snow2) to extract the
        pore network from the binary volume when no network is given.

        Args:
            binary_volume: 3D numpy array where 0 = pore space, 1 = solid.
                           Shape should be (Lx, Ly, Lz). Not required when
                           ``network`` is provided.
            voxel_size: Physical size of each voxel in meters.
            trim_disconnected: If True, remove pores not connected to the
                               main network (recommended).
            network: A pre-extracted PoreSpy network dict (or NpzFile from
                     ``np.load``). When provided, the volume is not needed.
            volume_length: The edge length of the cubic sample in voxels.
                           Inferred from ``binary_volume.shape[0]`` when a
                           volume is given, or from ``pore.coords`` bounding
                           box when only a network is given. Can always be
                           overridden explicitly.

        Returns:
            A new PoreNetworkPermeability instance.

        Raises:
            ImportError: If poregen is not available (volume path only).
            ValueError: If neither binary_volume nor network is provided, or
                        if volume_length cannot be determined.
        """
        if network is not None:
            # Convert NpzFile (from np.load) to a plain dict
            if hasattr(network, 'files'):
                network_dict = dict(network)
            else:
                network_dict = network

            # Determine volume_length and volume_dims
            volume_dims = None
            if volume_length is not None:
                pass  # explicitly given, use as-is
            elif binary_volume is not None:
                volume_length = binary_volume.shape[0]
                volume_dims = tuple(float(s) for s in binary_volume.shape)
            elif 'pore.coords' in network_dict:
                coords = network_dict['pore.coords']
                volume_length = int(np.ceil(coords.max()))
                # Infer per-axis extents from coordinate bounding box
                volume_dims = tuple(
                    float(np.ceil(coords[:, i].max()))
                    for i in range(3)
                )
            else:
                raise ValueError(
                    "Cannot determine volume_length. Provide it explicitly, "
                    "pass binary_volume, or ensure network has 'pore.coords'."
                )

            return cls.from_porespy_network(
                network_dict,
                volume_length=volume_length,
                voxel_size=voxel_size,
                trim_disconnected=trim_disconnected,
                volume_dims=volume_dims,
            )

        if binary_volume is None:
            raise ValueError(
                "Either binary_volume or network must be provided."
            )

        try:
            from poregen.features.snow2 import snow2
        except ImportError:
            raise ImportError(
                "poregen is required for network extraction from binary volumes. "
                "Install it or use from_porespy_network() with a pre-extracted network."
            )

        if volume_length is None:
            volume_length = binary_volume.shape[0]

        # SNOW expects pore space = True (1), solid = False (0)
        # Our convention is pore space = 0, solid = 1, so we invert
        pore_space = 1 - binary_volume

        # Run SNOW algorithm to partition pore space and extract network
        partitioning = snow2(pore_space, voxel_size=1.0)
        porespy_network = partitioning.network

        return cls.from_porespy_network(
            porespy_network,
            volume_length=volume_length,
            voxel_size=voxel_size,
            trim_disconnected=trim_disconnected,
        )

    @classmethod
    def from_porespy_network(
        cls,
        network_dict: dict,
        volume_length: int,
        voxel_size: float = 1.0,
        trim_disconnected: bool = True,
        volume_dims: Optional[tuple] = None,
    ) -> "PoreNetworkPermeability":
        """
        Create a PoreNetworkPermeability from a PoreSpy network dictionary.

        This is the output format from PoreSpy extraction methods like SNOW.

        Args:
            network_dict: Dictionary with pore/throat properties as returned
                          by PoreSpy network extraction.
            volume_length: The edge length of the cubic sample in voxels.
            voxel_size: Physical size of each voxel in meters.
            trim_disconnected: If True, remove disconnected pores.
            volume_dims: Tuple (Lx, Ly, Lz) for non-cubic samples.

        Returns:
            A new PoreNetworkPermeability instance.
        """
        # Convert PoreSpy network to OpenPNM network
        pn = openpnm.io.network_from_porespy(network_dict)

        # Trim disconnected pores (important for flow calculations)
        if trim_disconnected:
            health = openpnm.utils.check_network_health(pn)
            disconnected = health['disconnected_pores']
            if len(disconnected) > 0:
                openpnm.topotools.trim(network=pn, pores=disconnected)

        # Set up standard diameter/spacing properties
        # These map PoreSpy output names to OpenPNM expected names
        cls._setup_network_geometry(pn)

        return cls(pn, volume_length=volume_length, voxel_size=voxel_size,
                   volume_dims=volume_dims)

    @classmethod
    def from_openpnm_network(
        cls,
        network: openpnm.network.Network,
        volume_length: int,
        voxel_size: float = 1.0,
        trim_disconnected: bool = True,
        setup_geometry: bool = True,
        volume_dims: Optional[tuple] = None,
    ) -> "PoreNetworkPermeability":
        """
        Create a PoreNetworkPermeability from an existing OpenPNM network.

        Use this when you have a network that was created or loaded directly
        in OpenPNM (e.g., from a saved file or generated procedurally).

        Args:
            network: An OpenPNM Network object.
            volume_length: The edge length of the cubic sample in voxels.
            voxel_size: Physical size of each voxel in meters.
            trim_disconnected: If True, remove disconnected pores.
            setup_geometry: If True, set up standard geometry properties.
                            Set to False if properties are already configured.
            volume_dims: Tuple (Lx, Ly, Lz) for non-cubic samples.

        Returns:
            A new PoreNetworkPermeability instance.
        """
        if trim_disconnected:
            health = openpnm.utils.check_network_health(network)
            disconnected = health['disconnected_pores']
            if len(disconnected) > 0:
                openpnm.topotools.trim(network=network, pores=disconnected)

        if setup_geometry:
            cls._setup_network_geometry(network)

        return cls(network, volume_length=volume_length, voxel_size=voxel_size,
                   volume_dims=volume_dims)

    @staticmethod
    def _setup_network_geometry(pn: openpnm.network.Network) -> None:
        """
        Set up standard geometry properties on the network.

        This maps PoreSpy/SNOW output property names to the standard names
        expected by OpenPNM physics models and flow algorithms.

        Modifies the network in-place.
        """
        # Pore diameter: use equivalent diameter if available
        if 'pore.equivalent_diameter' in pn:
            pn['pore.diameter'] = pn['pore.equivalent_diameter']

        # Throat diameter for conductance: use equivalent diameter
        # (from cross-sectional area, r_e = sqrt(area/pi)).
        # This is the physically correct choice for Hagen-Poiseuille
        # conductance through non-circular cross-sections.
        if 'throat.equivalent_diameter' in pn:
            pn['throat.diameter'] = pn['throat.equivalent_diameter']
        elif 'throat.inscribed_diameter' in pn:
            pn['throat.diameter'] = pn['throat.inscribed_diameter']

        # Throat spacing: use total length (pore-to-pore distance)
        if 'throat.total_length' in pn:
            pn['throat.spacing'] = pn['throat.total_length']

        # Compute throat radius for convenience (used in conductance)
        if 'throat.diameter' in pn:
            pn['throat.radius'] = pn['throat.diameter'] / 2.0

    # =========================================================================
    # Absolute Permeability
    # =========================================================================

    def calculate_absolute_permeability(
        self,
        directions: Sequence[Literal['x', 'y', 'z']] = ('x', 'y', 'z'),
    ) -> AbsolutePermeabilityResult:
        """
        Calculate the absolute (single-phase) permeability of the network.

        Uses Stokes flow simulation with Hagen-Poiseuille conductance model
        (cylindrical tubes). A pressure gradient is applied in each direction
        and the resulting flow rate is used to compute permeability.

        The result is cached and also stored internally for use in relative
        permeability calculations.

        Args:
            directions: Which flow directions to simulate. Default is all three.

        Returns:
            AbsolutePermeabilityResult containing permeabilities in each direction.
            Access dimensionless values via K_x, K_y, K_z, K_mean, K_array.
            Access physical values [m^2] via K_x_physical, K_y_physical, etc.

        Notes:
            - Permeability is computed from Darcy's law: K = Q * L / (A * dP)
            - Dimensionless values are stored internally (used for kr calculations)
            - Physical values [m^2] = dimensionless * voxel_size^2
            - Common unit conversions:
              - 1 Darcy ≈ 9.87e-13 m^2
              - 1 milliDarcy ≈ 9.87e-16 m^2
        """
        # Compute base hydraulic conductance using Hagen-Poiseuille equation
        # g = pi * r^4 / (8 * mu * L) [here mu=1 for dimensionless]
        self._g_base = (
            np.pi * (self.pn['throat.radius'] ** 4) /
            (8.0 * self.pn['throat.spacing'])
        )

        # Set the conductance on the network
        self.pn['throat.hydraulic_conductance'] = self._g_base

        # Create a dummy phase for flow simulation
        phase = openpnm.phase.Phase(network=self.pn)

        # Per-direction Darcy geometry (L = flow length, A = cross-section)
        Lx, Ly, Lz = self.volume_dims
        dims_LA = {
            'x': (Lx, Ly * Lz),
            'y': (Ly, Lx * Lz),
            'z': (Lz, Lx * Ly),
        }

        K_results = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        for direction in directions:
            # Get inlet and outlet pores for this direction
            inlet_pores = self.pn.pores(f'{direction}min')
            outlet_pores = self.pn.pores(f'{direction}max')

            # Skip if no pores at boundaries (shouldn't happen for normal networks)
            if len(inlet_pores) == 0 or len(outlet_pores) == 0:
                warnings.warn(
                    f"No pores found at {direction} boundaries. "
                    f"K_{direction} will be 0."
                )
                continue

            # Set up and run Stokes flow
            sf = openpnm.algorithms.StokesFlow(network=self.pn, phase=phase)
            sf.set_value_BC(pores=inlet_pores, values=1.0)  # P_in = 1
            sf.set_value_BC(pores=outlet_pores, values=0.0)  # P_out = 0
            sf.run()

            # Compute permeability from Darcy's law:
            # Q = K * A * dP / (mu * L)  =>  K = Q * L / (A * dP)
            # With mu=1 and dP=1, K = Q * L / A (dimensionless, in voxel units)
            L, A = dims_LA[direction]
            Q = sf.rate(pores=inlet_pores)[0]
            K_results[direction] = Q * (L / A)

        # Cache and return result
        # Store dimensionless values; physical units accessed via properties
        self._abs_perm_result = AbsolutePermeabilityResult(
            K_x=K_results['x'],
            K_y=K_results['y'],
            K_z=K_results['z'],
            voxel_size=self.voxel_size,
        )

        return self._abs_perm_result

    # =========================================================================
    # Drainage Simulation
    # =========================================================================

    _OPPOSITE_FACE = {
        'xmin': 'xmax', 'xmax': 'xmin',
        'ymin': 'ymax', 'ymax': 'ymin',
        'zmin': 'zmax', 'zmax': 'zmin',
    }

    def run_drainage_simulation(
        self,
        inlet_face: Literal['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] = 'xmin',
        contact_angle: float = 140.0,  # degrees
        surface_tension: float = 0.48,  # N/m
        trapping: bool = False,
        outlet_face: Optional[Literal['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']] = None,
    ) -> np.ndarray:
        """
        Run a drainage (non-wetting phase invasion) simulation.

        Simulates the invasion of a non-wetting phase (e.g., oil, mercury) into
        a network initially saturated with wetting phase (e.g., water).

        The invasion proceeds by pressure-controlled drainage: throats are
        invaded when the applied capillary pressure exceeds their entry pressure,
        calculated using the Washburn equation.

        This method only runs the invasion simulation. The Pc-saturation curve
        and relative permeabilities are computed step-by-step in
        calculate_relative_permeability_curves(), which calculates saturations
        directly from invaded pore volumes at each Pc.

        Args:
            inlet_face: Which face the non-wetting phase invades from.
                        One of 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'.
            contact_angle: Contact angle of the non-wetting phase [degrees].
                           >90 means non-wetting. Default 140 (mercury-like).
            surface_tension: Interfacial tension [N/m]. Default 0.48 (mercury/air).
            trapping: If True, apply trapping after drainage. Defending-phase
                      clusters disconnected from the outlet face are marked as
                      trapped and cannot be invaded. This yields realistic
                      residual saturations.
            outlet_face: Face where the defending phase can escape (required
                         for trapping). If None and trapping is True, defaults
                         to the face opposite the inlet.

        Returns:
            Array of unique capillary pressures (Pc) at which invasion events
            occurred, sorted in ascending order. Pass these to
            calculate_relative_permeability_curves() or use a subset.

        Notes:
            - Physical throat diameters are computed using self.voxel_size
            - Entry pressure uses the Washburn equation: Pc = -4*sigma*cos(theta)/d
        """
        # Compute physical throat diameters for capillary entry pressure.
        # Use inscribed diameter (narrowest constriction) for Washburn Pc,
        # NOT the equivalent diameter used for conductance. The meniscus
        # must fit through the tightest opening to invade.
        if 'throat.inscribed_diameter' in self.pn:
            capillary_diameter = self.pn['throat.inscribed_diameter']
        else:
            capillary_diameter = self.pn['throat.diameter']
        self.pn['throat.physical_diameter'] = capillary_diameter * self.voxel_size

        # Create non-wetting phase and set properties
        nwp = openpnm.phase.Phase(network=self.pn)
        nwp['pore.contact_angle'] = contact_angle
        nwp['pore.surface_tension'] = surface_tension
        nwp['throat.contact_angle'] = contact_angle
        nwp['throat.surface_tension'] = surface_tension

        # Add Washburn entry pressure model
        # Pc = -4 * sigma * cos(theta) / d
        nwp.add_model(
            propname='throat.entry_pressure',
            model=openpnm.models.physics.capillary_pressure.washburn,
            surface_tension='throat.surface_tension',
            contact_angle='throat.contact_angle',
            diameter='throat.physical_diameter',
        )
        nwp.regenerate_models()

        # Set up and run drainage simulation
        drn = openpnm.algorithms.Drainage(network=self.pn, phase=nwp)
        inlet_pores = self.pn.pores(inlet_face)
        drn.set_inlet_BC(pores=inlet_pores)

        # Set outlet BC for trapping (defending phase escape route)
        if trapping:
            if outlet_face is None:
                outlet_face = self._OPPOSITE_FACE[inlet_face]
            outlet_pores = self.pn.pores(outlet_face)
            drn.set_outlet_BC(pores=outlet_pores)

        drn.run()

        # Apply trapping: disconnected defending-phase clusters become trapped
        if trapping:
            drn.apply_trapping()

        # Cache the drainage algorithm for relative permeability calculations
        self._drainage_algorithm = drn

        # Return the unique Pc values where invasion events occurred
        # (these can be passed to calculate_relative_permeability_curves)
        Pc_values = np.unique(drn['throat.invasion_pressure'])
        Pc_values = Pc_values[np.isfinite(Pc_values)]
        Pc_values = np.sort(Pc_values)

        return Pc_values

    # =========================================================================
    # Relative Permeability
    # =========================================================================

    def calculate_relative_permeability_curves(
        self,
        Pc_values: Optional[np.ndarray] = None,
        directions: Sequence[Literal['x', 'y', 'z']] = ('x', 'y', 'z'),
    ) -> RelativePermeabilityResult:
        """
        Calculate relative permeability curves for both phases.

        For each specified capillary pressure, this method:
        1. Determines which pores/throats are invaded by non-wetting phase
        2. Calculates effective permeability for each phase
        3. Normalizes by absolute permeability to get relative permeability

        The wetting phase flows only through non-invaded regions, and the
        non-wetting phase flows only through invaded regions.

        Args:
            Pc_values: Array of capillary pressures at which to calculate kr.
                       If None, uses all unique invasion pressures from drainage.
            directions: Which flow directions to simulate.

        Returns:
            RelativePermeabilityResult with Sw, Pc, kr_wetting, kr_nonwetting arrays.

        Raises:
            RuntimeError: If drainage simulation hasn't been run yet.
            RuntimeError: If absolute permeability hasn't been calculated yet.

        Notes:
            - This method can be slow for many Pc values, as it runs multiple
              Stokes flow simulations per saturation state.
            - Relative permeability is kr = K_eff / K_abs for each phase.
            - Zero conductance (1e-10) is assigned to throats not conducting
              each respective phase.
        """
        # Validate prerequisites
        if self._drainage_algorithm is None:
            raise RuntimeError(
                "Must run drainage simulation first. "
                "Call run_drainage_simulation() before calculating relative permeability."
            )

        if self._abs_perm_result is None:
            raise RuntimeError(
                "Must calculate absolute permeability first. "
                "Call calculate_absolute_permeability() before calculating relative permeability."
            )

        drn = self._drainage_algorithm
        K_abs = self._abs_perm_result.K_array

        # Get Pc values to sample
        if Pc_values is None:
            # Use all unique invasion pressures
            Pc_values = np.unique(drn['throat.invasion_pressure'])
            Pc_values = Pc_values[np.isfinite(Pc_values)]
            Pc_values = np.sort(Pc_values)

        # Get volume arrays for saturation calculation
        # Pore volume is always defined; throat volume is often negligible/undefined
        pore_vol = self.pn['pore.volume']
        throat_vol = self.pn.get('throat.volume', np.zeros(self.pn.Nt))
        total_vol = pore_vol.sum() + throat_vol.sum()

        # Per-direction Darcy geometry
        Lx, Ly, Lz = self.volume_dims
        dims_LA = {
            'x': (Lx, Ly * Lz),
            'y': (Ly, Lx * Lz),
            'z': (Lz, Lx * Ly),
        }

        # Results storage
        results = {
            'Sw': [],
            'Pc': [],
            'kr_wetting': [],
            'kr_nonwetting': [],
        }

        # Loop over each capillary pressure state
        for Pc in Pc_values:
            # Determine invaded regions at this Pc
            # Invaded = invasion_pressure <= current Pc (with small tolerance)
            throat_invaded = drn['throat.invasion_pressure'] <= Pc + 1e-6
            pore_invaded = drn['pore.invasion_pressure'] <= Pc + 1e-6

            # Calculate saturation
            vol_nwp = (
                pore_vol[pore_invaded].sum() +
                throat_vol[throat_invaded].sum()
            )
            Snwp = vol_nwp / total_vol
            Sw = 1.0 - Snwp

            results['Sw'].append(Sw)
            results['Pc'].append(Pc)

            # --- Calculate kr for non-wetting phase ---
            # NWP flows only through invaded throats
            kr_nwp_dir = self._calculate_phase_kr(
                throat_mask=throat_invaded,
                pore_mask=pore_invaded,
                K_abs=K_abs,
                directions=directions,
                dims_LA=dims_LA,
            )
            results['kr_nonwetting'].append(kr_nwp_dir)

            # --- Calculate kr for wetting phase ---
            # WP flows only through non-invaded throats
            kr_wp_dir = self._calculate_phase_kr(
                throat_mask=~throat_invaded,
                pore_mask=~pore_invaded,
                K_abs=K_abs,
                directions=directions,
                dims_LA=dims_LA,
            )
            results['kr_wetting'].append(kr_wp_dir)

        return RelativePermeabilityResult(
            Sw=np.array(results['Sw']),
            Pc=np.array(results['Pc']),
            kr_wetting=np.array(results['kr_wetting']),
            kr_nonwetting=np.array(results['kr_nonwetting']),
        )

    def _calculate_phase_kr(
        self,
        throat_mask: np.ndarray,
        pore_mask: np.ndarray,
        K_abs: np.ndarray,
        directions: Sequence[str],
        dims_LA: dict,
    ) -> list[float]:
        """
        Calculate relative permeability for a phase given its occupation mask.

        Internal method used by calculate_relative_permeability_curves().

        Args:
            throat_mask: Boolean array - True where this phase occupies throats.
            pore_mask: Boolean array - True where this phase occupies pores.
            K_abs: Absolute permeability array [K_x, K_y, K_z].
            directions: Which directions to calculate.
            dims_LA: Dict mapping direction -> (L, A) for Darcy geometry.

        Returns:
            List of relative permeabilities [kr_x, kr_y, kr_z] (or subset based
            on directions argument).
        """
        # Set up conductance: high where phase is present, negligible elsewhere
        g_phase = self._g_base.copy()
        g_phase[~throat_mask] = 1e-10  # Non-conducting throats
        self.pn['throat.hydraulic_conductance'] = g_phase

        kr_results = []

        for i, direction in enumerate(['x', 'y', 'z']):
            if direction not in directions:
                kr_results.append(0.0)
                continue

            # Skip if absolute permeability is zero (can't normalize)
            if K_abs[i] == 0:
                kr_results.append(0.0)
                continue

            inlet_pores = self.pn.pores(f'{direction}min')
            outlet_pores = self.pn.pores(f'{direction}max')

            # Check if phase spans from inlet to outlet
            # Simple connectivity check: are there phase-filled pores at both faces?
            inlet_has_phase = pore_mask[inlet_pores].any()
            outlet_has_phase = pore_mask[outlet_pores].any()

            if not (inlet_has_phase and outlet_has_phase):
                # Phase doesn't span the sample - zero relative permeability
                kr_results.append(0.0)
                continue

            # Run Stokes flow for this phase distribution
            try:
                phase = openpnm.phase.Phase(network=self.pn)
                sf = openpnm.algorithms.StokesFlow(network=self.pn, phase=phase)
                sf.set_value_BC(pores=inlet_pores, values=1.0)
                sf.set_value_BC(pores=outlet_pores, values=0.0)
                sf.run()

                # Compute effective permeability (dimensionless)
                L, A = dims_LA[direction]
                Q = sf.rate(pores=inlet_pores)[0]
                K_eff = Q * (L / A)

                # Relative permeability = K_eff / K_abs (both dimensionless)
                kr_results.append(K_eff / K_abs[i])

            except Exception:
                # Solver may fail if network is poorly connected
                kr_results.append(0.0)

        return kr_results

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def run_full_analysis(
        self,
        inlet_face: str = 'xmin',
        contact_angle: float = 140.0,
        surface_tension: float = 0.48,
    ) -> tuple[AbsolutePermeabilityResult, RelativePermeabilityResult]:
        """
        Run the complete two-phase flow analysis pipeline.

        This is a convenience method that runs:
        1. Absolute permeability calculation
        2. Drainage simulation
        3. Relative permeability curves (which includes Pc-Sw data)

        Args:
            inlet_face: Which face for non-wetting phase invasion.
            contact_angle: Contact angle for non-wetting phase [degrees].
            surface_tension: Interfacial tension [N/m].

        Returns:
            Tuple of (AbsolutePermeabilityResult, RelativePermeabilityResult).
            The RelativePermeabilityResult contains Sw, Pc, kr_wetting, and
            kr_nonwetting arrays - effectively combining drainage curve and
            relative permeability data.
        """
        abs_perm = self.calculate_absolute_permeability()
        self.run_drainage_simulation(
            inlet_face=inlet_face,
            contact_angle=contact_angle,
            surface_tension=surface_tension,
        )
        rel_perm = self.calculate_relative_permeability_curves()

        return abs_perm, rel_perm

    @property
    def network(self) -> openpnm.network.Network:
        """Access the underlying OpenPNM network object."""
        return self.pn

    @property
    def num_pores(self) -> int:
        """Number of pores in the network."""
        return self.pn.Np

    @property
    def num_throats(self) -> int:
        """Number of throats in the network."""
        return self.pn.Nt

    def __repr__(self) -> str:
        return (
            f"PoreNetworkPermeability("
            f"pores={self.num_pores}, "
            f"throats={self.num_throats}, "
            f"volume_length={self.volume_length}, "
            f"voxel_size={self.voxel_size})"
        )

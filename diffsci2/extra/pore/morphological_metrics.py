"""
Morphological Metrics for Binary Porous Media
===============================================

This module provides a class for computing morphological (non-PNM) metrics
from 3D binary porous media volumes.

Metrics included:
    - Porosity
    - Two-point correlation (3D volume and 2D slices)
    - Surface area density (via marching cubes)
    - Mean pore size (from local thickness)
    - Mean curvature of the pore-solid interface
    - Euler number density

Convention:
    Binary volumes are [H, W, D] numpy arrays where 1 = solid, 0 = pore.

Example usage:
    >>> from diffsci2.extra.pore import MorphologicalMetrics
    >>>
    >>> metrics = MorphologicalMetrics(binary_volume, voxel_size=3e-6)
    >>> phi = metrics.porosity()
    >>> tpc = metrics.two_point_correlation_3d()
    >>> Sa = metrics.surface_area_density()
    >>> pore_size = metrics.mean_pore_size()
    >>> curv = metrics.curvature()
    >>> chi = metrics.euler_number_density()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import porespy
from scipy.ndimage import distance_transform_edt, laplace
from skimage.measure import marching_cubes, euler_number


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class TwoPointCorrelationResult:
    """
    Container for two-point correlation function results.

    Attributes:
        distance: Array of lag distances [physical units if voxel_size given].
        probability: S2(r), the two-point probability function. S2(0) = porosity.
    """
    distance: np.ndarray
    probability: np.ndarray

    @property
    def porosity(self) -> float:
        """Porosity extracted from S2(0)."""
        return float(self.probability[0])

    @property
    def probability_normalized(self) -> np.ndarray:
        """Normalized TPC: (S2 - phi^2) / (phi - phi^2), equals 1 at r=0."""
        phi = self.probability[0]
        denom = phi - phi**2
        if denom == 0:
            return np.zeros_like(self.probability)
        return (self.probability - phi**2) / denom


@dataclass
class MeanPoreSizeResult:
    """
    Container for mean pore size results from local thickness.

    Attributes:
        mean: Mean pore diameter [physical units].
        median: Median pore diameter [physical units].
        std: Standard deviation of pore diameter [physical units].
        local_thickness_field: Full 3D local thickness field [physical units].
    """
    mean: float
    median: float
    std: float
    local_thickness_field: np.ndarray


@dataclass
class CurvatureResult:
    """
    Container for interface curvature results.

    Curvature is estimated from the Laplacian of the signed distance
    function at the pore-solid interface (level-set approach).

    Attributes:
        mean: Mean curvature across interface voxels [1/physical unit].
        median: Median curvature [1/physical unit].
        std: Standard deviation of curvature [1/physical unit].
        values: Curvature at each interface voxel [1/physical unit].
    """
    mean: float
    median: float
    std: float
    values: np.ndarray


# =============================================================================
# Main Class
# =============================================================================

class MorphologicalMetrics:
    """
    Compute morphological metrics from a 3D binary porous media volume.

    This class wraps PoreSpy, scikit-image, and SciPy functionality to provide
    a streamlined interface for non-PNM morphological characterization of
    porous media.

    Convention:
        binary_volume[i, j, k] = 1 means solid, 0 means pore space.

    Attributes:
        binary_volume: The input volume as boolean array (True = solid).
        pore_space: Inverted volume (True = pore).
        voxel_size: Physical size of each voxel [m].
    """

    def __init__(self, binary_volume: np.ndarray, voxel_size: float = 1.0):
        """
        Initialize from a 3D binary volume.

        Args:
            binary_volume: 3D numpy array of shape [H, W, D] where
                           1 = solid, 0 = pore space.
            voxel_size: Physical size of each voxel in meters.
        """
        self.binary_volume = np.asarray(binary_volume, dtype=bool)
        self.pore_space = ~self.binary_volume
        self.voxel_size = voxel_size
        self._local_thickness: Optional[np.ndarray] = None

    # =========================================================================
    # Porosity
    # =========================================================================

    def porosity(self) -> float:
        """
        Compute the porosity (volume fraction of pore space).

        Returns:
            Porosity as a float in [0, 1].
        """
        return float(self.pore_space.sum() / self.pore_space.size)

    # =========================================================================
    # Two-Point Correlation
    # =========================================================================

    def two_point_correlation_3d(self) -> TwoPointCorrelationResult:
        """
        Compute the two-point correlation function S2(r) on the full 3D volume.

        Uses porespy.metrics.two_point_correlation on the pore phase.

        Returns:
            TwoPointCorrelationResult with distance and S2(r) arrays.
        """
        tpc = porespy.metrics.two_point_correlation(self.pore_space)
        return TwoPointCorrelationResult(
            distance=tpc.distance * self.voxel_size,
            probability=tpc.probability_scaled,
        )

    def two_point_correlation_2d(
        self,
        axis: int = 0,
        slice_idx: Optional[int] = None,
    ) -> TwoPointCorrelationResult:
        """
        Compute the two-point correlation function S2(r) on a 2D slice.

        Extracts a single 2D slice from the volume along the given axis
        and computes the TPC on that slice.

        Args:
            axis: Axis perpendicular to the slice (0, 1, or 2).
            slice_idx: Index of the slice along the axis.
                       Defaults to the middle slice.

        Returns:
            TwoPointCorrelationResult with distance and S2(r) arrays.
        """
        if slice_idx is None:
            slice_idx = self.pore_space.shape[axis] // 2
        slicing = [slice(None)] * 3
        slicing[axis] = slice_idx
        slice_2d = self.pore_space[tuple(slicing)]
        tpc = porespy.metrics.two_point_correlation(slice_2d)
        return TwoPointCorrelationResult(
            distance=tpc.distance * self.voxel_size,
            probability=tpc.probability_scaled,
        )

    # =========================================================================
    # Surface Area Density
    # =========================================================================

    def surface_area_density(self) -> float:
        """
        Compute the specific surface area (surface area per unit volume).

        Uses the marching cubes algorithm to extract the pore-solid interface
        and computes the total triangle area of the resulting mesh.

        Returns:
            Surface area density [1/physical unit], i.e. area / volume.
        """
        verts, faces, _, _ = marching_cubes(
            self.pore_space.astype(float),
            level=0.5,
            spacing=(self.voxel_size,) * 3,
        )
        # Triangle areas via cross product
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        total_area = 0.5 * np.linalg.norm(cross, axis=1).sum()
        total_volume = np.prod(np.array(self.pore_space.shape)) * self.voxel_size**3
        return float(total_area / total_volume)

    # =========================================================================
    # Mean Pore Size
    # =========================================================================

    def mean_pore_size(self) -> MeanPoreSizeResult:
        """
        Compute pore size statistics from the local thickness transform.

        Uses porespy.filters.local_thickness to assign each pore voxel
        a diameter equal to the largest inscribed sphere that contains it.

        Returns:
            MeanPoreSizeResult with mean, median, std, and the full field.
        """
        if self._local_thickness is None:
            self._local_thickness = porespy.filters.local_thickness(self.pore_space)
        lt = self._local_thickness * self.voxel_size
        pore_values = lt[self.pore_space & (lt > 0)]
        if len(pore_values) == 0:
            return MeanPoreSizeResult(
                mean=0.0, median=0.0, std=0.0,
                local_thickness_field=lt,
            )
        return MeanPoreSizeResult(
            mean=float(np.mean(pore_values)),
            median=float(np.median(pore_values)),
            std=float(np.std(pore_values)),
            local_thickness_field=lt,
        )

    # =========================================================================
    # Curvature
    # =========================================================================

    def curvature(self) -> CurvatureResult:
        """
        Estimate the mean curvature of the pore-solid interface.

        Uses a level-set approach: the signed distance function is computed
        (positive in pore space, negative in solid), and its Laplacian at
        the interface approximates the sum of principal curvatures
        (kappa_1 + kappa_2).

        Interface voxels are defined as pore voxels within sqrt(3) voxels
        of the solid phase (i.e., face-, edge-, or corner-adjacent to solid).

        Returns:
            CurvatureResult with mean, median, std, and per-voxel values.
        """
        dist_pore = distance_transform_edt(self.pore_space)
        dist_solid = distance_transform_edt(~self.pore_space)
        signed_dist = dist_pore - dist_solid

        kappa_field = laplace(signed_dist.astype(np.float64))

        # Interface: pore voxels adjacent to solid (within sqrt(3) voxels)
        interface_mask = (dist_pore > 0) & (dist_pore <= np.sqrt(3))
        curvature_values = kappa_field[interface_mask] / self.voxel_size

        if len(curvature_values) == 0:
            return CurvatureResult(mean=0.0, median=0.0, std=0.0, values=curvature_values)
        return CurvatureResult(
            mean=float(np.mean(curvature_values)),
            median=float(np.median(curvature_values)),
            std=float(np.std(curvature_values)),
            values=curvature_values,
        )

    # =========================================================================
    # Euler Number Density
    # =========================================================================

    def euler_number_density(self) -> float:
        """
        Compute the Euler number density of the pore phase.

        The Euler number chi is a topological invariant related to
        connectivity: chi = #components - #tunnels + #cavities.
        A more negative chi indicates a more connected pore network.

        Uses 6-connectivity (face-adjacent) for the pore phase, which is
        the standard choice for porous media (conservative connectivity).

        Returns:
            Euler number density [1/physical_volume], i.e. chi / V_total.
        """
        chi = euler_number(self.pore_space, connectivity=1)
        total_volume = np.prod(np.array(self.pore_space.shape)) * self.voxel_size**3
        return float(chi / total_volume)

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self) -> str:
        H, W, D = self.binary_volume.shape
        return (
            f"MorphologicalMetrics("
            f"shape=({H}, {W}, {D}), "
            f"voxel_size={self.voxel_size})"
        )

"""
Corey and Brooks-Corey relative permeability / capillary pressure models.

Provides analytical curve models that can be fitted to pore-network simulation
data, producing smooth constitutive relations suitable for reservoir-scale
solvers (e.g. Buckley-Leverett).

Corey model
-----------
    Se  = (Sw - Swr) / (1 - Swr - Snwr)
    kr_w  = kr_w0 * Se^nw
    kr_nw = kr_nw0 * (1 - Se)^nnw

Brooks-Corey capillary pressure
-------------------------------
    Pc = Pe * Se^(-1/lambda_bc)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# =============================================================================
# Corey relative permeability model
# =============================================================================

@dataclass
class CoreyModelParameters:
    """
    Parameters of the Corey relative permeability model.

    Attributes:
        Swr: Residual (irreducible) wetting phase saturation
        Snwr: Residual (trapped) non-wetting phase saturation
        kr_w0: Endpoint kr for the wetting phase (at Se = 1)
        kr_nw0: Endpoint kr for the non-wetting phase (at Se = 0)
        nw: Corey exponent for the wetting phase
        nnw: Corey exponent for the non-wetting phase
    """
    Swr: float
    Snwr: float
    kr_w0: float
    kr_nw0: float
    nw: float
    nnw: float

    def Se(self, Sw) -> np.ndarray:
        """Normalized (effective) saturation, clipped to [0, 1]."""
        Sw = np.asarray(Sw, dtype=float)
        Se = (Sw - self.Swr) / (1.0 - self.Swr - self.Snwr)
        return np.clip(Se, 0.0, 1.0)

    def kr_wetting(self, Sw) -> np.ndarray:
        """Wetting-phase relative permeability."""
        Se = self.Se(Sw)
        return self.kr_w0 * Se ** self.nw

    def kr_nonwetting(self, Sw) -> np.ndarray:
        """Non-wetting-phase relative permeability."""
        Se = self.Se(Sw)
        return self.kr_nw0 * (1.0 - Se) ** self.nnw

    def fractional_flow(self, Sw, mu_w: float, mu_nw: float) -> np.ndarray:
        """
        Fractional flow of the wetting phase.

        fw = 1 / (1 + (kr_nw / kr_w) * (mu_w / mu_nw))

        Returns 0 where kr_w = 0, and 1 where kr_nw = 0.
        """
        kr_w = self.kr_wetting(Sw)
        kr_nw = self.kr_nonwetting(Sw)
        Sw = np.asarray(Sw, dtype=float)
        fw = np.where(
            kr_w == 0, 0.0,
            np.where(
                kr_nw == 0, 1.0,
                1.0 / (1.0 + (kr_nw / kr_w) * (mu_w / mu_nw)),
            ),
        )
        return fw

    def __repr__(self) -> str:
        return (
            f"CoreyModelParameters(\n"
            f"  Swr={self.Swr:.4f}, Snwr={self.Snwr:.4f},\n"
            f"  kr_w0={self.kr_w0:.4f}, kr_nw0={self.kr_nw0:.4f},\n"
            f"  nw={self.nw:.2f}, nnw={self.nnw:.2f}\n"
            f")"
        )


# =============================================================================
# Brooks-Corey capillary pressure model
# =============================================================================

@dataclass
class BrooksCoreyCapillaryPressure:
    """
    Brooks-Corey capillary pressure model.

        Pc(Sw) = Pe * Se^(-1 / lambda_bc)

    Attributes:
        Pe: Entry pressure [Pa]
        lambda_bc: Pore size distribution index
        Swr: Residual wetting saturation
        Snwr: Residual non-wetting saturation
    """
    Pe: float
    lambda_bc: float
    Swr: float
    Snwr: float

    def Se(self, Sw) -> np.ndarray:
        """Normalized saturation, clipped to (0, 1]."""
        Sw = np.asarray(Sw, dtype=float)
        Se = (Sw - self.Swr) / (1.0 - self.Swr - self.Snwr)
        return np.clip(Se, 1e-12, 1.0)

    def Pc(self, Sw) -> np.ndarray:
        """Capillary pressure from wetting saturation."""
        return self.Pe * self.Se(Sw) ** (-1.0 / self.lambda_bc)

    def Sw_from_Pc(self, Pc) -> np.ndarray:
        """Wetting saturation from capillary pressure."""
        Pc = np.asarray(Pc, dtype=float)
        Se = (self.Pe / Pc) ** self.lambda_bc
        Se = np.clip(Se, 0.0, 1.0)
        return self.Swr + (1.0 - self.Swr - self.Snwr) * Se


# =============================================================================
# Fitting functions
# =============================================================================

def fit_corey_model(
    rel_perm_result,
    direction: str = 'mean',
    Swr_bounds: tuple = (0.0, 0.5),
    Snwr_bounds: tuple = (0.0, 0.5),
    nw_bounds: tuple = (1.0, 10.0),
    nnw_bounds: tuple = (1.0, 10.0),
) -> CoreyModelParameters:
    """
    Fit a Corey model to pore-network relative permeability data.

    Fits both kr_w and kr_nw simultaneously so that Swr and Snwr are
    consistent across the two curves.

    Args:
        rel_perm_result: A RelativePermeabilityResult (from PoreNetworkPermeability).
        direction: 'mean', 'x', 'y', or 'z'.
        Swr_bounds: Bounds on residual wetting saturation.
        Snwr_bounds: Bounds on residual non-wetting saturation.
        nw_bounds: Bounds on wetting-phase Corey exponent.
        nnw_bounds: Bounds on non-wetting-phase Corey exponent.

    Returns:
        Fitted CoreyModelParameters.
    """
    from scipy.optimize import curve_fit

    Sw = np.asarray(rel_perm_result.Sw, dtype=float)

    # Select kr columns
    if direction == 'mean':
        kr_w = np.asarray(rel_perm_result.kr_wetting_mean, dtype=float)
        kr_nw = np.asarray(rel_perm_result.kr_nonwetting_mean, dtype=float)
    else:
        idx = {'x': 0, 'y': 1, 'z': 2}[direction]
        kr_w = np.asarray(rel_perm_result.kr_wetting[:, idx], dtype=float)
        kr_nw = np.asarray(rel_perm_result.kr_nonwetting[:, idx], dtype=float)

    # Filter valid points (positive, finite)
    valid = np.isfinite(kr_w) & np.isfinite(kr_nw) & np.isfinite(Sw)
    Sw_v = Sw[valid]
    kr_w_v = kr_w[valid]
    kr_nw_v = kr_nw[valid]

    n = len(Sw_v)

    # Initial guesses from data
    Swr0 = max(Sw_v.min() - 0.01, 0.0)
    Snwr0 = max(1.0 - Sw_v.max() - 0.01, 0.0)
    kr_w0_0 = kr_w_v[Sw_v.argmax()] if kr_w_v.max() > 0 else 0.5
    kr_nw0_0 = kr_nw_v[Sw_v.argmin()] if kr_nw_v.max() > 0 else 0.5

    bounds_lo = [Swr_bounds[0], Snwr_bounds[0], 1e-6, 1e-6, nw_bounds[0], nnw_bounds[0]]
    bounds_hi = [Swr_bounds[1], Snwr_bounds[1], 1.0, 1.0, nw_bounds[1], nnw_bounds[1]]

    # Clamp initial guesses to stay within bounds
    p0 = [
        np.clip(Swr0, bounds_lo[0] + 1e-6, bounds_hi[0] - 1e-6),
        np.clip(Snwr0, bounds_lo[1] + 1e-6, bounds_hi[1] - 1e-6),
        np.clip(kr_w0_0, bounds_lo[2] + 1e-6, bounds_hi[2] - 1e-6),
        np.clip(kr_nw0_0, bounds_lo[3] + 1e-6, bounds_hi[3] - 1e-6),
        2.0,
        2.0,
    ]

    # Combined model: first n points are kr_w, next n are kr_nw
    def combined_model(Sw_doubled, Swr, Snwr, kr_w0, kr_nw0, nw, nnw):
        Sw_single = Sw_doubled[:n]
        Se = np.clip((Sw_single - Swr) / (1.0 - Swr - Snwr), 0.0, 1.0)
        kr_w_pred = kr_w0 * Se ** nw
        kr_nw_pred = kr_nw0 * (1.0 - Se) ** nnw
        return np.concatenate([kr_w_pred, kr_nw_pred])

    Sw_doubled = np.concatenate([Sw_v, Sw_v])
    kr_target = np.concatenate([kr_w_v, kr_nw_v])

    popt, _ = curve_fit(
        combined_model, Sw_doubled, kr_target,
        p0=p0, bounds=(bounds_lo, bounds_hi),
        method='trf', maxfev=10000,
    )

    return CoreyModelParameters(
        Swr=popt[0], Snwr=popt[1],
        kr_w0=popt[2], kr_nw0=popt[3],
        nw=popt[4], nnw=popt[5],
    )


def fit_brooks_corey_pc(
    rel_perm_result,
    Swr: Optional[float] = None,
    Snwr: Optional[float] = None,
) -> BrooksCoreyCapillaryPressure:
    """
    Fit a Brooks-Corey capillary pressure model.

    Args:
        rel_perm_result: A RelativePermeabilityResult.
        Swr: If None, estimated from data.
        Snwr: If None, estimated from data.

    Returns:
        Fitted BrooksCoreyCapillaryPressure.
    """
    from scipy.optimize import curve_fit

    Sw = np.asarray(rel_perm_result.Sw, dtype=float)
    Pc = np.asarray(rel_perm_result.Pc, dtype=float)

    valid = np.isfinite(Sw) & np.isfinite(Pc) & (Pc > 0)
    Sw_v = Sw[valid]
    Pc_v = Pc[valid]

    if Swr is None:
        Swr = max(Sw_v.min() - 0.01, 0.0)
    if Snwr is None:
        Snwr = max(1.0 - Sw_v.max() - 0.01, 0.0)

    Se_v = np.clip((Sw_v - Swr) / (1.0 - Swr - Snwr), 1e-12, 1.0)

    # Fit in log-log space: log(Pc) = log(Pe) - (1/lambda) * log(Se)
    log_Se = np.log(Se_v)
    log_Pc = np.log(Pc_v)

    valid_log = np.isfinite(log_Se) & np.isfinite(log_Pc)
    coeffs = np.polyfit(log_Se[valid_log], log_Pc[valid_log], 1)
    # coeffs[0] = -1/lambda, coeffs[1] = log(Pe)
    lambda_bc = -1.0 / coeffs[0]
    Pe = np.exp(coeffs[1])

    return BrooksCoreyCapillaryPressure(
        Pe=Pe, lambda_bc=lambda_bc, Swr=Swr, Snwr=Snwr,
    )

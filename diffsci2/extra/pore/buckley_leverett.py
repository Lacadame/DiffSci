"""
Buckley-Leverett 1D two-phase displacement solver.

Solves the classical BL equation using the method of characteristics
with Welge tangent construction (exact analytical solution).

    dSw/dt + (qt / (phi * A)) * dfw/dx = 0

where fw is the fractional flow of the wetting phase:

    fw = 1 / (1 + (kr_nw / kr_w) * (mu_w / mu_nw))

The solution consists of:
- A rarefaction fan where saturation varies continuously
- A shock front (Welge construction) where saturation jumps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .corey_model import CoreyModelParameters


# =============================================================================
# Result container
# =============================================================================

@dataclass
class BuckleyLeverettResult:
    """
    Result of a Buckley-Leverett displacement calculation.

    Attributes:
        Sw_profile: Saturations in the composite profile (rarefaction + shock)
        xD_profile: Corresponding dimensionless positions xD = dfw/dSw
        Sw_shock: Saturation at the shock front (downstream side)
        Sw_injection: Injection (upstream) saturation
        fw_shock: Fractional flow at the shock
        xD_shock: Dimensionless shock velocity (= dfw/dSw at shock)
        Sw_array: Fine Sw grid used for fw computation
        fw_array: Fractional flow on that grid
        dfw_dSw_array: Derivative of fw on that grid
    """
    Sw_profile: np.ndarray
    xD_profile: np.ndarray
    Sw_shock: float
    Sw_injection: float
    fw_shock: float
    xD_shock: float
    Sw_array: np.ndarray
    fw_array: np.ndarray
    dfw_dSw_array: np.ndarray


# =============================================================================
# Solver
# =============================================================================

class BuckleyLeverettSolver:
    """
    1D Buckley-Leverett solver using Welge tangent construction.

    Args:
        kr_model: Fitted Corey model providing kr_w(Sw) and kr_nw(Sw).
        mu_w: Wetting phase viscosity [Pa.s].
        mu_nw: Non-wetting phase viscosity [Pa.s].
        porosity: Porosity of the medium [-].
        total_rate: Total volumetric injection rate [m^3/s] (for dimensional output).
        area: Cross-sectional area [m^2] (for dimensional output).
        length: Domain length [m] (for dimensional output).
    """

    def __init__(
        self,
        kr_model: CoreyModelParameters,
        mu_w: float,
        mu_nw: float,
        porosity: float,
        total_rate: float = 1.0,
        area: float = 1.0,
        length: float = 1.0,
    ):
        self.kr_model = kr_model
        self.mu_w = mu_w
        self.mu_nw = mu_nw
        self.porosity = porosity
        self.total_rate = total_rate
        self.area = area
        self.length = length

    @property
    def interstitial_velocity(self) -> float:
        """Interstitial velocity v = qt / (phi * A) [m/s]."""
        return self.total_rate / (self.porosity * self.area)

    def fractional_flow(self, Sw) -> np.ndarray:
        """Fractional flow fw(Sw)."""
        return self.kr_model.fractional_flow(Sw, self.mu_w, self.mu_nw)

    def welge_construction(
        self,
        Sw_initial: Optional[float] = None,
        n_sw: int = 1000,
    ) -> BuckleyLeverettResult:
        """
        Compute the BL solution via Welge tangent construction.

        Args:
            Sw_initial: Initial reservoir saturation. Defaults to Swr
                        (connate water, reservoir full of non-wetting phase).
            n_sw: Number of points on the Sw grid.

        Returns:
            BuckleyLeverettResult with the saturation profile and diagnostics.
        """
        m = self.kr_model
        if Sw_initial is None:
            Sw_initial = m.Swr

        Sw_max = 1.0 - m.Snwr  # injection saturation (maximum wetting)

        # Fine Sw grid over the mobile range
        Sw_arr = np.linspace(Sw_initial, Sw_max, n_sw)
        fw_arr = self.fractional_flow(Sw_arr)
        fw_initial = self.fractional_flow(Sw_initial)

        # Numerical derivative dfw/dSw
        dfw = np.gradient(fw_arr, Sw_arr)

        # Welge tangent construction.
        # The tangent from (Sw_initial, fw_initial) to the fw curve touches
        # at the point where the secant slope is maximized. This is because
        # d/dSw [secant] = 0  =>  dfw/dSw = secant at that point.
        dSw = Sw_arr - Sw_initial
        dSw[0] = np.nan  # avoid 0/0
        secant_slope = (fw_arr - fw_initial) / dSw
        secant_slope[0] = 0.0

        # The shock saturation is where the secant slope is maximized
        idx_shock = np.nanargmax(secant_slope)
        Sw_shock = float(Sw_arr[idx_shock])
        fw_shock = float(fw_arr[idx_shock])
        xD_shock = float(secant_slope[idx_shock])  # = dfw/dSw at shock

        # Build the composite saturation profile (xD = x / (v * t)):
        #   xD = 0            : Sw = Sw_max  (injection)
        #   rarefaction fan   : Sw decreases, xD = dfw/dSw(Sw)
        #   xD = xD_shock     : Sw jumps from Sw_shock to Sw_initial
        #   xD > xD_shock     : Sw = Sw_initial

        # Rarefaction: Sw from Sw_shock to Sw_max, sorted by increasing xD
        rar_mask = (Sw_arr >= Sw_shock) & (Sw_arr <= Sw_max)
        Sw_rar = Sw_arr[rar_mask]
        xD_rar = dfw[rar_mask]
        order = np.argsort(xD_rar)
        Sw_rar = Sw_rar[order]
        xD_rar = xD_rar[order]

        Sw_profile = np.concatenate([
            [Sw_max],             # behind rarefaction
            Sw_rar,               # rarefaction fan
            [Sw_shock],           # just before shock
            [Sw_initial],         # just after shock
            [Sw_initial],         # ahead of shock
        ])
        xD_profile = np.concatenate([
            [0.0],                # injector
            xD_rar,               # rarefaction
            [xD_shock],           # shock (left)
            [xD_shock],           # shock (right)
            [xD_shock * 2.0],     # far ahead
        ])

        return BuckleyLeverettResult(
            Sw_profile=Sw_profile,
            xD_profile=xD_profile,
            Sw_shock=Sw_shock,
            Sw_injection=Sw_max,
            fw_shock=fw_shock,
            xD_shock=xD_shock,
            Sw_array=Sw_arr,
            fw_array=fw_arr,
            dfw_dSw_array=dfw,
        )

    def saturation_profile(
        self,
        x: np.ndarray,
        t: float,
        result: Optional[BuckleyLeverettResult] = None,
    ) -> np.ndarray:
        """
        Evaluate Sw(x) at a given time t.

        Args:
            x: Spatial positions [m].
            t: Time [s].
            result: Pre-computed BL result. If None, runs welge_construction().

        Returns:
            Array of Sw values at each x.
        """
        if result is None:
            result = self.welge_construction()

        v = self.interstitial_velocity
        xD_query = np.asarray(x) / (v * t)

        return np.interp(
            xD_query, result.xD_profile, result.Sw_profile,
            left=result.Sw_injection,                          # behind front
            right=result.Sw_profile[-1],                       # ahead of front
        )

    def breakthrough_time(
        self,
        result: Optional[BuckleyLeverettResult] = None,
    ) -> float:
        """Time [s] at which the shock front reaches x = length."""
        if result is None:
            result = self.welge_construction()
        v = self.interstitial_velocity
        return self.length / (v * result.xD_shock)

    def oil_recovery(
        self,
        t: float,
        result: Optional[BuckleyLeverettResult] = None,
        n_x: int = 500,
    ) -> float:
        """
        Fractional oil (non-wetting phase) recovery at time t.

        Recovery = (initial oil in place - current oil in place) / initial OOIP.
        """
        if result is None:
            result = self.welge_construction()

        Sw_initial = self.kr_model.Swr
        x = np.linspace(0, self.length, n_x)
        Sw = self.saturation_profile(x, t, result)
        # Average saturation
        Sw_avg = np.trapz(Sw, x) / self.length
        # Recovery = change in average Sw / initial oil saturation
        return (Sw_avg - Sw_initial) / (1.0 - Sw_initial)

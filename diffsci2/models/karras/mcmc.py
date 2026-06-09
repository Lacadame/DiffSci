"""
MCMC Steppers for Posterior Sampling.

This module provides various MCMC methods for sampling from distributions
defined by a score function. Used by LanPaint and other posterior sampling methods.

Methods:
- Overdamped Langevin (ULA)
- Tamed ULA (gradient clipping for stability)
- MALA (with Metropolis-Hastings correction)
- BAOAB (stable underdamped Langevin splitting)
- FLD (Fast Langevin Dynamics)
- HMC (Hamiltonian Monte Carlo)
"""

from abc import ABC, abstractmethod
from typing import Callable
import math
import torch
from torch import Tensor


class MCMCStepper(ABC):
    """
    Base class for MCMC steppers.

    All steppers take a score function and current state, returning an updated state.
    Stateful steppers (underdamped methods) maintain momentum internally.
    """

    @abstractmethod
    def step(self, x: Tensor, score_fn: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Take one MCMC step.

        Args:
            x: Current state [batch, *shape]
            score_fn: Callable that returns score ∇log p(x) at x

        Returns:
            Updated state [batch, *shape]
        """
        pass

    def reset(self):
        """Reset internal state (e.g., momentum). Called at start of each noise level."""
        pass


# =============================================================================
# Overdamped Methods (no momentum)
# =============================================================================

class OverdampedLangevin(MCMCStepper):
    """
    Overdamped (Unadjusted) Langevin Algorithm.

    Update: x ← x + h·s(x) + √(2h)·ξ,  ξ ~ N(0, I)

    Simple and fast but biased for large step sizes.
    """

    def __init__(self, h: float = 0.1):
        """
        Args:
            h: Step size
        """
        self.h = h

    def step(self, x: Tensor, score_fn: Callable[[Tensor], Tensor]) -> Tensor:
        score = score_fn(x)
        noise = torch.randn_like(x)
        return x + self.h * score + math.sqrt(2 * self.h) * noise


class TamedULA(MCMCStepper):
    """
    Tamed Unadjusted Langevin Algorithm.

    Update: x ← x + h·s(x)/(1 + h·||s(x)||) + √(2h)·ξ

    The "taming" bounds the gradient to prevent explosion, providing
    stability guarantees even for non-convex targets.

    Reference: Atchadé (2017) "Taming the Monster"
    """

    def __init__(self, h: float = 0.1):
        """
        Args:
            h: Step size
        """
        self.h = h

    def step(self, x: Tensor, score_fn: Callable[[Tensor], Tensor]) -> Tensor:
        score = score_fn(x)

        # Compute per-sample score norm
        # Flatten spatial dims, keep batch dim
        batch_size = x.shape[0]
        score_flat = score.reshape(batch_size, -1)
        score_norm = torch.norm(score_flat, dim=1, keepdim=True)

        # Reshape norm for broadcasting: [batch, 1, 1, ...]
        for _ in range(x.dim() - 1):
            score_norm = score_norm.unsqueeze(-1)
        score_norm = score_norm.squeeze(1)  # Remove the keepdim dimension

        # Tame the score
        tamed_score = score / (1 + self.h * score_norm)

        noise = torch.randn_like(x)
        return x + self.h * tamed_score + math.sqrt(2 * self.h) * noise


class MALA(MCMCStepper):
    """
    Metropolis-Adjusted Langevin Algorithm.

    Proposes using Langevin dynamics, then accepts/rejects with MH step.
    Asymptotically exact but requires computing log-density (not just score).

    Note: Currently implements UNADJUSTED version (no MH step).
    TODO: Add proper MH correction when log-density is available.
    """

    def __init__(self, h: float = 0.1):
        """
        Args:
            h: Step size
        """
        self.h = h

    def step(self, x: Tensor, score_fn: Callable[[Tensor], Tensor]) -> Tensor:
        # Currently just overdamped (no MH correction)
        score = score_fn(x)
        noise = torch.randn_like(x)
        return x + self.h * score + math.sqrt(2 * self.h) * noise


# =============================================================================
# Underdamped Methods (with momentum)
# =============================================================================

class BAOAB(MCMCStepper):
    """
    BAOAB splitting scheme for underdamped Langevin dynamics.

    The underdamped Langevin SDE is:
        dx = v dt
        dv = -γv dt + ∇log p(x) dt + √(2γ) dW

    BAOAB splits this into:
        B: v ← v + (h/2)·∇log p(x)     (half kick)
        A: x ← x + (h/2)·v             (half drift)
        O: v ← e^(-γh)·v + √(1-e^(-2γh))·ξ  (O-U step)
        A: x ← x + (h/2)·v             (half drift)
        B: v ← v + (h/2)·∇log p(x)     (half kick)

    This is the gold standard for underdamped Langevin - very stable
    and accurate without requiring MH correction.

    Note: Requires 2 score evaluations per step.

    Reference: Leimkuhler & Matthews (2013)
    """

    def __init__(self, h: float = 0.1, gamma: float = 1.0):
        """
        Args:
            h: Step size
            gamma: Friction coefficient. Higher = more damping, faster mixing.
                   Lower = more momentum, better exploration.
        """
        self.h = h
        self.gamma = gamma
        self.v = None  # Momentum, initialized on first step

    def step(self, x: Tensor, score_fn: Callable[[Tensor], Tensor]) -> Tensor:
        h = self.h
        gamma = self.gamma

        # Initialize momentum if needed
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.randn_like(x)

        # B: half kick
        score = score_fn(x)
        self.v = self.v + (h / 2) * score

        # A: half drift
        x = x + (h / 2) * self.v

        # O: Ornstein-Uhlenbeck step
        c1 = math.exp(-gamma * h)
        c2 = math.sqrt(1 - c1 ** 2)
        noise = torch.randn_like(self.v)
        self.v = c1 * self.v + c2 * noise

        # A: half drift
        x = x + (h / 2) * self.v

        # B: half kick (recompute score at new position)
        score = score_fn(x)
        self.v = self.v + (h / 2) * score

        return x

    def reset(self):
        """Reset momentum."""
        self.v = None


class FLD(MCMCStepper):
    """
    Fast Langevin Dynamics (from LanPaint paper).

    Uses diffusion damping A = 1/σ² with score decomposition s = C - A·x.

    Note: This method can be unstable. Consider using BAOAB instead.
    """

    def __init__(self, h: float = 0.3, gamma: float = 15.0, sigma: float = 1.0):
        """
        Args:
            h: Step size
            gamma: Friction coefficient
            sigma: Current noise level (for computing A = 1/σ²)
        """
        self.h = h
        self.gamma = gamma
        self.sigma = sigma
        self.v = None

    def set_sigma(self, sigma: float):
        """Update sigma for current noise level."""
        self.sigma = sigma

    def step(self, x: Tensor, score_fn: Callable[[Tensor], Tensor]) -> Tensor:
        h = self.h
        gamma = self.gamma

        # Initialize momentum if needed
        if self.v is None or self.v.shape != x.shape:
            self.v = math.sqrt(gamma) * torch.randn_like(x)

        score = score_fn(x)

        # Diffusion damping
        A = 1.0 / (self.sigma ** 2)

        # Decompose: s = C - A*x, so C = s + A*x
        C = score + A * x

        # Decay factor
        exp_Gh = math.exp(-gamma * h)

        # Momentum update
        # v_drift = exp(-Γh) v + (1-exp(-Γh))/Γ · (C - A·x)
        # Note: C - A*x = score
        v_drift = exp_Gh * self.v + (1 - exp_Gh) * score / gamma

        # Stochastic term
        noise_scale = math.sqrt(gamma * (1 - exp_Gh ** 2))
        noise = torch.randn_like(self.v)
        self.v = v_drift + noise_scale * noise

        # Position update (trapezoidal)
        x_new = x + h * (self.v + v_drift) / 2

        return x_new

    def reset(self):
        """Reset momentum."""
        self.v = None


# =============================================================================
# Hamiltonian Monte Carlo
# =============================================================================

class HMC(MCMCStepper):
    """
    Hamiltonian Monte Carlo with leapfrog integration.

    Uses Hamiltonian dynamics to propose moves, then optionally
    accepts/rejects with MH step.

    Currently implements unadjusted HMC (no MH correction).

    Note: Requires n_leapfrog * 2 score evaluations per step.
    """

    def __init__(self, epsilon: float = 0.01, n_leapfrog: int = 10):
        """
        Args:
            epsilon: Leapfrog step size
            n_leapfrog: Number of leapfrog steps per HMC trajectory
        """
        self.epsilon = epsilon
        self.n_leapfrog = n_leapfrog

    def step(self, x: Tensor, score_fn: Callable[[Tensor], Tensor]) -> Tensor:
        epsilon = self.epsilon

        # Sample momentum
        p = torch.randn_like(x)

        # Leapfrog integration
        for _ in range(self.n_leapfrog):
            # Half step momentum
            score = score_fn(x)
            p = p + (epsilon / 2) * score

            # Full step position
            x = x + epsilon * p

            # Half step momentum
            score = score_fn(x)
            p = p + (epsilon / 2) * score

        return x


# =============================================================================
# Factory Function
# =============================================================================

STEPPER_REGISTRY = {
    "overdamped": OverdampedLangevin,
    "ula": OverdampedLangevin,
    "tamed_ula": TamedULA,
    "tamed": TamedULA,
    "mala": MALA,
    "baoab": BAOAB,
    "fld": FLD,
    "hmc": HMC,
}


def get_stepper(method: str, **kwargs) -> MCMCStepper:
    """
    Factory function to create MCMC steppers.

    Args:
        method: One of "overdamped", "tamed_ula", "mala", "baoab", "fld", "hmc"
        **kwargs: Method-specific arguments

    Returns:
        Initialized MCMCStepper instance

    Example:
        >>> stepper = get_stepper("tamed_ula", h=0.1)
        >>> x_new = stepper.step(x, score_fn)
    """
    method = method.lower()
    if method not in STEPPER_REGISTRY:
        available = ", ".join(sorted(STEPPER_REGISTRY.keys()))
        raise ValueError(f"Unknown MCMC method '{method}'. Available: {available}")

    stepper_class = STEPPER_REGISTRY[method]
    return stepper_class(**kwargs)


def list_methods() -> list[str]:
    """Return list of available MCMC methods."""
    return sorted(set(STEPPER_REGISTRY.keys()))

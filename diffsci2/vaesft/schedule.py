"""Loss-weight schedule for SFT.

Phase 1 -- BCE warmup, steps [0, warmup_bce_steps):
    (w_bce, w_reg) = (base_bce, 0.0)

Phase 2 -- linear ramp of the regressor SFT term over `ramp_steps`:
    t = step - warmup_bce_steps
    alpha = t / ramp_steps
    (w_bce, w_reg) = (base_bce, base_reg * alpha)

Phase 3 -- steady state, steps >= warmup_bce_steps + ramp_steps:
    (w_bce, w_reg) = (base_bce, base_reg)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    bce: float
    reg: float


@dataclass(frozen=True)
class ScheduleCfg:
    warmup_bce_steps: int = 1000
    ramp_steps: int = 1000
    base_bce: float = 1.0
    base_reg: float = 1.0


def weights_at_step(step: int, cfg: ScheduleCfg) -> LossWeights:
    if step < cfg.warmup_bce_steps:
        return LossWeights(bce=cfg.base_bce, reg=0.0)
    t = step - cfg.warmup_bce_steps
    if t < cfg.ramp_steps:
        alpha = t / max(1, cfg.ramp_steps)
        return LossWeights(bce=cfg.base_bce, reg=cfg.base_reg * alpha)
    return LossWeights(bce=cfg.base_bce, reg=cfg.base_reg)

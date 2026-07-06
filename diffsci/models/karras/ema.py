from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


def _power_function_exp_from_std(std: float) -> float:
    """Convert EDM2 power-function EMA relative std to its exponent."""
    if std <= 0:
        raise ValueError("Power-function EMA std must be positive")
    target = float(std) ** -2
    roots = np.roots([1.0, 7.0, 16.0 - target, 12.0 - target])
    return float(np.max(roots.real))


def _power_function_beta(std: float, next_update: int) -> float:
    """EMA decay for one optimizer update using the EDM2 power profile."""
    if next_update <= 1:
        return 0.0
    exp = _power_function_exp_from_std(std)
    return float((1.0 - 1.0 / next_update) ** (exp + 1.0))


class ModelEMA:
    """Non-registered EMA shadow weights for a torch module.

    The helper intentionally keeps EMA tensors outside the LightningModule
    parameter tree so optimizers created from module.parameters() do not see
    duplicate frozen parameters.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            ema_type: str = "traditional",
            decay: float = 0.999,
            halflife_steps: Optional[float] = None,
            rampup_ratio: Optional[float] = None,
            power_function_stds: Optional[list[float]] = None,
            device: Optional[str | torch.device] = None,
            profile_index: int = 0):
        self.ema_type = str(ema_type).lower()
        if self.ema_type not in {"traditional", "power"}:
            raise ValueError("ema_type must be 'traditional' or 'power'")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")

        self.decay = float(decay)
        self.halflife_steps = halflife_steps
        self.rampup_ratio = rampup_ratio
        self.power_function_stds = (
            [0.05] if power_function_stds is None else list(power_function_stds)
        )
        if len(self.power_function_stds) == 0:
            raise ValueError("power_function_stds must contain at least one value")
        self.device = torch.device(device) if device is not None else None
        self.profile_index = int(profile_index)
        self.num_updates = 0
        self.last_beta: Optional[float] = None
        self.profiles: list[dict[str, Any]] = []
        self.reset(model)

    @property
    def has_shadow(self) -> bool:
        return len(self.profiles) > 0 and len(self.profiles[0]["params"]) > 0

    def _profile_specs(self) -> list[dict[str, Any]]:
        if self.ema_type == "power":
            return [
                {"name": f"power_std_{std:g}", "std": float(std)}
                for std in self.power_function_stds
            ]
        return [{"name": "traditional", "std": None}]

    def _clone_for_shadow(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor.clone()

    def _coerce_like_shadow(
            self,
            tensor: torch.Tensor,
            shadow: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device=shadow.device, dtype=shadow.dtype)

    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        self.profiles = []
        params = {
            name: self._clone_for_shadow(param)
            for name, param in model.named_parameters()
        }
        buffers = {
            name: self._clone_for_shadow(buffer)
            for name, buffer in model.named_buffers()
        }
        for spec in self._profile_specs():
            self.profiles.append({
                **spec,
                "params": {name: value.clone() for name, value in params.items()},
                "buffers": {
                    name: value.clone() for name, value in buffers.items()
                },
            })
        self.num_updates = 0
        self.last_beta = None

    def _traditional_beta(self, next_update: int) -> float:
        if self.halflife_steps is None:
            return self.decay
        halflife_steps = float(self.halflife_steps)
        if self.rampup_ratio is not None:
            halflife_steps = min(
                halflife_steps,
                max(float(next_update), 1.0) * float(self.rampup_ratio),
            )
        return float(0.5 ** (1.0 / max(halflife_steps, 1e-8)))

    def _beta_for_profile(self, profile: dict[str, Any], next_update: int) -> float:
        if self.ema_type == "power":
            return _power_function_beta(profile["std"], next_update)
        return self._traditional_beta(next_update)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if not self.has_shadow:
            self.reset(model)

        next_update = self.num_updates + 1
        named_params = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())

        for profile in self.profiles:
            beta = self._beta_for_profile(profile, next_update)
            profile["last_beta"] = beta
            for name, param in named_params.items():
                if name not in profile["params"]:
                    profile["params"][name] = self._clone_for_shadow(param)
                    continue
                shadow = profile["params"][name]
                if self.device is None and shadow.device != param.device:
                    shadow = shadow.to(param.device)
                    profile["params"][name] = shadow
                shadow.lerp_(self._coerce_like_shadow(param, shadow), 1.0 - beta)

            profile["buffers"] = {
                name: self._clone_for_shadow(buffer)
                for name, buffer in named_buffers.items()
            }

        self.num_updates = next_update
        index = min(max(self.profile_index, 0), len(self.profiles) - 1)
        self.last_beta = self.profiles[index].get("last_beta")

    def selected_profile(self) -> dict[str, Any]:
        index = min(max(self.profile_index, 0), len(self.profiles) - 1)
        return self.profiles[index]

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
        profile = self.selected_profile()
        backup = {"params": {}, "buffers": {}}

        for name, param in model.named_parameters():
            if name not in profile["params"]:
                raise KeyError(f"EMA state is missing parameter {name!r}")
            backup["params"][name] = param.detach().clone()
            shadow = profile["params"][name].to(
                device=param.device,
                dtype=param.dtype,
            )
            param.copy_(shadow)

        for name, buffer in model.named_buffers():
            backup["buffers"][name] = buffer.detach().clone()
            if name in profile["buffers"]:
                shadow = profile["buffers"][name].to(
                    device=buffer.device,
                    dtype=buffer.dtype,
                )
                buffer.copy_(shadow)

        return backup

    @torch.no_grad()
    def restore(
            self,
            model: torch.nn.Module,
            backup: dict[str, dict[str, torch.Tensor]]) -> None:
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        for name, value in backup.get("params", {}).items():
            params[name].copy_(value.to(params[name]))
        for name, value in backup.get("buffers", {}).items():
            buffers[name].copy_(value.to(buffers[name]))

    def state_dict(self) -> dict[str, Any]:
        profiles = []
        for profile in self.profiles:
            profiles.append({
                "name": profile["name"],
                "std": profile["std"],
                "last_beta": profile.get("last_beta"),
                "params": {
                    name: value.detach().clone()
                    for name, value in profile["params"].items()
                },
                "buffers": {
                    name: value.detach().clone()
                    for name, value in profile["buffers"].items()
                },
            })
        return {
            "ema_type": self.ema_type,
            "decay": self.decay,
            "halflife_steps": self.halflife_steps,
            "rampup_ratio": self.rampup_ratio,
            "power_function_stds": self.power_function_stds,
            "profile_index": self.profile_index,
            "num_updates": self.num_updates,
            "last_beta": self.last_beta,
            "profiles": profiles,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema_type = state.get("ema_type", self.ema_type)
        self.decay = state.get("decay", self.decay)
        self.halflife_steps = state.get("halflife_steps", self.halflife_steps)
        self.rampup_ratio = state.get("rampup_ratio", self.rampup_ratio)
        self.power_function_stds = state.get(
            "power_function_stds",
            self.power_function_stds,
        )
        self.profile_index = state.get("profile_index", self.profile_index)
        self.num_updates = state.get("num_updates", 0)
        self.last_beta = state.get("last_beta")
        self.profiles = state["profiles"]

from typing import Callable, Any

import torch
from torch import Tensor
from jaxtyping import Float

import numpy as np

from . import schedulingfunctions


ScoreFunction = Callable[[Float[Tensor, "batch *shape"],  # noqa: F821
                          Float[Tensor, "batch"]],  # noqa: F821
                         Float[Tensor, "batch *shape"]]  # noqa: F821


class Integrator(torch.nn.Module):
    stochastic = False
    need_fns = False

    def step(self, x: Float[Tensor, "batch *shape"],  # noqa: F821
             t: Float[Tensor, ""],  # noqa: F722
             dt: Float[Tensor, ""],  # noqa: F722
             rhs: ScoreFunction,
             noise_strength: None | Any = None):  # TODO: Complete this type
        raise NotImplementedError


class EulerIntegrator(Integrator):
    def step(self, x: Float[Tensor, "batch *shape"],  # noqa: F821
             t: Float[Tensor, ""],  # noqa: F722
             dt: Float[Tensor, ""],  # noqa: F722
             rhs: ScoreFunction,
             noise_strength: None | Any = None):  # TODO: Complete this type
        return x + dt*rhs(x, t)


class HeunIntegrator(Integrator):
    def step(self, x: Float[Tensor, "batch *shape"],  # noqa: F821
             t: Float[Tensor, ""],  # noqa: F722
             dt: Float[Tensor, ""],  # noqa: F722
             rhs: ScoreFunction,
             noise_strength: None | Any = None):  # TODO: Complete this type
        rhs_euler = rhs(x, t)
        x_heun = x + dt*rhs_euler
        t_heun = t + dt
        rhs_heun = rhs(x_heun, t_heun)
        x = x + 0.5*(rhs_euler + rhs_heun)*dt
        return x


class EulerMaruyamaIntegrator(Integrator):
    stochastic = True

    def step(self, x: Float[Tensor, "batch *shape"],  # noqa: F821
             t: Float[Tensor, ""],  # noqa: F722
             dt: Float[Tensor, ""],  # noqa: F722
             rhs: ScoreFunction,
             noise_strength: None | Any = None):  # TODO: Complete this type
        assert (noise_strength is not None)
        return (x + rhs(x, t)*dt +
                (noise_strength(t) *
                 torch.randn_like(x) *
                 torch.sqrt(torch.abs(dt))))


class KarrasIntegrator(Integrator):
    stochastic = False                  # the integration step is from the ODE
    need_fns = True

    def __init__(self,
                 s_schurn: float = 40,  # parameters for EDM (fig. 5, App. E.1)
                 s_tmin: float = 0.05,
                 s_tmax: float = 50,
                 s_noise: float = 1.003) -> None:
        super().__init__()
        self.s_schurn = s_schurn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def step(self, x: Float[Tensor, "batch *shape"],  # noqa: F821
             t: Float[Tensor, ""],  # noqa: F722
             dt: Float[Tensor, ""],  # noqa: F722
             rhs: ScoreFunction,
             scheduler_fns: schedulingfunctions.SchedulingFunctions,
             noise_strength: None | Any = None,  # TODO: Complete this type
             nsteps: int = 100):
        backstep = min(self.s_schurn/nsteps, np.sqrt(2)-1)
        if self.s_tmin is not None:
            if not self.s_tmin <= t <= self.s_tmax:
                backstep = 0
        sigma = scheduler_fns.noise_fn(t)
        sigma_noise = sigma + backstep*sigma
        t_noise = scheduler_fns.inverse_noise_fn(sigma_noise)
        scale = scheduler_fns.scaling_fn(t)
        scale_noise = scheduler_fns.scaling_fn(t_noise)
        std = scale_noise * torch.sqrt(sigma_noise**2 - sigma**2)
        x_noise = ((scale_noise/scale)*x +
                   std * self.s_noise * torch.randn_like(x))

        rhs_euler = rhs(x_noise, t_noise)
        dt_noise = (t+dt) - t_noise
        x = x_noise + dt_noise*rhs_euler
        if (t+dt) > 0:
            rhs_heun = rhs(x, t+dt)
            x = x_noise + 0.5*(rhs_euler+rhs_heun)*dt_noise
        return x


def name_to_integrator(name: str) -> Integrator:
    if name == "euler":
        return EulerIntegrator()
    elif name == "heun":
        return HeunIntegrator()
    elif name == "euler-maruyama":
        return EulerMaruyamaIntegrator()
    elif name == "karras":
        return KarrasIntegrator()
    else:
        raise ValueError(f"Unknown integrator: {name}")

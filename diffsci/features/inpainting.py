import torch
from torch import Tensor
from jaxtyping import Float

from typing import Callable

from diffsci.models.karras import schedulers, integrators

ScoreFunction = Callable[[Float[Tensor, "batch *shape"],  # noqa: F821
                          Float[Tensor, "batch"]],  # noqa: F821
                         Float[Tensor, "batch *shape"]]  # noqa: F821
PropagationReturnType = (Float[Tensor, "batch *shape"] |
                         Float[Tensor, "nsteps+1 batch *shape"])  # noqa: F821


class Inpainting():
    def __init__(self,
                 scheduler: schedulers.Scheduler):
        self.scheduler = scheduler
        self.scheduler_fns = scheduler.scheduler_fns

    def reconstruct(self,
                    x_initial: Float[Tensor, "batch *shape"],  # noqa: F821
                    score_fn: ScoreFunction,
                    mask: Float[Tensor, "*shape"],  # noqa: F821
                    nsteps: int = 100,
                    record_history: bool = False):
        raise NotImplementedError


class RePaint(Inpainting):
    def __init__(self,
                 scheduler: schedulers.Scheduler,
                 integrator: integrators.Integrator
                 = integrators.EulerMaruyamaIntegrator()):
        super().__init__(scheduler)
        self.scheduler.integrator = integrator

    def renoise(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821
                t: float,
                t_noise: float):
        sigma = self.scheduler_fns.noise_fn(t)
        sigma_noise = self.scheduler_fns.noise_fn(t_noise)
        scale = self.scheduler_fns.scaling_fn(t)
        scale_noise = self.scheduler_fns.scaling_fn(t_noise)
        std = scale_noise * torch.sqrt(sigma_noise**2 - sigma**2)
        x_noise = (scale_noise/scale)*x + std * torch.randn_like(x)
        return x_noise

    def reconstruct(self,
                    x_initial: Float[Tensor, "batch *shape"],  # noqa: F821
                    score_fn: ScoreFunction,
                    mask: Float[Tensor, "*shape"],  # noqa: F821
                    n_resamples: int = 2,
                    resample_steps: int = 2,
                    nsteps: int = 100,
                    record_history: bool = False):
        if not (nsteps % resample_steps) == 0:
            raise ValueError("resample_steps should divide nsteps")
        partial = self.scheduler.propagate_partial
        x = (torch.randn_like(x_initial).to(x_initial) *
             self.scheduler.maximum_scale)
        t = self.scheduler.create_steps(nsteps+1)
        if record_history:
            history_shape = ([int(n_resamples*(nsteps/resample_steps))+1] +
                             list(x.shape))
            history = torch.zeros(history_shape).to(x)
            history[0] = x

        step = 0
        rem_steps = nsteps - (step + resample_steps)
        x = partial(x, score_fn, nsteps, step, rem_steps)
        step = step + resample_steps
        rem_steps = nsteps - (step + resample_steps)
        level = 0
        if rem_steps > 0:
            x = partial(x, score_fn, nsteps, step, rem_steps)
            for i in range(n_resamples):
                x = x_initial*mask + x*(1-mask)
                x = self.renoise(x, t[step+resample_steps], t[step])
                x = partial(x, score_fn, nsteps, step, rem_steps)
                if record_history:
                    history[level+i+1] = x
            step = step + resample_steps
            rem_steps = nsteps - (step + resample_steps)
            level = level + n_resamples
        if record_history:
            return history
        else:
            return x

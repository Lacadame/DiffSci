import torch
from torch import Tensor
from jaxtyping import Float

import functools

from typing import Callable

from diffsci.models.karras import schedulers, integrators
from diffsci.torchutils import broadcast_from_below

ScoreFunction = Callable[[Float[Tensor, "batch *shape"],  # noqa: F821
                          Float[Tensor, "batch"]],  # noqa: F821
                         Float[Tensor, "batch *shape"]]  # noqa: F821
PropagationReturnType = (Float[Tensor, "batch *shape"] |
                         Float[Tensor, "nsteps+1 batch *shape"])  # noqa: F821


class AnomalyDetector():
    def __init__(self,
                 scheduler: schedulers.Scheduler):
        self.scheduler = scheduler
        self.scheduler_fns = scheduler.scheduler_fns

    def reconstruct(self,
                    x_initial: Float[Tensor, "batch *shape"],  # noqa: F821
                    score_fn: ScoreFunction,
                    step: int,
                    nsteps: int = 100,
                    record_history: bool = False):
        raise NotImplementedError


class AnoDDPM(AnomalyDetector):
    def __init__(self,
                 scheduler: schedulers.Scheduler,
                 integrator: integrators.Integrator
                 = integrators.EulerMaruyamaIntegrator()):
        super().__init__(scheduler)
        self.scheduler.integrator = integrator

    def reconstruct(self,
                    x_initial: Float[Tensor, "batch *shape"],  # noqa: F821
                    score_fn: ScoreFunction,
                    step: int,
                    nsteps: int = 100,
                    record_history: bool = False,
                    Gaussian: bool = True):
        if not Gaussian:
            raise NotImplementedError
        x_noised = self.scheduler.apply_noise(x_initial, nsteps, step)

        x_rec = self.scheduler.propagate_partial(x=x_noised,
                                                 score_fn=score_fn,
                                                 nsteps=nsteps,
                                                 initial_step=step,
                                                 record_history=record_history)
        return x_rec

    def reconstruction_error(self,
                             x_initial:
                             Float[Tensor, "batch *shape"],  # noqa: F821
                             score_fn: ScoreFunction,
                             step: int,
                             nsteps: int = 100,
                             input_dim: int = 1):
        x_rec = self.reconstruct(x_initial, score_fn, step, nsteps)
        e_sq = (x_initial - x_rec)**2
        if input_dim == 1:
            error = torch.sum(e_sq, dim=-1)
        elif input_dim == 2:
            error = torch.sum(e_sq, dim=(-1,-2))
        else:
            raise NotImplementedError
        return error


class DDAD(AnomalyDetector):
    def __init__(self,
                 scheduler: schedulers.Scheduler):
        super().__init__(scheduler)

    def correction(self,
                   x: Float[Tensor, "batch *shape"],  # noqa: F821
                   y_history: Float[Tensor, "b *sh"],  # noqa: F821
                   ti: Float[Tensor, "batch"]):  # noqa: F821):
        y = y_history[int(ti)]
        return y - x

    def rhs_reconstruction(self,
                           x: Float[Tensor, "batch *shape"],  # noqa: F821
                           ti: Float[Tensor, "batch"],  # noqa: F821
                           y_history: Float[Tensor, "b *sh"],   # noqa: F821
                           score_fn: ScoreFunction,
                           nsteps: int = 100,
                           w: float = 3.0,
                           backward: bool = True,
                           ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        t = ti*torch.ones(x.shape[0]).to(x)
        t_ = broadcast_from_below(t, x)
        sigma = self.scheduler_fns.noise_fn(t)
        sigma_deriv = self.scheduler_fns.noise_fn_deriv(t)
        sigma_ = broadcast_from_below(sigma, x)
        sigma_deriv_ = broadcast_from_below(sigma_deriv, x)

        if self.scheduler_fns.constant_scaling_fn:
            if self.scheduler_fns.identity_noise_fn:
                multiplier = t_
            if self.scheduler_fns.has_pf_score_multiplier:
                multiplier = self.scheduler_fns.pf_score_multiplier(t_)
            else:
                multiplier = (sigma_ * sigma_deriv_)
            score = score_fn(x, sigma) + w*self.correction(x, y_history, ti)
            res = -multiplier*score
            if self.scheduler.integrator.stochastic:
                stochastic_factor = -(self.scheduler.langevin_factor(t_) *
                                      sigma_**2 *
                                      score)
                if not backward:
                    stochastic_factor = -stochastic_factor
                res += stochastic_factor
        else:
            s = self.scheduler_fns.scaling_fn(t_)
            sderiv = self.scheduler_fns.scaling_fn_deriv(t_)  # 0
            scale_multiplier = sderiv/s  # 0
            if self.scheduler_fns.has_pf_score_multiplier:
                multiplier = self.scheduler_fns.pf_score_multiplier(t_)
            else:
                cov_deriv = (self.scheduler_fns.noise_fn_deriv(t_) *
                             self.scheduler_fns.noise_fn(t_))
                multiplier = s*cov_deriv
            score = score_fn(x/s, sigma) + w*self.correction(x/s,
                                                             y_history,
                                                             ti)
            res = scale_multiplier*x - multiplier*score
        return res

    def reconstruct(self,
                    x_initial: Float[Tensor, "batch *shape"],  # noqa: F821
                    score_fn: ScoreFunction,
                    nsteps: int = 100,
                    initial_step: int = 0,
                    w: float = 3.0,
                    integrator: integrators.Integrator
                    = integrators.HeunIntegrator(),
                    record_history: bool = False,
                    backward: bool = True,
                    ) -> PropagationReturnType:  # noqa: F821
        if initial_step > nsteps:
            print(initial_step, nsteps)
            raise ValueError("Step larger than num of steps:{step}>{nsteps}")
        x = self.scheduler.apply_noise(x_initial, nsteps, initial_step)
        self.scheduler.integrator = integrators.HeunIntegrator()
        y_history = self.scheduler.propagate_forward(x_initial,
                                                     score_fn,
                                                     nsteps,
                                                     stochastic=True,
                                                     record_history=True)
        t = self.scheduler.create_steps(nsteps+1).to(x)
        if not backward:
            t = t.flip(0)
            raise NotImplementedError
        dt = torch.diff(t)
        if record_history:
            history_shape = [nsteps+1] + list(x.shape)
            history = torch.zeros(history_shape).to(x)
            history[initial_step] = x
        rhs = functools.partial(self.rhs_reconstruction,
                                y_history=y_history,
                                score_fn=score_fn,
                                nsteps=nsteps,
                                w=w)
        self.scheduler.integrator = integrator
        for i in range(initial_step, nsteps):
            x = self.scheduler.integrator.step(x, t[i], dt[i], rhs,
                                               self.scheduler.noise_injection)
            if record_history:
                history[i+1] = x
        if record_history:
            return history
        else:
            return x

    def reconstruction_error(self,
                             x_initial:
                             Float[Tensor, "batch *shape"],     # noqa: F821
                             score_fn: ScoreFunction,
                             step: int = 0,
                             nsteps: int = 100,
                             w: float = 3.0,
                             integrator: integrators.Integrator
                             = integrators.HeunIntegrator(),
                             input_dim: int = 1,):
        x_rec = self.reconstruct(x_initial, score_fn, nsteps, step, w,
                                 integrator)
        e_sq = (x_initial - x_rec)**2
        if input_dim == 1:
            error = torch.sum(e_sq, dim=-1)
        elif input_dim == 2:
            error = torch.sum(e_sq, dim=(-1,-2))
        else:
            raise NotImplementedError
        return error

import functools
from typing import Callable

import torch
from torch import Tensor
from jaxtyping import Float

from diffsci.torchutils import broadcast_from_below
from . import schedulingfunctions
from . import integrators


SchFunType = str | schedulingfunctions.SchedulingFunctions
ScoreFunction = Callable[[Float[Tensor, "batch *shape"],  # noqa: F821
                          Float[Tensor, "batch"]],  # noqa: F821
                         Float[Tensor, "batch *shape"]]  # noqa: F821
ScoreRHS = Callable  # TODO: Complete this type
OdeIntegrator = str | Callable[[Float[Tensor, "batch *shape"],  # noqa: F821]]]
                                Float[Tensor, "batch"],  # noqa: F821
                                Float[Tensor, ""],  # noqa: F722
                                ScoreFunction],  # noqa: F821
                               Float[Tensor, "batch *shape"]]  # noqa: F821
PropagationReturnType = (Float[Tensor, "batch *shape"] |
                         Float[Tensor, "nsteps+1 batch *shape"])  # noqa: F821


class Scheduler(torch.nn.Module):
    def __init__(
            self,
            scheduler_fns: schedulingfunctions.SchedulingFunctions,
            integrator: integrators.Integrator,
            maximum_scale: float,
            stochastic_integrator: integrators.Integrator | None = None
            ):
        super().__init__()
        self.scheduler_fns = scheduler_fns
        self._integrator = integrator
        self.maximum_scale = maximum_scale
        if stochastic_integrator is None:
            stochastic_integrator = integrators.EulerMaruyamaIntegrator()
        else:
            assert stochastic_integrator.stochastic is True
        self.stochastic_integrator = stochastic_integrator
        self._temporary_integrator = None
        self.langevin_const = 1.0
        self.langevin_interval = None

    def propagate(self,
                  x: Float[Tensor, "batch *shape"],  # noqa: F821
                  score_fn: ScoreFunction,
                  nsteps: int = 100,
                  record_history: bool = False,
                  backward: bool = True,
                  stochastic: bool = False,
                  ) -> PropagationReturnType:  # noqa: F821

        integrator = (self.integrator
                      if not stochastic
                      else self.stochastic_integrator)
        t = self.create_steps(nsteps+1).to(x)  # [nsteps]
        if not backward:
            t = t.flip(0)
        dt = torch.diff(t)
        if record_history:
            history_shape = [nsteps+1] + list(x.shape)
            history = torch.zeros(history_shape).to(x)
            history[0] = x
        rhs = functools.partial(self.rhs,
                                score_fn=score_fn,
                                backward=backward,
                                stochastic=integrator.stochastic)

        if integrator.need_fns:
            step = functools.partial(integrator.step,
                                     scheduler_fns=self.scheduler_fns,
                                     nsteps=nsteps)
        else:
            step = integrator.step
        for i in range(nsteps):
            x = step(x, t[i], dt[i], rhs, noise_strength=self.noise_injection)
            if record_history:
                history[i+1] = x
        if record_history:
            return history
        else:
            return x

    def inpaint(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821
                y: Float[Tensor, "nsteps+1 batch *shape"],  # noqa: F722
                mask: Float[Tensor, "*shape"],  # noqa: F821
                score_fn: ScoreFunction,
                nsteps: int = 100,
                record_history: bool = False,
                ) -> PropagationReturnType:  # noqa: F821

        t = self.create_steps(nsteps+1).to(x)  # [nsteps]
        dt = torch.diff(t)
        if record_history:
            history_shape = [nsteps+1] + list(x.shape)
            history = torch.zeros(history_shape).to(x)
            history[0] = x
        rhs = functools.partial(self.rhs,
                                score_fn=score_fn,
                                backward=True)
        x = x*(1-mask) + y[-1]*mask
        for i in range(nsteps):
            yind = -i-2
            x = self.integrator.step(x, t[i], dt[i], rhs, self.noise_injection)
            x = x*(1-mask) + y[yind]*mask
            if record_history:
                history[i+1] = x
        if record_history:
            return history
        else:
            return x

    def repaint(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821
                y: Float[Tensor, "nsteps+1 batch *shape"],  # noqa: F722
                mask: Float[Tensor, "*shape"],  # noqa: F821
                score_fn: ScoreFunction,
                nsteps: int = 100,
                rsteps: int = 10,       # num of steps performed at resample
                nresamples: int = 10,   # num of resamples for resample level
                record_history: bool = False,
                ) -> PropagationReturnType:  # noqa: F821
        if not (nsteps % rsteps) == 0:
            raise ValueError("rsteps should divide nsteps")
        t = self.create_steps(nsteps+1).to(x)
        if record_history:
            history_shape = ([int(nresamples*(nsteps/rsteps - 1))+2] +
                             list(x.shape))
            history = torch.zeros(history_shape).to(x)
            history[0] = x

        x = x*(1-mask) + y[-1]*mask
        step = 0
        fstep = rsteps
        x = self.propagate_partial(x, score_fn, nsteps, step, fstep)
        step = fstep
        fstep = fstep + rsteps
        level = 0
        while fstep <= nsteps:
            x = self.propagate_partial(x, score_fn, nsteps, step, fstep)
            for i in range(nresamples):
                x = x*(1-mask) + y[-fstep-1]*mask
                if record_history:
                    history[level+i+1] = x
                x = self.renoise(x, t[fstep], t[step])
                x = self.propagate_partial(x, score_fn, nsteps, step, fstep)
            step = fstep
            fstep = fstep + rsteps
            level = level + nresamples
        if not step == nsteps:
            raise ValueError('Wrong counting')
        if record_history:
            history[level+1] = x
            return history
        else:
            return x

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

    def propagate_partial(self,
                          x: Float[Tensor, "batch *shape"],  # noqa: F821
                          score_fn: ScoreFunction,
                          nsteps: int = 100,
                          initial_step: int = 0,
                          final_step: int = 100,
                          record_history: bool = False,
                          backward: bool = True,
                          stochastic: bool = False,
                          ) -> PropagationReturnType:  # noqa: F821
        integrator = (self.integrator
                      if not stochastic
                      else self.stochastic_integrator)
        t = self.create_steps(nsteps+1).to(x)  # [nsteps]
        if not backward:
            t = t.flip(0)
            raise NotImplementedError
        dt = torch.diff(t)
        if record_history:
            history_shape = [final_step - initial_step + 1] + list(x.shape)
            history = torch.zeros(history_shape).to(x)
            history[0] = x
        rhs = functools.partial(self.rhs,
                                score_fn=score_fn,
                                backward=backward,
                                stochastic=integrator.stochastic)
        if integrator.need_fns:
            step = functools.partial(integrator.step,
                                     scheduler_fns=self.scheduler_fns,
                                     nsteps=nsteps)
        else:
            step = integrator.step
        for i in range(initial_step, final_step):
            x = step(x, t[i], dt[i], rhs, noise_strength=self.noise_injection)
            if record_history:
                history[i-initial_step+1] = x
        if record_history:
            return history
        else:
            return x

    def langevin_factor(self,
                        t: Float[Tensor, "batch"],  # noqa: F821
                        type: str = 'const',
                        ) -> Float[Tensor, "batch"]:  # noqa: F821
        # Only multiples of the Song's Langevin factor are implemented
        standard_factor = (self.scheduler_fns.scaling_fn(t)**2 *
                           self.scheduler_fns.noise_fn_deriv(t) /
                           self.scheduler_fns.noise_fn(t))
        if type == 'const':
            if self.langevin_interval is not None:
                if len(t.shape) > 0:            # TODO: improve this hack
                    t_ = t[0]
                else:
                    t_ = t
                if t_ > self.langevin_interval[0] and t_ < self.langevin_interval[1]:
                    return self.langevin_const * standard_factor + 0*t
                else:
                    return 0*t
            else:
                return self.langevin_const * standard_factor + 0*t
        else:
            raise NotImplementedError

    def noise_injection(self,
                        t: Float[Tensor, "batch"],  # noqa: F821
                        ) -> Float[Tensor, "batch"]:  # noqa: F821
        return (torch.sqrt(2*self.langevin_factor(t)) *
                self.scheduler_fns.noise_fn(t))

    def rhs(self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            ti: Float[Tensor, "batch"],  # noqa: F821
            score_fn: ScoreFunction,
            backward: bool = True,
            stochastic: bool = False,
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
            score = score_fn(x, sigma)
            res = -multiplier*score
            if stochastic:
                stochastic_factor = -(self.langevin_factor(t_) *
                                      sigma_**2 *
                                      score)
                if not backward:
                    stochastic_factor = -stochastic_factor
                res += stochastic_factor
        else:
            s = self.scheduler_fns.scaling_fn(t_)
            sderiv = self.scheduler_fns.scaling_fn_deriv(t_)
            scale_multiplier = sderiv/s
            if self.scheduler_fns.has_pf_score_multiplier:
                multiplier = self.scheduler_fns.pf_score_multiplier(t_)
            else:
                cov_deriv = (self.scheduler_fns.noise_fn_deriv(t_) *
                             self.scheduler_fns.noise_fn(t_))
                multiplier = s*cov_deriv
            score = score_fn(x/s, sigma)
            res = scale_multiplier*x - multiplier*score
            if stochastic:
                stochastic_factor = -(self.langevin_factor(t_) *
                                      sigma_**2 *
                                      1/s *
                                      score)
                if not backward:
                    stochastic_factor = -stochastic_factor
                res += stochastic_factor
        return res

    def propagate_backward(self,
                           x: Float[Tensor, "batch *shape"],  # noqa: F821
                           score_fn: ScoreFunction,
                           nsteps: int = 100,
                           record_history: bool = False,
                           stochastic: bool = False,
                           ) -> PropagationReturnType:  # noqa: F821
        return self.propagate(x,
                              score_fn,
                              nsteps,
                              record_history,
                              backward=True,
                              stochastic=stochastic)

    def propagate_forward(self,
                          x: Float[Tensor, "batch *shape"],  # noqa: F821
                          score_fn: ScoreFunction,
                          nsteps: int = 100,
                          record_history: bool = False,
                          stochastic: bool = False,
                          ) -> PropagationReturnType:  # noqa: F821
        return self.propagate(x,
                              score_fn,
                              nsteps,
                              record_history,
                              backward=False,
                              stochastic=stochastic)

    def create_steps(n: int) -> Float[Tensor, 'n']:  # noqa: F821
        raise NotImplementedError

    def apply_noise(self,
                    x: Float[Tensor, "batch *shape"],  # noqa: F821
                    nsteps: int = 100,
                    step: int = 0
                    ) -> PropagationReturnType:  # noqa: F821
        if step > nsteps:
            raise ValueError("Step larger than num of steps:{step}>{nsteps}")
        t = self.create_steps(nsteps+1).to(x)
        t_step = t[step]
        sigma = self.scheduler_fns.noise_fn(t_step)
        scale = self.scheduler_fns.scaling_fn(t_step)
        noise = torch.randn(x.shape).to(x)
        x_noised = scale*x + scale*sigma*noise
        return x_noised

    def unset_temporary_integrator(self):
        self._temporary_integrator = None

    def set_temporary_integrator(
            self,
            integrator: str | integrators.Integrator):
        if type(integrator) is str:
            integrator = integrators.name_to_integrator(integrator)
        self._temporary_integrator = integrator

    @property
    def integrator(self) -> integrators.Integrator:
        if self._temporary_integrator is not None:
            return self._temporary_integrator
        else:
            return self._integrator


class EDMScheduler(Scheduler):
    def __init__(self,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 expoent_steps: float = 7.0,
                 scheduler_fns: SchFunType = "EDM"):
        if type(scheduler_fns) is str:
            scheduler_fns = schedulingfunctions.\
                name_to_scheduling_functions(scheduler_fns)
        integrator = integrators.HeunIntegrator()
        super().__init__(scheduler_fns,
                         integrator,
                         sigma_max)
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))
        self.register_buffer("expoent_steps", torch.tensor(expoent_steps))

    def create_steps(self, n: int) -> Float[Tensor, 'n']:  # noqa: F821
        s = torch.arange(n).to(self.expoent_steps)/(n-1)
        start = self.sigma_max**(1/self.expoent_steps)
        end = self.sigma_min**(1/self.expoent_steps)
        steps = (start + s*(end - start))**(self.expoent_steps)
        if not self.scheduler_fns.identity_noise_fn:
            steps = self.scheduler_fns.inverse_noise_fn(steps)
        return steps


class VPScheduler(Scheduler):
    def __init__(self,
                 epsilon_min: float = 0.001,
                 scheduler_fns: SchFunType = "VP",
                 *args,
                 **kwargs):
        if type(scheduler_fns) is str:
            scheduler_fns = schedulingfunctions.\
                name_to_scheduling_functions(scheduler_fns,
                                             *args,
                                             **kwargs)
        sigma_max = (scheduler_fns.noise_fn(torch.ones([1])) *
                     scheduler_fns.scaling_fn(torch.ones([1]))).item()
        integrator = integrators.HeunIntegrator()
        super().__init__(scheduler_fns,
                         integrator,
                         sigma_max)
        self.register_buffer("epsilon_min", torch.tensor(epsilon_min))

    def create_steps(self, n: int) -> Float[Tensor, 'n']:  # noqa: F821
        s = torch.arange(n).to(self.epsilon_min)/(n-1)
        steps = 1 + s*(self.epsilon_min - 1)
        return steps


class VEScheduler(Scheduler):
    def __init__(self,
                 sigma_min: float = 0.02,
                 sigma_max: float = 100,
                 scheduler_fns: SchFunType = "VE",
                 *args,
                 **kwargs):
        if type(scheduler_fns) is str:
            scheduler_fns = schedulingfunctions.\
                name_to_scheduling_functions(scheduler_fns,
                                             *args,
                                             **kwargs)
        integrator = integrators.HeunIntegrator()
        super().__init__(scheduler_fns,
                         integrator,
                         sigma_max)
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))

    def create_steps(self, n: int) -> Float[Tensor, 'n']:  # noqa: F821
        s = torch.arange(n).to(self.sigma_min)/(n-1)
        steps = self.sigma_max**2 * (self.sigma_min**2/self.sigma_max**2)**s
        return steps

from typing import Any, Callable

import torch
from torch import Tensor
from jaxtyping import Float

from porenet.torchutils import broadcast_from_below
from . import schedulers


NoisePredictor = Any  # TODO: Put the actual type here
NoiseInjector = Callable[[Float[Tensor, "..."]],
                         Float[Tensor, "..."]]
BackwardPropagationResult = (Float[Tensor, "nsteps+1 batch *shape"] |
                             Float[Tensor, "batch *shape"])


class Integrator(torch.nn.Module):
    def __init__(self,
                 scheduler: schedulers.DDPMScheduler):
        super().__init__()
        self.scheduler = scheduler

    def propagate_backward(self,
                           x: Float[Tensor, "batch *shape"],  # noqa: F821
                           noise_predictor: NoisePredictor,
                           nsteps: None | int = None,
                           record_history: bool = False
                           ) -> BackwardPropagationResult:
        raise NotImplementedError

    def propagate_forward(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            noise_predictor: NoisePredictor | None = None,
            nsteps: None | int = None,
            record_history: bool = False
            ) -> Float[Tensor, "nsteps batch *shape"]:  # noqa: F722
        raise NotImplementedError


# FORMULATION FROM DDPM PAPER

class ClassicalDDPMIntegrator(Integrator):
    def propagate_backward(self,
                           x: Float[Tensor, "batch *shape"],  # noqa: F821
                           noise_predictor: NoisePredictor,
                           nsteps: None | int = None,
                           record_history: bool = False
                           ) -> BackwardPropagationResult:
        T = self.scheduler.T if nsteps is None else nsteps
        tlist = torch.arange(T).to(x) + 1  # [T]
        if record_history:
            history = [x.clone()]
        for t in tlist.flip(0):
            x = self.step_backward(x, t, noise_predictor, T)
            if record_history:
                history.append(x)
        if record_history:
            history = torch.stack(history, axis=0)
            return history
        else:
            return x

    def propagate_forward(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            noise_predictor: NoisePredictor | None = None,
            nsteps: None | int = None,
            record_history: bool = False
            ) -> Float[Tensor, "nsteps batch *shape"]:  # noqa: F722
        x0 = x.clone()
        T = self.scheduler.T if nsteps is None else nsteps
        tlist = torch.arange(T).to(x) + 1  # [T]
        history = [x.clone()]
        for t in tlist.flip(0):
            x = self.step_forward(x, x0, t, noise_predictor, T)
            history.append(x)
        if record_history:
            history = torch.stack(history, axis=0)
            return history
        else:
            return x

    def step_backward(self,
                      x: Float[Tensor, "batch *shape"],  # noqa: F821
                      t: Float[Tensor, ""],  # noqa: F722
                      noise_predictor: NoisePredictor,
                      T: None | int = None,
                      ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        t = t.unsqueeze(0).expand(x.shape[0])  # [b]
        t_ = broadcast_from_below(t, x)  # [b *shape]
        sigma_t = self.noise_injector(t_, T)  # [b, *shape]
        calpha_t = self.scheduler.calpha(t_, T)  # [b, *shape]
        alpha_t = self.scheduler.alpha(t_, T)  # [b, *shape]
        beta_t = 1 - alpha_t
        noise_pred = noise_predictor(x, t)
        x0_direction = x - beta_t/torch.sqrt(1 - calpha_t)*noise_pred
        noise = torch.randn_like(x)
        # print(x)
        # print(noise)
        # print(sigma_t)
        # print(alpha_t)
        # print(noise_pred)
        result = 1/torch.sqrt(alpha_t)*x0_direction + sigma_t*noise
        # print(result)
        return result

    def step_forward(self,
                     x: Float[Tensor, "batch *shape"],  # noqa: F821
                     x0: Float[Tensor, "batch *shape"],  # noqa: F821
                     t: Float[Tensor, ""],  # noqa: F722
                     noise_predictor: NoisePredictor,
                     T: None | int = None,
                     ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        t = t.unsqueeze(0).expand(x.shape[0])  # [b]
        t_ = broadcast_from_below(t, x)  # [b *shape]
        noise = torch.randn_like(x)
        beta_t = self.scheduler.beta(t_, T)
        x_forward = (torch.sqrt(1 - beta_t) * x +
                     torch.sqrt(beta_t) * noise)
        return x_forward

    def noise_injector(self,
                       t: Float[Tensor, "..."],
                       T: None | int = None,
                       ) -> Float[Tensor, "..."]:
        raise NotImplementedError


class ClassicalDDPMIntegratorType1(ClassicalDDPMIntegrator):
    def noise_injector(self,
                       t: Float[Tensor, "..."],
                       T: None | int = None,
                       ) -> Float[Tensor, "..."]:
        return torch.sqrt(self.scheduler.beta(t, T))


class ClassicalDDPMIntegratorType2(ClassicalDDPMIntegrator):
    def noise_injector(self,
                       t: Float[Tensor, "..."],
                       T: None | int = None,
                       ) -> Float[Tensor, "..."]:
        calpha_prev = self.scheduler.calpha(t-1, T)
        calpha = self.scheduler.calpha(t, T)
        beta = self.scheduler.beta(t, T)
        return torch.sqrt((1 - calpha_prev)/(1 - calpha) * beta)


# FORMULATION FROM DDIM PAPER

class GeneralizedDDPMIntegrator(Integrator):
    def propagate_backward(self,
                           x: Float[Tensor, "batch *shape"],  # noqa: F821
                           noise_predictor: NoisePredictor,
                           nsteps: None | int = None,
                           record_history: bool = False
                           ) -> BackwardPropagationResult:
        T = self.scheduler.T if nsteps is None else nsteps
        tlist = torch.arange(T).to(x) + 1  # [T]
        if record_history:
            history = [x.clone()]
        for t in tlist.flip(0):
            x = self.step_backward(x, t, noise_predictor, T)
            if record_history:
                history.append(x)
        if record_history:
            history = torch.stack(history, axis=0)
            return history
        else:
            return x

    def propagate_forward(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            nsteps: None | int = None,
            record_history: bool = False
            ) -> Float[Tensor, "nsteps batch *shape"]:  # noqa: F722
        T = self.scheduler.T if nsteps is None else nsteps
        tlist = torch.arange(T).to(x) + 1  # [T]
        history = [x.clone()]
        for t in tlist.flip(0):
            x = self.step_forward(x, t, T)
            history.append(x)
        if record_history:
            history = torch.stack(history, axis=0)
            return history
        else:
            return x

    def step_backward(self,
                      x: Float[Tensor, "batch *shape"],  # noqa: F821
                      t: Float[Tensor, ""],  # noqa: F722
                      noise_predictor: NoisePredictor,
                      T: None | int = None,
                      ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        t = t.unsqueeze(0).expand(x.shape[0])  # [b]
        t_ = broadcast_from_below(t, x)  # [b *shape]
        sigma_t = self.noise_injector(t_, T)  # [b, *shape]
        calpha_t = self.scheduler.calpha(t_, T)  # [b, *shape]
        calpha_t_prev = self.scheduler.calpha(t_-1, T)  # [b, *shape]
        noise_pred = noise_predictor(x, t)
        x0_pred = ((x - noise_pred*torch.sqrt(1 - calpha_t)) /
                   torch.sqrt(calpha_t))
        x0_pred_direction = torch.sqrt(calpha_t_prev) * x0_pred
        xt_direction_factor = torch.relu(1 - calpha_t_prev - sigma_t**2)
        xt_direction = torch.sqrt(xt_direction_factor) * noise_pred
        random_noise = sigma_t * torch.randn_like(x)
        result = x0_pred_direction + xt_direction + random_noise
        return result

    def step_forward(self,
                     x: Float[Tensor, "batch *shape"],  # noqa: F821
                     t: Float[Tensor, ""],  # noqa: F722
                     T: None | int = None,
                     ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        t = t.unsqueeze(0).expand(x.shape[0])  # [b]
        t_ = broadcast_from_below(t, x)  # [b *shape]
        calpha_t = self.scheduler.calpha(t_, T)
        calpha_t_prev = self.scheduler.calpha(t_-1, T)
        noise = torch.randn_like(x)
        x_forward_mean = torch.sqrt(calpha_t / calpha_t_prev) * x
        x_forward_noise = (1 - calpha_t/calpha_t_prev) * noise
        x_forward = x_forward_mean + x_forward_noise
        return x_forward

    def noise_injector(self,
                       t: Float[Tensor, "..."],
                       T: None | int = None,
                       ) -> Float[Tensor, "..."]:
        raise NotImplementedError


class DDPMIntegrator(GeneralizedDDPMIntegrator):
    def noise_injector(self,
                       t: Float[Tensor, "..."],
                       T: None | int = None,
                       ) -> Float[Tensor, "..."]:
        calpha_t = self.scheduler.calpha(t, T)
        calpha_t_prev = self.scheduler.calpha(t-1, T)
        term1sq = (1 - calpha_t_prev)/(1 - calpha_t)
        term2sq = 1 - calpha_t/calpha_t_prev
        termsq = term1sq * term2sq
        term = torch.sqrt(termsq)
        return term


class DDIMIntegrator(GeneralizedDDPMIntegrator):
    def noise_injector(self,
                       t: Float[Tensor, "..."],
                       T: None | int = None,
                       ) -> Float[Tensor, "..."]:
        return 0 * t

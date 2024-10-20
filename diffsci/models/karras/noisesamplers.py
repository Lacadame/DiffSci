from typing import Any

import torch
from torch import Tensor
from jaxtyping import Float
from scipy.integrate import quad


class NoiseSampler(torch.nn.Module):
    def loss_weighing(self,
                      sigma: Float[Tensor, '...']
                      ) -> Float[Tensor, '...']:
        raise NotImplementedError

    def sample(self,
               shape: list[int]
               ) -> Float[Tensor, '*shape']:  # noqa: F821
        raise NotImplementedError


class EDMNoiseSampler(NoiseSampler):
    def __init__(self,
                 sigma_data: float = 0.5,
                 prior_mean: float = -1.2,
                 prior_std: float = 1.2):
        super().__init__()
        self.register_buffer("sigma_data", torch.tensor(sigma_data))
        self.register_buffer("prior_mean", torch.tensor(prior_mean))
        self.register_buffer("prior_std", torch.tensor(prior_std))

    def loss_weighting(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return (sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2)

    def sample(self,
               shape: list[int]
               ) -> Float[Tensor, '*shape']:  # noqa: F821
        white_noise = torch.randn(shape).to(self.prior_mean.device)
        logsigma = white_noise*self.prior_std + self.prior_mean
        sigma = torch.exp(logsigma)
        return sigma


class VPNoiseSampler(NoiseSampler):
    def __init__(self,
                 noise_scheduler: Any,  # TODO: put actual type
                 epsilon: float = 1e-3):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def loss_weighting(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return 1/(sigma**2)

    def sample(self,
               shape: list[int]
               ) -> Float[Tensor, '*shape']:  # noqa: F821
        t = torch.rand(shape).to(self.epsilon)
        t = t*(1-self.epsilon) + self.epsilon
        sigma = self.noise_scheduler.scheduler_fns.noise_fn(t)
        return sigma


class VENoiseSampler(NoiseSampler):
    def __init__(self,
                 sigma_min: float = 0.02,
                 sigma_max: float = 100):
        super().__init__()
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))

    def loss_weighting(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return 1/(sigma**2)

    def sample(self,
               shape: list[int]
               ) -> Float[Tensor, '*shape']:  # noqa: F821
        unif = torch.rand(shape).to(self.sigma_min.device)
        logsigma_min = torch.log(self.sigma_min)
        logsigma_max = torch.log(self.sigma_max)
        logsigma = logsigma_min + unif*(logsigma_max - logsigma_min)
        sigma = torch.exp(logsigma)
        return sigma


class BoundNoiseSampler(NoiseSampler):
    def __init__(self,
                 sigma_data: float = 0.5,
                 prior_mean: float = -1.2,
                 prior_std: float = 1.2,
                 R: float = 7,        # maybe use module.norm as a proxy for this when latent=True
                 T: float = 80.0,      # get this from max_level noise in module
                 delta: float = 0.5,
                 normalization: float = 2.0):      # This is a hack
        super().__init__()
        self.register_buffer("sigma_data", torch.tensor(sigma_data))
        self.register_buffer("prior_mean", torch.tensor(prior_mean))
        self.register_buffer("prior_std", torch.tensor(prior_std))
        self.register_buffer("R", torch.tensor(R))
        self.register_buffer("T", torch.tensor(T))
        self.register_buffer("delta", torch.tensor(delta))
        self.register_buffer("normalization", torch.tensor(normalization))

    def loss_weighting(self,
                       sigma: Float[Tensor, '...'],
                       ) -> Float[Tensor, '...']:
        
        # compute new weighting
        rev_t = self.T - sigma    # Only the EDM noise_fn is implemented
        R = self.R
        C = 6*(4*R**2 + sigma**2) * torch.exp(4*R**2 / sigma**2)
        inv_C = torch.where(C == float('inf'), torch.tensor(0.0), 1.0 / C)
        integral = []
        for i in range(rev_t.shape[0]):
            integral.append(compute_integral(rev_t[i], self.T, inv_C[i], self.delta))
        integral = torch.tensor(integral).to(rev_t)
        new_weight = 1/(4*(sigma**2)*self.delta) * torch.exp(-integral)

        # compute karras weighting
        karras_weight = (sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2)

        return karras_weight + self.normalization * new_weight

    def sample(self,
               shape: list[int]
               ) -> Float[Tensor, '*shape']:  # noqa: F821
        white_noise = torch.randn(shape).to(self.prior_mean.device)
        logsigma = white_noise*self.prior_std + self.prior_mean
        sigma = torch.exp(logsigma)
        return sigma


def integrand(rev_t, inv_C, T, delta=0.5):
    t = T - rev_t
    return 2 * inv_C * t * (1 - delta)


def compute_integral(t_initial, t_final, inv_C, delta=0.5):
    result, error = quad(integrand, t_initial, t_final, args=(inv_C, t_final, delta,))
    return result

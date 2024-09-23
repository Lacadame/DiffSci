from typing import Any

import torch
from torch import Tensor
from jaxtyping import Float


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
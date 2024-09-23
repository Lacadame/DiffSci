import torch
from torch import Tensor
from jaxtyping import Float

from . import schedulers


class KarrasPreconditioner(torch.nn.Module):
    def skip_scaling(self,
                     sigma: Float[Tensor, '...']
                     ) -> Float[Tensor, '...']:
        raise NotImplementedError

    def output_scaling(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        raise NotImplementedError

    def input_scaling(self,
                      sigma: Float[Tensor, '...']
                      ) -> Float[Tensor, '...']:
        raise NotImplementedError

    def noise_conditioner(self,
                          sigma: Float[Tensor, '...']
                          ) -> Float[Tensor, '...']:
        raise NotImplementedError


class EDMPreconditioner(KarrasPreconditioner):
    def __init__(self, sigma_data: float = 0.5):
        super().__init__()
        self.register_buffer("sigma_data", torch.tensor(sigma_data))

    def skip_scaling(self,
                     sigma: Float[Tensor, '...']
                     ) -> Float[Tensor, '...']:
        return self.sigma_data**2/(sigma**2 + self.sigma_data**2)

    def output_scaling(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return sigma*self.sigma_data/torch.sqrt(sigma**2 + self.sigma_data**2)

    def input_scaling(self,
                      sigma: Float[Tensor, '...']
                      ) -> Float[Tensor, '...']:
        return 1/torch.sqrt(sigma**2 + self.sigma_data**2)

    def noise_conditioner(self,
                          sigma: Float[Tensor, '...']
                          ) -> Float[Tensor, '...']:
        return 0.5*torch.log(sigma)


class VPPreconditioner(KarrasPreconditioner):
    def __init__(self,
                 scheduler: schedulers.Scheduler,
                 M: int = 1000,
                 ):
        super().__init__()
        self.scheduler = scheduler
        self.M = M

    def skip_scaling(self,
                     sigma: Float[Tensor, '...']
                     ) -> Float[Tensor, '...']:
        return 1 + 0.0*sigma

    def output_scaling(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return -sigma

    def input_scaling(self,
                      sigma: Float[Tensor, '...']
                      ) -> Float[Tensor, '...']:
        return 1/torch.sqrt(sigma**2 + 1.0)

    def noise_conditioner(self,
                          sigma: Float[Tensor, '...']
                          ) -> Float[Tensor, '...']:
        finv = self.scheduler.scheduler_fns.inverse_noise_fn
        return (self.M - 1) * finv(sigma)


class VEPreconditioner(KarrasPreconditioner):
    def __init__(self):
        super().__init__()

    def skip_scaling(self,
                     sigma: Float[Tensor, '...']
                     ) -> Float[Tensor, '...']:
        return 1 + 0.0*sigma

    def output_scaling(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return sigma

    def input_scaling(self,
                      sigma: Float[Tensor, '...']
                      ) -> Float[Tensor, '...']:
        return 1 + 0.0*sigma

    def noise_conditioner(self,
                          sigma: Float[Tensor, '...']
                          ) -> Float[Tensor, '...']:
        return torch.log(0.5*sigma)


class SR3Preconditioner(KarrasPreconditioner):       # for SR3 super resolution
    def __init__(self, sigma_data: float = 0.5):
        super().__init__()
        self.register_buffer("sigma_data", torch.tensor(sigma_data))

    def skip_scaling(self,
                     sigma: Float[Tensor, '...']
                     ) -> Float[Tensor, '...']:
        return self.sigma_data**2/(2*(sigma**2 + self.sigma_data**2))

    def output_scaling(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return sigma*self.sigma_data/(2*torch.sqrt(sigma**2 +
                                                   self.sigma_data**2))

    def input_scaling(self,
                      sigma: Float[Tensor, '...']
                      ) -> Float[Tensor, '...']:
        return 1/torch.sqrt(sigma**2 + self.sigma_data**2)

    def noise_conditioner(self,
                          sigma: Float[Tensor, '...']
                          ) -> Float[Tensor, '...']:
        return 0.5*torch.log(sigma)


class NullPreconditioner(KarrasPreconditioner):
    def __init__(self):
        super().__init__()

    def skip_scaling(self,
                     sigma: Float[Tensor, '...']
                     ) -> Float[Tensor, '...']:
        return 0.0*sigma

    def output_scaling(self,
                       sigma: Float[Tensor, '...']
                       ) -> Float[Tensor, '...']:
        return 1.0 + 0.0*sigma

    def input_scaling(self,
                      sigma: Float[Tensor, '...']
                      ) -> Float[Tensor, '...']:
        return 1.0 + 0.0*sigma

    def noise_conditioner(self,
                          sigma: Float[Tensor, '...']
                          ) -> Float[Tensor, '...']:
        return sigma

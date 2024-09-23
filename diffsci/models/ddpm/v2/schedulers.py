import math

import torch
from torch import Tensor
from jaxtyping import Float


class DDPMScheduler(torch.nn.Module):
    def __init__(self, T: int = 1000):
        super().__init__()
        self.T = T

    def calpha_norm(self,
                    s: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch"]:  # noqa: F821
        r"""
        Returns \bar{\alpha}(s)
        """
        raise NotImplementedError

    def calpha(self,
               t: Float[Tensor, "batch"],  # noqa: F821
               T: None | int = None,
               ) -> Float[Tensor, "batch"]:  # noqa: F821
        r"""
        Returns \bar{\alpha}_t = \bar{\alpha}(t/T)
        """
        if T is None:
            T = self.T
        return self.calpha_norm(t / T)

    def alpha(self,
              t: Float[Tensor, "batch"],  # noqa: F821
              T: None | int = None,
              ) -> Float[Tensor, "batch"]:  # noqa: F821
        r"""
        Returns \alpha_t = \alpha(t/T)
        """
        return self.calpha(t, T)/self.calpha(t-1, T)

    def beta(self,
             t: Float[Tensor, "batch"],  # noqa: F821
             T: None | int = None,
             ) -> Float[Tensor, "batch"]:  # noqa: F821
        return 1 - self.alpha(t, T)


class ClassicalDDPMScheduler(DDPMScheduler):
    def __init__(self,
                 beta1T: float = 20.0,
                 beta0: float = 1e-4,
                 T: int = 1000):
        super().__init__()
        self.beta1T = beta1T
        self.beta0 = beta0

    def calpha_norm(self,
                    s: Float[Tensor, "batch *shape"]  # noqa: F821
                    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        raise NotImplementedError

    def calpha(self,
               t: Float[Tensor, "batch"],  # noqa: F821
               T: None | int = None,
               ) -> Float[Tensor, "batch"]:  # noqa: F821
        r"""
        Returns \bar{\alpha}_t = \bar{\alpha}(t/T)
        """
        # t : (nbatch,)
        res = []
        for tt in t:
            tt = torch.round(tt).int().item()
            s = torch.arange(tt).to(t.device) + 1
            alphas = self.alpha(s, T)
            calphai = torch.exp(torch.sum(torch.log(alphas)))
            res.append(calphai)
        res = torch.stack(res, axis=0)
        res = res.reshape(t.shape)
        return res

    def alpha(self,
              t: Float[Tensor, "batch"],  # noqa: F821
              T: None | int = None,
              ) -> Float[Tensor, "batch"]:  # noqa: F821
        r"""
        Returns \alpha_t = \alpha(t/T)
        """
        return 1.0 - self.beta(t, T)

    def beta(self,
             t: Float[Tensor, "batch"],  # noqa: F821
             T: None | int = None,
             ) -> Float[Tensor, "batch"]:  # noqa: F821
        T = self.T if T is None else T
        s = (t-1)/(T-1)
        return self.beta0*(1 - s) + self.beta1T/T*s


class ExpDDPMScheduler(DDPMScheduler):
    def __init__(self,
                 beta_data: float = 19.9,
                 beta0: float = 1e-4,
                 T: int = 1000):
        super().__init__()
        self.beta_data = beta_data
        self.beta0 = beta0

    def calpha_norm(self,
                    s: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch"]:  # noqa: F821
        return torch.exp(-0.5*(self.beta_data * s**2 + self.beta0))


class CosineDDPMScheduler(DDPMScheduler):
    def __init__(self,
                 stabilizer: float = 0.008,
                 T: int = 1000):
        super().__init__()
        self.stabilizer = stabilizer
        self.f0 = math.cos((stabilizer/(1 + stabilizer) * math.pi/2))**2

    def calpha_norm(self,
                    s: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch"]:  # noqa: F821
        ft = torch.cos(((self.stabilizer + s)/(1 + self.stabilizer) *
                       math.pi/2))**2
        return ft/self.f0

import torch

from diffsci.torchutils import broadcast_from_below
from .sde_schedulers import SDEScheduler


class VEScheduler(SDEScheduler):

    def __init__(self, T=1, Tmin=1e-5):

        """
        Variance Exploding scheduler, given by the sde

        dXt = sigma(t) dWt

        Specific instantiations should come from subclassing
        this class and implementing the beta and betaint functions

        Parameters
        ----------
        scheduler_fn : VPBetaFunction
        """

        super().__init__(T=T, Tmin=Tmin)

    def mean(self, t, x):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number
        x : torch.Tensor of shape (nbatch, ...)
            Current sample
        Returns
        -------
        mean : torch.Tensor of shape (nbatch, ...)
        """

        t = broadcast_from_below(t, x)  # (nbatch, ...)
        return x + 0.0

    def std2_(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        std : torch.Tensor of shape (nbatch,)
        """

        return self.g2int(t)  # (nbatch,)

    def drift_term(self, t, x):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number
        x : torch.Tensor of shape (nbatch, ...)
            Current sample
        Returns
        -------
        drift : torch.Tensor of shape (nbatch, ...)
        """

        return 0.0*x  # (nbatch, ...)

    def diffusion_term(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        diffusion : torch.Tensor of shape (nbatch,)
        """

        return self.g(t)   # (nbatch,)

    def g(self, t):
        raise NotImplementedError

    def g2int(self, t):
        raise NotImplementedError


class VESchedulerSqrt(VEScheduler):
    """
        g(t) = sqrt(2 t)
    """
    def g(self, t):
        return torch.sqrt(2*t)

    def g2int(self, t):
        return t**2

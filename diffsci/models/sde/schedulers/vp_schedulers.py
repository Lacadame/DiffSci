import torch

from diffsci.torchutils import broadcast_from_below
from .sde_schedulers import SDEScheduler


# FIXME : Something is wrong here. Don't trust this code
class VPScheduler(SDEScheduler):

    def __init__(self, T=1, Tmin=1e-5):

        """
        Variance Preserving scheduler, given by the sde

        dXt = -0.5*beta(t)*Xt dt + sqrt(beta(t)) dWt

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
        return x*torch.exp(-0.5*self.betaint(t))

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

        return 1.0 - torch.exp(-self.betaint(t))

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

        t = broadcast_from_below(t, x)  # (nbatch, ...)
        return -0.5*self.beta(t)*x  # (nbatch, ...)

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

        return torch.sqrt(self.beta(t))   # (nbatch,)

    def beta(self, t):
        raise NotImplementedError

    def betaint(self, t):
        raise NotImplementedError


class VPSchedulerConstant(VPScheduler):

    def __init__(self, T=1, Tmin=1e-5, coef=1.0):
        super().__init__(T, Tmin)
        self.coef = coef

    def beta(self, t):
        return self.coef + 0.0*t

    def betaint(self, t):
        return self.coef*t


class VPSchedulerLinear(VPScheduler):

    def __init__(self, T=1, Tmin=1e-5, coef=1.0):
        super().__init__(T, Tmin)
        self.coef = coef

    def beta(self, t):
        return self.coef * t

    def betaint(self, t):
        return 0.5*self.coef*t**2


class VPSchedulerCustom(VPScheduler):

    def __init__(self, beta, betaint, T=1, Tmin=1e-5):

        """
        Parameters
        ----------
        beta : callable
            beta(t) function
        betaint : callable
            betaint(t) function
        """

        super().__init__(T, Tmin)
        self.beta = beta
        self.betaint = betaint

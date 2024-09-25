import torch

from diffsci.global_constants import SCHEDULER_VARIANCE_STABILIZER


class SDEScheduler(torch.nn.Module):

    def __init__(self, T=1, Tmin=1e-5, stabilizer=1e-2):

        """
        Abstract class for scheduler

        Parameters
        ----------
        T : float
            Final time of the SDE
        Tmin : float
            Minimum time of the SDE
        """

        super().__init__()
        self.register_buffer("T", torch.tensor(T))
        self.register_buffer("Tmin", torch.tensor(Tmin))
        self.register_buffer("stabilizer",
                             torch.tensor(SCHEDULER_VARIANCE_STABILIZER))

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

        raise NotImplementedError()

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

        raise NotImplementedError()

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

        raise NotImplementedError()

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

        raise NotImplementedError()

    def std2(self, t):
        return self.std2_(t) + self.stabilizer

    def std(self, t):
        return torch.sqrt(self.std2(t))

    def sample_time(self, nbatch):

        """
        Parameters
        ----------
        nbatch : int
            Number of samples to generate

        Returns
        -------
        t : torch.Tensor of shape (nbatch,)
        """

        u = torch.rand(nbatch).to(self.T)
        t = (self.T - self.Tmin)*u + self.Tmin
        return t

    def sample_fully_noised(self, nbatch, xshape, device=None):
        """
        Parameters
        ----------
        nbatch : int
            Number of samples to be generated
        xshape : list[int]
            Shape of the samples to be generated

        Returns
        -------
        xt : torch.Tensor of shape (nbatch, *xshape)
            Fully noised samples
        """
        t = torch.ones([nbatch]).to(self.T)*self.T
        std = self.std(t)
        tshape = [nbatch] + [1]*len(xshape)
        noise = torch.randn([nbatch] + xshape).to(self.T)
        xt = noise*(std.view(tshape))
        if device:
            xt = xt.to(device)
        return xt

    def sample_noise_at_t(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor of shape [nbatch]
            Time to be samples
        x : torch.Tensor of shape [nbatch, *xshape]
            Initial unnoised samples
        
        Returns
        -------
        xt : torch.Tensor of shape [nbatch, *xshape]
            Noised samples
        """
        device = x.device
        nbatch = x.shape[0]
        mean = self.mean(t, x)
        std = self.std(t)
        xshape = list(mean.shape)
        noise = torch.randn([nbatch] + xshape).to(self.T)
        xt = noise*std + mean
        xt = xt.to(device)
        return xt

import math

import torch


class DDPMScheduler(torch.nn.Module):

    def __init__(self, beta0=1e-4, beta1=2*1e-2, T=1000):

        """
        Scheduler for the DDPM framework in
        "Denoising Diffusion Probabilistic Models" by Ho et al.

        Parameters
        ----------
        beta0 : float
            Initial value of beta
        beta1 : float
            Final value of beta
        T : int
            Number of iterations to reach beta1
        """

        super().__init__()
        self.beta0 = beta0
        self.beta1 = beta1
        self.T = T
        self.dummy_param = torch.nn.Parameter(torch.zeros(size=[]))

    def beta(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        beta : torch.Tensor of shape (nbatch,)
        """

        # t : (nbatch,)
        s = (t-1)/(self.T-1)
        return self.beta0*(1 - s) + self.beta1*s

    def alpha(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        alpha : torch.Tensor of shape (nbatch,), equals 1 - beta
        """

        # t : (nbatch,)
        return 1.0 - self.beta(t)

    def sigma(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        sigma : torch.Tensor of shape (nbatch,), equals sqrt(beta)
        """

        # t : (nbatch,)
        return torch.sqrt(self.beta(t))

    def calpha(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        calpha : torch.Tensor of shape (nbatch,), equals cumprod(alpha)
        """

        # t : (nbatch,)
        res = []
        for tt in t:
            s = torch.arange(tt).to(t.device) + 1
            alphas = self.alpha(s)
            calphai = torch.exp(torch.sum(torch.log(alphas)))
            res.append(calphai)
        return torch.stack(res, axis=0)

    def sample(self, nbatch):

        """
        Parameters
        ----------
        nbatch : int
            Number of samples to generate

        Returns
        -------
        t : torch.Tensor of shape (nbatch,)
        """

        device = self.dummy_param.device
        return torch.randint(1, self.T+1, [nbatch]).to(device)

    def schedule(self, reverse=False):
        """
        Parameters
        ----------
        reverse : bool

        Returns
        -------
        schedule
        """
        schedule = (torch.arange(self.T)+1)
        if reversed:
            schedule = reversed(schedule)
        return schedule


class NormalDDPMScheduler(torch.nn.Module):

    def __init__(self, C=20.0, T=1000, T_orig=1000):

        """
        EXPERIMENTAL

        Parameters
        ----------
        C : float
        T : int
        """

        super().__init__()
        self.C = C
        self.T = T
        self.T_orig = T_orig
        self.scale_ratio = self.T_orig/self.T
        self.dummy_param = torch.nn.Parameter(torch.zeros(size=[]))

    def beta(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        beta : torch.Tensor of shape (nbatch,)
        """

        # t : (nbatch,)
        return 1-self.alpha(t)

    def alpha(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        alpha : torch.Tensor of shape (nbatch,), equals 1 - beta
        """

        # t : (nbatch,)
        calpha = self.calpha(t)
        calpha_prev = self.calpha(t-1)
        return calpha/calpha_prev

    def sigma(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        sigma : torch.Tensor of shape (nbatch,), equals sqrt(beta)
        """

        # t : (nbatch,)
        return torch.sqrt(self.beta(t))

    def calpha(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        calpha : torch.Tensor of shape (nbatch,), equals cumprod(alpha)
        """
        s = t/self.T_orig
        return torch.exp(-self.C/2*s**2)

    def sample(self, nbatch):

        """
        Parameters
        ----------
        nbatch : int
            Number of samples to generate

        Returns
        -------
        t : torch.Tensor of shape (nbatch,)
        """

        device = self.dummy_param.device
        t = torch.randint(1, self.T+1, [nbatch]).to(device)
        t = t*self.scale_ratio
        return t

    def schedule(self, reverse=False):
        """
        Parameters
        ----------
        reverse : bool

        Returns
        -------
        schedule
        """
        schedule = (torch.arange(self.T)+1)*self.scale_ratio
        if reverse:
            schedule = reversed(schedule)
        return schedule


class DDPMCossineScheduler(torch.nn.Module):

    def __init__(self, s=0.008, T=1000):

        """
        Cossine Scheduler for the DDPM framework in
        "Improved Denoising Diffusion Probabilistic Models" by
        Alex Nichol and Prafulla Dhariwal.

        Parameters
        ----------
        s : float
            Hyper parameter to avoid distortions at t=0 and t=T
        T : int
            Number of iterations to reach betaT

        """

        super().__init__()
        self.betat = torch.zeros(size=(1, T))
        self.T = T
        self.s = s
        self.dummy_param = torch.nn.Parameter(torch.zeros(size=[]))

    def f(self, t):
        """
        Function used to cumpute aplha_bar. Presented in Improved Denoising
        Diffusion Probabilistic Models" by Alex Nichol and Prafulla Dhariwal.

        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number
        """

        return torch.square(
                torch.cos((t/self.T+self.s)/(1+self.s)*math.pi/2)
               )

    def calpha(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        calpha : torch.Tensor of shape (nbatch,), equals bar(alpha)
        on the paper
        """

        # t : (nbatch,)
        fs = self.f(t)
        f_0 = fs[0]
        return fs/f_0

    def beta(self, t):

        """
        Compute beta_t for the cossine schedule

        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        beta : torch.Tensor of shape (nbatch,)
        """
        a = self.calpha(t)
        res = [min(1-a[0], 0.999)]

        for i, ii in enumerate(a[1:]):
            betai = min(1-ii/a[i], 0.999)
            res.append(betai)

        return torch.stack(res, axis=0)

    def sigma(self, t):

        """
        Parameters
        ----------
        t : torch.Tensor of shape (nbatch,)
            Current iteration number

        Returns
        -------
        sigma : torch.Tensor of shape (nbatch,), equals sqrt(beta)
        """

        # t : (nbatch,)
        return torch.sqrt(self.beta(t))

    def sample(self, nbatch):

        """
        Parameters
        ----------
        nbatch : int
            Number of samples to generate

        Returns
        -------
        t : torch.Tensor of shape (nbatch,)
        """

        device = self.dummy_param.device
        return torch.randint(1, self.T+1, [nbatch]).to(device)

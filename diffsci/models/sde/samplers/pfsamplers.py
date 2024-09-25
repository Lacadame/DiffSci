import warnings

import torch

from porenet.torchutils import broadcast_from_below


class PFSampler(object):
    def __init__(self, model, scheduler, shape):
        """
        Abstract class for sampling from, and flowing foward from,
        the probability flow ODE
        dx_t/dt = (-f(x_t, t) - 1/2 g(t)^2 \nabla \log p_t(x_t)),
        where g(t) and f(x_t, t) are provided by the scheduler, and
        \nabla \log p_t(x_t) is estimated by the model

        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape (B, [shape]), the original noise
                t : torch.Tensor of shape (B,)
                y : torch.Tensor of shape (B, [yshape]), the conditional data
                and as output
                torch.Tensor of shape (B, [shape])
        scheduler : SDEcheduler
        self.shape : list[int]
            The non-batch shape of x.
        device : torch.device
        """
        self.model = model
        self.scheduler = scheduler
        self.shape = shape

    def sample(self, y=None, nsamples=1, device=None, nsteps=100):
        if self.shape is None:
            raise ValueError("Cannot sample without specifying shape")
        x = self.scheduler.sample_fully_noised(nsamples, self.shape)
        if device is not None:
            x = x.to(device)
        return self.forward(x, y=y, nsteps=nsteps)

    def forward(self, x, y=None, nsteps=100):
        """
        Parameters
        ----------
        y : None or torch.Tensor of shape (...) (no batch dimension)
            depending on whether we are dealing with conditional
            or unconditional sampling
        nsteps : int
            Number of steps to take in the Euler-Maruyama scheme
        nsamples : int | None
            Number of samples. If None, assume single sample to be squeezed

        Returns
        -------
        x : torch.Tensor of shape [nsamples, *shape]
        """
        self.model.eval()
        if y is not None:
            y = y.unsqueeze(0)  # Add batch dimension, shape [1, [yshape]]
        nsamples = x.shape[0]

        deltastep = (self.scheduler.T - self.scheduler.Tmin)/nsteps
        with torch.no_grad():
            for step in range(nsteps):
                x = self.propagate_backward(x, step, deltastep, y,
                                            nsamples)
        if nsamples is None:
            x = x.squeeze(0)
        return x

    def pf_rhs(self, x, t, y, batch_shape):
        """
        Calculates f(x, t) - 1/4*g(t)^2*\nabla\log p_t(x)

        Parameters
        ----------
        x : torch.Tensor of shape [batch_shape, *shape]
        t : torch.Tensor of shape [batch_shape]
        y : None or torch.Tensor of shape [batch_shape, *yshape]
            The conditional data
        batch_shape : int
        shape : list[int]
            The non-batch shape of x
        """
        t_ = t*(torch.ones([batch_shape]).to(x))  # [1]
        drift = self.scheduler.drift_term(t_, x)  # Shape [1, [shape]]
        diffusion = self.scheduler.diffusion_term(t_)  # Shape [1]
        diffusion = broadcast_from_below(diffusion, x)
        if y is not None:  # Conditional sampling
            score = self.model(x, t_, y)  # Shape [1, [shape]]
        else:  # Unconditional sampling
            score = self.model(x, t_)  # Shape [1, [shape]]
        reverse_drift = (drift - 0.5*diffusion**2*score)
        return reverse_drift

    def propagate_backward(self, x, step, deltastep, y,
                           batch_shape):
        """
        Override this class with your own explicit RK method
        """
        raise NotImplementedError


class EulerPFSampler(PFSampler):
    def propagate_backward(self, x, step, deltastep, y,
                           batch_shape):
        t = self.scheduler.T - step*deltastep
        pf_rhs = self.pf_rhs(x, t, y,
                             batch_shape)
        x = x - pf_rhs*deltastep  # Shape [1, [shape]]
        return x


class HeunPFSampler(PFSampler):
    def propagate_backward(self, x, step, deltastep, y,
                           batch_shape):
        t = self.scheduler.T - step*deltastep
        pf_rhs = self.pf_rhs(x, t, y,
                             batch_shape)
        x_h = x - pf_rhs*deltastep  # Shape [1, [shape]]
        t_h = t - deltastep
        pf_rhs_h = self.pf_rhs(x_h, t_h, y,
                               batch_shape)
        x = x - 0.5*(pf_rhs + pf_rhs_h)*deltastep
        return x


class PFSamplerOld(object):
    def __init__(self, model, scheduler, shape, device="cpu"):
        """
        Abstract class for sampling from, and flowing foward from,
        the probability flow ODE
        dx_t/dt = (-f(x_t, t) - 1/2 g(t)^2 \nabla \log p_t(x_t)),
        where g(t) and f(x_t, t) are provided by the scheduler, and
        \nabla \log p_t(x_t) is estimated by the model

        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape (B, [shape]), the original noise
                t : torch.Tensor of shape (B,)
                y : torch.Tensor of shape (B, [yshape]), the conditional data
                and as output
                torch.Tensor of shape (B, [shape])
        scheduler : SDEcheduler
        self.shape : list[int]
            The non-batch shape of x.
        device : torch.device
        """
        warning_string = \
            "Deprecated. Use DDPMSampler"
        warnings.warn(warning_string)
        self.model = model
        self.scheduler = scheduler
        self.shape = shape
        self.device = device

    def sample(self, y=None, nsteps=100, nsamples=None):
        """
        Parameters
        ----------
        y : None or torch.Tensor of shape (...) (no batch dimension)
            depending on whether we are dealing with conditional
            or unconditional sampling
        nsteps : int
            Number of steps to take in the Euler-Maruyama scheme
        nsamples : int | None
            Number of samples. If None, assume single sample to be squeezed

        Returns
        -------
        x : torch.Tensor of shape [nsamples, *shape]
        """
        self.model.eval()
        shape = self.shape
        if y is not None:
            y = y.to(self.device)
            y = y.unsqueeze(0)  # Add batch dimension, shape [1, ...]
        batch_shape = 1 if nsamples is None else nsamples

        x = self.scheduler.sample_fully_noised(batch_shape, shape)
        x = x.to(self.device)

        deltastep = (self.scheduler.T - self.scheduler.Tmin)/nsteps
        with torch.no_grad():
            for step in range(nsteps):
                x = self.propagate_backward(x, step, deltastep, y,
                                            batch_shape, shape)
        if nsamples is None:
            x = x.squeeze(0)
        return x

    def pf_rhs(self, x, t, y, batch_shape, shape):
        """
        Calculates f(x, t) - 1/4*g(t)^2*\nabla\log p_t(x)

        Parameters
        ----------
        x : torch.Tensor of shape [batch_shape, *shape]
        t : torch.Tensor of shape [batch_shape]
        y : None or torch.Tensor of shape [batch_shape, *yshape]
            The conditional data
        batch_shape : int
        shape : list[int]
            The non-batch shape of x
        """
        t_ = t*torch.ones([batch_shape]).to(self.device)  # [1]
        drift = self.scheduler.drift_term(t_, x)  # Shape [1, [shape]]
        diffusion = self.scheduler.diffusion_term(t_)  # Shape [1]
        diffusion_shape = [batch_shape] + [1]*len(shape)
        diffusion = diffusion.view(diffusion_shape)  # Shape [1, ...]
        if y is not None:  # Conditional sampling
            score = self.model(x, t_, y)  # Shape [1, [shape]]
        else:  # Unconditional sampling
            score = self.model(x, t_)  # Shape [1, [shape]]
        reverse_drift = (drift - 0.5*diffusion**2*score)
        return reverse_drift

    def propagate_backward(self, x, step, deltastep, y,
                           batch_shape, shape):
        """
        Override this class with your own explicit RK method
        """
        raise NotImplementedError


class EulerPFSamplerOld(PFSamplerOld):
    def propagate_backward(self, x, step, deltastep, y,
                           batch_shape, shape):
        t = self.scheduler.T - step*deltastep
        pf_rhs = self.pf_rhs(x, t, y,
                             batch_shape, shape)
        x = x - pf_rhs*deltastep  # Shape [1, [shape]]
        return x


class HeunPFSamplerOld(PFSamplerOld):
    def propagate_backward(self, x, step, deltastep, y,
                           batch_shape, shape):
        t = self.scheduler.T - step*deltastep
        pf_rhs = self.pf_rhs(x, t, y,
                             batch_shape, shape)
        x_h = x - pf_rhs*deltastep  # Shape [1, [shape]]
        t_h = t - deltastep
        pf_rhs_h = self.pf_rhs(x_h, t_h, y,
                               batch_shape, shape)
        x = x - 0.5*(pf_rhs + pf_rhs_h)*deltastep
        return x

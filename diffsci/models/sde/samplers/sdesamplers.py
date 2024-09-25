import math
import warnings

import torch

from diffsci.torchutils import broadcast_from_below


class EulerMaruyamaSampler(object):

    def __init__(self, model, scheduler, shape=None):

        """
        Predict a sample, from a trained model,
        according to the DDPM framework in
        "Score-Based Generative Modeling
        Through Stochastic Differential Equations"
        by Song et al.

        Parameters
        ----------
        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape (B, C, H, W), the original noise
                t : torch.Tensor of shape (B,)
                y : torch.Tensor of shape (B, ...), the conditional data
                and as output
                torch.Tensor of shape (B, C, H, W)
        scheduler : SDEcheduler
        self.shape : list[int] of format [C, H, W] or None
            If shape equals none, assume only conditional sampling,
            y of shape [C, H, W] in self.sample,
            and shape = y.shape.
            It is recommended that shape is passed explicitly.
        device : torch.device
        """

        self.model = model
        self.scheduler = scheduler
        self.shape = shape

    def sample(self, y=None, nsamples=1, nsteps=500, device=None):
        if self.shape is None:
            raise ValueError("Cannot sample without specifying shape")
        x = self.scheduler.sample_fully_noised(nsamples, self.shape)
        if device is not None:
            x = x.to(device)
        return self.forward(x, y=y, nsteps=nsteps)

    def forward(self, x, y=None, nsteps=500):

        """
        Parameters
        ----------
        y : None or torch.tensor of shape (...) (no batch dimension)
            depending on whether we are dealing with conditional or
            unconditional sampling
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
        for step in range(nsteps):
            t = self.scheduler.T - step*deltastep
            t_ = t*(torch.ones([nsamples]).to(self.x))  # [1]
            drift = self.scheduler.drift_term(t_, x)  # Shape [1, C, H, W]
            diffusion = self.scheduler.diffusion_term(t_)  # Shape [1]
            diffusion = broadcast_from_below(diffusion, x)
            if step == nsteps - 1:
                z = torch.randn_like(x)  # Shape [1, C, H, W]
                noise = diffusion*z*math.sqrt(deltastep)  # Shape [1, C, H, W]
            else:
                noise = 0.0
            if y is not None:  # Conditional sampling
                score = self.model(x, t_, y)  # Shape [1, C, H, W]
            else:  # Unconditional sampling
                score = self.model(x, t_)  # Shape [1, C, H, W]
            reverse_drift = (drift - diffusion**2*score)*deltastep
            x = x - reverse_drift - noise  # Shape [1, C, H, W]
        return x


class EulerMaruyamaSamplerOld(object):

    def __init__(self, model, scheduler, shape=None, device="cpu"):

        """
        Predict a sample, from a trained model,
        according to the DDPM framework in
        "Score-Based Generative Modeling
        Through Stochastic Differential Equations"
        by Song et al.

        Parameters
        ----------
        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape (B, C, H, W), the original noise
                t : torch.Tensor of shape (B,)
                y : torch.Tensor of shape (B, ...), the conditional data
                and as output
                torch.Tensor of shape (B, C, H, W)
        scheduler : SDEcheduler
        self.shape : list[int] of format [C, H, W] or None
            If shape equals none, assume only conditional sampling,
            y of shape [C, H, W] in self.sample,
            and shape = y.shape.
            It is recommended that shape is passed explicitly.
        device : torch.device
        """
        warning_string = \
            "Deprecated. Use DDPMSampler"
        warnings.warn(warning_string)
        self.model = model
        self.scheduler = scheduler
        self.shape = shape
        self.device = device

    def sample(self, y=None, nsteps=500, nsamples=None):

        """
        Parameters
        ----------
        y : None or torch.tensor of shape (...) (no batch dimension)
            depending on whether we are dealing with conditional or
            unconditional sampling
        nsteps : int
            Number of steps to take in the Euler-Maruyama scheme
        nsamples : int | None
            Number of samples. If None, assume single sample to be squeezed
        Returns
        -------
        x : torch.Tensor of shape [nsamples, *shape]
        """

        self.model.eval()
        if self.shape is None:
            if y is None or nsamples is not None:
                raise ValueError("Can't infer shape")
            else:
                shape = list(y.shape)
        else:
            shape = self.shape
        if y is not None:
            y = y.to(self.device)
            y = y.unsqueeze(0)  # Add batch dimension, shape [1, C, H, W]
        batch_shape = 1 if nsamples is None else nsamples

        # [1, C, H, W]
        x = self.scheduler.sample_fully_noised(batch_shape, shape)
        x = x.to(self.device)

        deltastep = (self.scheduler.T - self.scheduler.Tmin)/nsteps
        for step in range(nsteps):
            t = self.scheduler.T - step*deltastep
            t_ = t*torch.ones([batch_shape]).to(self.device)  # [1]
            drift = self.scheduler.drift_term(t_, x)  # Shape [1, C, H, W]
            diffusion = self.scheduler.diffusion_term(t_)  # Shape [1]
            diffusion_shape = [batch_shape] + [1]*len(shape)
            diffusion = diffusion.view(diffusion_shape)
            if step == nsteps - 1:
                z = torch.randn_like(x)  # Shape [1, C, H, W]
                noise = diffusion*z*math.sqrt(deltastep)  # Shape [1, C, H, W]
            else:
                noise = 0.0
            if y is not None:  # Conditional sampling
                score = self.model(x, t_, y)  # Shape [1, C, H, W]
            else:  # Unconditional sampling
                score = self.model(x, t_)  # Shape [1, C, H, W]
            reverse_drift = (drift - diffusion**2*score)*deltastep
            x = x - reverse_drift - noise  # Shape [1, C, H, W]
        if nsamples is None:
            x = x.squeeze(0)
        return x

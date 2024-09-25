import warnings

import torch

from porenet.torchutils import broadcast_from_below


class DDPMSampler(torch.nn.Module):

    def __init__(self, model, scheduler, shape=None, noise_type=1):

        """
        Predict a sample, from a trained model, according to the DDPM framework
        in "Denoising Diffusion Probabilistic Models" by Ho et al.


        Parameters
        ----------
        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape [B, [shape]], the original noise
                t : torch.Tensor of shape [B]
                y : torch.Tensor of shape [B, [yshape]], the conditional data
                and as output
                torch.Tensor of shape [B, [shape]]
        scheduler : DDPMScheduler
        self.shape : list[int]
        device : torch.device
        """
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.shape = shape
        self.noise_type = noise_type

    def sample(self, y=None, nsamples=1, device="cpu"):
        if self.shape is None:
            raise ValueError("Cannot sample without specifying shape")
        x = torch.randn([nsamples]+self.shape).to(device)
        return self.backward(x, y=y)

    def backward(self, x, y=None):

        """
        Parameters
        ----------
        x : torch.Tensor of shape [B, [shape]]
        y : None or torch.nn.Module of shape [B, [yshape]]
            depending on whether we are dealing with conditional or
            unconditional sampling

        Returns
        -------
        x : torch.nn.Module of shape (batch, [shape])
        """

        self.model.eval()
        if y is not None:
            y = y.unsqueeze(0)  # Add batch dimension, shape [1, [yshape]]
        print(x.shape)
        nsamples = x.shape[0]
        with torch.no_grad():
            for t in self.scheduler.schedule(reverse=True):
                # Shape [nsamples]
                t_ = t*torch.ones([nsamples]).to(x)
                calpha = self.scheduler.calpha(t_)  # Shape [nsamples]
                alpha = self.scheduler.alpha(t_)  # Shape [nsamples]
                calpha = broadcast_from_below(calpha, x)
                alpha = broadcast_from_below(alpha, x)
                if self.noise_type == 1:
                    sigma = torch.sqrt(1-alpha)  # Shape [nsamples]
                elif self.noise_type == 2:
                    calpha_prev = calpha/alpha
                    sigma = torch.sqrt((1-alpha)*(1-calpha_prev)/(1-calpha))
                else:
                    sigma = 0.0
                if y is not None:  # Conditional sampling
                    score = self.model(x, t_, y)  # Shape [nsamples, [shape]]
                else:  # Unconditional sampling
                    score = self.model(x, t_)  # Shape [nsamples, [shape]]
                if t > 1:
                    z = torch.randn_like(x)  # Shape [nsamples, [shape]]
                else:
                    z = torch.zeros_like(x)  # Shape [nsamples, [shape]]
                # print(x)
                # print(z)
                # print(sigma)
                # print(alpha)
                # print(score)
                x = 1/torch.sqrt(alpha)*(
                    x - (1-alpha)/torch.sqrt(1-calpha)*score
                    ) + sigma*z  # Shape [nsamples, [shape]]
                # print(x)
                # raise NotImplementedError
        return x

    def apply_noise(self, x, t):
        """
        Parameters
        ----------
        x : torch.Tensor of shape [B, [shape]]
        t : torch.Tensor of shape [B]
        """
        calpha = self.scheduler.calpha(t)  # [nbatch]
        calpha = broadcast_from_below(calpha, x)
        noise = torch.randn_like(x)  # [nbatch, shape]
        x_noised = torch.sqrt(calpha)*x + torch.sqrt(1-calpha)*noise
        return x_noised


class DDIMSampler(torch.nn.Module):

    def __init__(self, model, scheduler, shape=None, noise_type=0):

        """
        Predict a sample, from a trained model, according to the DDIM framework
        in "Denoising Diffusion Implicit Models" by Song et al.


        Parameters
        ----------
        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape [B, [shape]], the original noise
                t : torch.Tensor of shape [B]
                y : torch.Tensor of shape [B, [yshape]], the conditional data
                and as output
                torch.Tensor of shape [B, [shape]]
        scheduler : DDPMScheduler
        self.shape : list[int]
        device : torch.device
        """
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.shape = shape
        self.noise_type = noise_type

    def sample(self, y=None, nsamples=1, device="cpu"):
        if self.shape is None:
            raise ValueError("Cannot sample without specifying shape")
        x = torch.randn([nsamples]+self.shape).to(device)
        return self.backward(x, y=y)

    def backward(self, x, y=None):

        """
        Parameters
        ----------
        x : torch.Tensor of shape [B, [shape]]
        y : None or torch.nn.Module of shape [B, [yshape]]
            depending on whether we are dealing with conditional or
            unconditional sampling

        Returns
        -------
        x : torch.nn.Module of shape (batch, [shape])
        """

        self.model.eval()
        if y is not None:
            y = y.unsqueeze(0)  # Add batch dimension, shape [1, [yshape]]
        nsamples = x.shape[0]
        with torch.no_grad():
            for t in self.scheduler.schedule(reverse=True):
                # Shape [nsamples]
                t_ = t*torch.ones([nsamples]).to(x)
                calpha = self.scheduler.calpha(t_)  # Shape [nsamples]
                alpha = self.scheduler.alpha(t_)  # Shape [nsamples]
                calpha = broadcast_from_below(calpha, x)
                alpha = broadcast_from_below(alpha, x)
                calpha_prev = calpha/alpha
                if self.noise_type == 1:
                    sigma = torch.sqrt(1-alpha)  # Shape [nsamples]
                elif self.noise_type == 2:
                    sigma = torch.sqrt((1-alpha)*(1-calpha_prev)/(1-calpha))
                else:  # Implicit model
                    sigma = 0.0
                if y is not None:  # Conditional sampling
                    score = self.model(x, t_, y)  # Shape [nsamples, [shape]]
                else:  # Unconditional sampling
                    score = self.model(x, t_)  # Shape [nsamples, [shape]]
                if t > 1:
                    z = torch.randn_like(x)  # Shape [nsamples, [shape]]
                else:
                    z = torch.zeros_like(x)  # Shape [nsamples, [shape]]
                predicted_term = 1/torch.sqrt(alpha)*(
                    x - torch.sqrt(1-calpha)*score
                )
                pointing_term = torch.sqrt(1 - calpha_prev - sigma**2)*score
                noise_term = sigma*z
                x = predicted_term + pointing_term + noise_term
        return x

    def apply_noise(self, x, t):
        """
        Parameters
        ----------
        x : torch.Tensor of shape [B, [shape]]
        t : torch.Tensor of shape [B]
        """
        calpha = self.scheduler.calpha(t)  # [nbatch]
        calpha = broadcast_from_below(calpha, x)
        noise = torch.randn_like(x)  # [nbatch, shape]
        x_noised = torch.sqrt(calpha)*x + torch.sqrt(1-calpha)*noise
        return x_noised


class DDPMSamplerOld(object):

    def __init__(self, model, scheduler, shape=None, device="cpu"):

        """
        Predict a sample, from a trained model, according to the DDPM framework
        in "Denoising Diffusion Probabilistic Models" by Ho et al.


        Parameters
        ----------
        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape (B, C, H, W), the original noise
                t : torch.Tensor of shape (B,)
                y : torch.Tensor of shape (B, ...), the conditional data
                and as output
                torch.Tensor of shape (B, C, H, W)
        scheduler : DDPMScheduler
        self.shape : list[int] of format [C, H, W] or None
            If shape equals none, assume only conditional sampling, y of shape
            [C, H, W] in self.sample, and shape = y.shape. It is recommended
            that shape is passed explicitly.
        device : torch.device
        """
        warning_string = \
            "Deprecated. Use DDPMSampler"
        warnings.warn(warning_string)
        self.model = model
        self.scheduler = scheduler
        self.shape = shape
        self.device = device

    def sample(self, y=None):

        """
        Parameters
        ----------
        y : None or torch.nn.Module of shape (...) (no batch dimension)
            depending on whether we are dealing with conditional or
            unconditional sampling

        Returns
        -------
        x : torch.nn.Module of shape (..., C, H, W)
        """

        self.model.eval()
        if self.shape is None:
            if y is None:
                raise ValueError(
                    "Cannot infer shape for unconditional sampling."
                )
            else:
                shape = list(y.shape)
        else:
            shape = self.shape
        if y is not None:
            y = y.to(self.device)
            y = y.unsqueeze(0)  # Add batch dimension, shape [1, C, H, W]
        batched_shape = [1] + shape

        # Sample from random normal, shape [1, C, H, W]
        x = torch.randn(*batched_shape).to(self.device)

        with torch.no_grad():
            for t in reversed(torch.arange(self.scheduler.T)+1):
                t_ = t*torch.ones([1]).to(self.device)  # Shape [1]
                calpha = self.scheduler.calpha(t_)  # Shape [1]
                alpha = self.scheduler.alpha(t_)  # Shape [1]
                sigma = torch.sqrt(1-alpha)  # Shape [1]
                if y is not None:  # Conditional sampling
                    score = self.model(x, t_, y)  # Shape [1, C, H, W]
                else:  # Unconditional sampling
                    score = self.model(x, t_)  # Shape [1, C, H, W]
                if t > 1:
                    z = torch.randn_like(x)  # Shape [1, C, H, W]
                else:
                    z = torch.zeros_like(x)  # Shape [1, C, H, W]
                x = 1/torch.sqrt(alpha)*(
                    x - (1-alpha)/torch.sqrt(1-calpha)*score
                    ) + sigma*z  # Shape [1, C, H, W]

        return x[0]

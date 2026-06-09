from typing import Literal, Callable, Any
from jaxtyping import Float
from torch import Tensor

import warnings
import math

import torch
import torch.nn as nn
import numpy as np
import lightning

from diffsci2.torchutils import broadcast_from_below, dict_unsqueeze, dict_to
from diffsci2.models.aux_scripts import DimensionAgnosticBatchNorm, ConstantBatchNorm, IdentityBatchNorm
from diffsci2.models.karras.mcmc import get_stepper, MCMCStepper


SampleType = Float[Tensor, "batch *shape"]
ConditionType = Float[Tensor, "batch *yshape"]
TimeType = Float[Tensor, "batch"]


class SIScheduler(object):
    """
    Stochastic Interpolant Scheduler.

    Defines the interpolation between data (t=0) and noise (t=1) via:
        x_t = alpha(t) * x_0 + sigma(t) * epsilon

    Can operate in two modes:
    - Normalized mode: t in [0, 1], with sigma_fn mapping to actual noise levels
    - Sigma-space mode: t IS sigma directly, t in [sigma_min, sigma_max]
    """

    def __init__(
        self,
        alpha_fn: Callable[[float], float],
        sigma_fn: Callable[[float], float],
        alpha_fn_dot: Callable[[float], float],
        sigma_fn_dot: Callable[[float], float],
        sigma_fn_inv: Callable[[float], float],
        time_domain: Literal['normalized', 'sigma'] = 'normalized',
        sigma_min: float | None = None,
        sigma_max: float | None = None
    ):
        self.alpha_fn = alpha_fn
        self.sigma_fn = sigma_fn
        self.alpha_fn_dot = alpha_fn_dot
        self.sigma_fn_dot = sigma_fn_dot
        self.sigma_fn_inv = sigma_fn_inv
        self.time_domain = time_domain
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @classmethod
    def linear(cls):
        return cls(
            alpha_fn=lambda t: 1 - t,
            sigma_fn=lambda t: t,
            alpha_fn_dot=lambda t: -1 * torch.ones_like(t),
            sigma_fn_dot=lambda t: torch.ones_like(t),
            sigma_fn_inv=lambda s: s,
        )

    @classmethod
    def cosine(cls):
        return cls(
            alpha_fn=lambda t: torch.cos(t * np.pi / 2),
            sigma_fn=lambda t: torch.sin(t * np.pi / 2),
            alpha_fn_dot=lambda t: -1 * torch.pi / 2 * torch.sin(t * np.pi / 2),
            sigma_fn_dot=lambda t: torch.pi / 2 * torch.cos(t * np.pi / 2),
            sigma_fn_inv=lambda s: (2 / np.pi) * torch.arcsin(s),
        )

    @classmethod
    def finterpolation(
        cls,
        f: Callable[[float], float],
        finv: Callable[[float], float],
        fdot: Callable[[float], float],
        sigma_min: float,
        sigma_max: float
    ):
        def sigma_fn(t):
            interpolated_finv_sigma = (1 - t) * finv(sigma_min) + t * finv(sigma_max)
            return f(interpolated_finv_sigma)

        def sigma_fn_inv(s):
            return (finv(s) - finv(sigma_min)) / (finv(sigma_max) - finv(sigma_min))

        def sigma_fn_dot(t):
            interpolated_finv_sigma = (1 - t) * finv(sigma_min) + t * finv(sigma_max)
            return fdot(interpolated_finv_sigma) * (finv(sigma_max) - finv(sigma_min))

        return cls(
            alpha_fn=lambda t: 0.0 * t + 1.0,
            sigma_fn=sigma_fn,
            alpha_fn_dot=lambda t: 0.0 * t,
            sigma_fn_dot=sigma_fn_dot,
            sigma_fn_inv=sigma_fn_inv,
        )

    @classmethod
    def edm(
        cls,
        expoent: float = 7.0,
        sigma_min: float = 0.02,
        sigma_max: float = 80.0
    ):
        f = lambda x: x**expoent  # noqa: E731
        finv = lambda x: x**(1 / expoent)  # noqa: E731
        fdot = lambda x: expoent * x**(expoent - 1)  # noqa: E731
        return cls.finterpolation(f, finv, fdot, sigma_min, sigma_max)

    @classmethod
    def sigma_space(
        cls,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0
    ):
        """
        Sigma-space scheduler where time variable IS sigma directly.

        In this mode:
        - t is in [sigma_min, sigma_max], not [0, 1]
        - sigma_fn(t) = t (identity)
        - alpha_fn(t) = 1 (constant, VE-style diffusion)

        This replicates the EDM paper's native parameterization
        where the model operates directly in sigma space.

        Args:
            sigma_min: Minimum noise level (typically 0.002)
            sigma_max: Maximum noise level (typically 80.0)
        """
        return cls(
            alpha_fn=lambda sigma: 1.0 + 0.0 * sigma,
            sigma_fn=lambda sigma: sigma,
            alpha_fn_dot=lambda sigma: 0.0 * sigma,
            sigma_fn_dot=lambda sigma: 1.0 + 0.0 * sigma,
            sigma_fn_inv=lambda s: s,
            time_domain='sigma',
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )

    @classmethod
    def get_interpolator(cls, name, *args, **kwargs):
        if name not in cls.named_interpolators():
            raise ValueError(f"Invalid interpolator: {name}")
        if name == 'linear':
            return cls.linear(*args, **kwargs)
        elif name == 'cosine':
            return cls.cosine(*args, **kwargs)
        elif name == 'edm':
            return cls.edm(*args, **kwargs)
        elif name == 'finterpolation':
            return cls.finterpolation(*args, **kwargs)
        elif name == 'sigma_space':
            return cls.sigma_space(*args, **kwargs)

    @classmethod
    def named_interpolators(cls):
        return ['linear', 'cosine', 'edm', 'finterpolation', 'sigma_space']

    @property
    def is_sigma_space(self) -> bool:
        """Check if operating in sigma-space mode."""
        return self.time_domain == 'sigma'

    def get_time_bounds(self) -> tuple[float, float]:
        """
        Get the bounds of the time variable.

        Returns:
            (t_min, t_max) - In normalized mode: (0, 1), in sigma mode: (sigma_min, sigma_max)
        """
        if self.is_sigma_space:
            return (self.sigma_min, self.sigma_max)
        else:
            return (0.0, 1.0)


class Preconditioner(object):
    def __init__(
        self,
        scheduler: SIScheduler,
        precondition_fn: Literal['identity', 'edm', 'edm_denoiser'] | Callable | None = 'identity',
        is_autonomous: bool = False,
        **kwargs
    ):
        self.scheduler = scheduler
        self.precondition_fn = precondition_fn
        self.is_autonomous = is_autonomous
        self.kwargs = kwargs

    def __call__(self, model, x, t=None, y=None):
        return self.get_flow_field(model, x, t, y)

    def _get_sigma(self, t):
        """Get sigma from t, handling both normalized and sigma-space modes."""
        if self.scheduler.is_sigma_space:
            return t  # t IS sigma in sigma-space mode
        else:
            return self.scheduler.sigma_fn(t)

    def _get_sigma_dot(self, t):
        """Get d(sigma)/dt, handling both modes."""
        if self.scheduler.is_sigma_space:
            return 1.0 + 0.0 * t  # d(sigma)/d(sigma) = 1
        else:
            return self.scheduler.sigma_fn_dot(t)

    def get_flow_field(self, model, x, t=None, y=None):
        if self.precondition_fn is None:
            v = self.identity(model, x, t, y)
        elif isinstance(self.precondition_fn, str):
            if self.precondition_fn == 'identity':
                v = self.identity(model, x, t, y)
            elif self.precondition_fn == 'edm':
                v = self.edm(model, x, t, y)
            elif self.precondition_fn == 'edm_denoiser':
                v = self.edm_denoiser(model, x, t, y)
            else:
                raise ValueError(f"Invalid condition function: {self.precondition_fn}")
        else:
            if self.is_autonomous:
                v = self.precondition_fn(model, x, y=y)
            else:
                v = self.precondition_fn(model, x, t, y=y)
        return v

    def get_denoiser_output(self, model, x, t=None, y=None):
        """
        Get the denoiser output D(x, sigma) directly for denoiser loss computation.

        This method returns the denoised estimate x̂_0 = D(x_noised, sigma),
        used for computing the Karras EDM-style denoiser loss:
            loss = lambda(sigma) * ||D(x_noised, sigma) - x_clean||^2

        Only supported for 'edm' precondition_fn currently.
        """
        if isinstance(self.precondition_fn, str):
            if self.precondition_fn == 'edm':
                return self.edm_denoiser(model, x, t, y)
            else:
                raise ValueError(
                    f"get_denoiser_output only supported for 'edm' precondition_fn, "
                    f"got '{self.precondition_fn}'"
                )
        else:
            raise ValueError(
                "get_denoiser_output only supported for string precondition_fn='edm'"
            )

    def identity(self, model, x, t=None, y=None):
        if self.is_autonomous:
            return model(x, y=y)
        else:
            return model(x, t, y=y)

    def edm(self, model, x, t=None, y=None):
        """
        EDM preconditioner that outputs flow field.

        The model predicts a denoised image, which is converted to flow field:
            flow_field = (d_sigma/dt) / sigma * (x - D(x))

        In sigma-space mode, d_sigma/dt = 1, so:
            flow_field = (x - D(x)) / sigma
        """
        sigma_data = self.kwargs.get("sigma_data", 0.5)
        sigma = self._get_sigma(t)
        sigma_dot = self._get_sigma_dot(t)
        sigma = broadcast_from_below(sigma, x)
        sigma_dot = broadcast_from_below(sigma_dot, x)
        cin = 1 / torch.sqrt(sigma_data**2 + sigma**2)
        cout = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        cskip = sigma_data ** 2 / (sigma_data**2 + sigma**2)
        if self.is_autonomous:
            flow_field = cskip * x + cout * model(x / cin, y=y)
        else:
            cnoise = 0.5 * torch.log(sigma)
            # Flatten cnoise for the model (it expects [batch] not [batch, 1, 1, ...])
            if cnoise.dim() > 1:
                cnoise = cnoise.view(cnoise.shape[0])
            denoiser = cskip * x + cout * model(cin * x, cnoise, y=y)

            flow_field = sigma_dot / sigma * (x - denoiser)
        return flow_field

    def edm_denoiser(self, model, x, t=None, y=None):
        """
        EDM preconditioner that outputs the denoised image directly.

        This is useful for computing score or when you need D(x) rather than flow field.
        """
        sigma_data = self.kwargs.get("sigma_data", 0.5)
        sigma = self._get_sigma(t)
        sigma = broadcast_from_below(sigma, x)
        cin = 1 / torch.sqrt(sigma_data**2 + sigma**2)
        cout = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        cskip = sigma_data ** 2 / (sigma_data**2 + sigma**2)

        cnoise = 0.5 * torch.log(sigma)
        if cnoise.dim() > 1:
            cnoise = cnoise.view(cnoise.shape[0])
        denoiser = cskip * x + cout * model(cin * x, cnoise, y=y)
        return denoiser


class LossWeighting(object):
    def __init__(
        self,
        scheduler: SIScheduler,
        weighting_class: Literal['edm', 'uniform', 'edm_sigma'] | dict[str, Any] = 'uniform',
        **kwargs
    ):
        self.scheduler = scheduler
        self.kwargs = kwargs
        self.weighting_class = weighting_class

        if not isinstance(weighting_class, str):
            assert 'weighting_function' in weighting_class
            assert 'weighting_sampler' in weighting_class

    def weighting_function(self, t):
        if isinstance(self.weighting_class, str):
            if self.weighting_class == 'edm':
                return self.edm_weighting_function(t)
            elif self.weighting_class == 'edm_sigma':
                return self.edm_sigma_weighting_function(t)
            elif self.weighting_class == 'uniform':
                return self.uniform_weighting_function(t)
            else:
                raise ValueError(f"Invalid weighting class: {self.weighting_class}")
        else:
            return self.weighting_class['weighting_function'](t)

    def weighting_sampler(self, nsamples):
        if isinstance(self.weighting_class, str):
            if self.weighting_class == 'edm':
                return self.edm_weighting_sampler(nsamples)
            elif self.weighting_class == 'edm_sigma':
                return self.edm_sigma_weighting_sampler(nsamples)
            elif self.weighting_class == 'uniform':
                return self.uniform_weighting_sampler(nsamples)
            else:
                raise ValueError(f"Invalid weighting class: {self.weighting_class}")
        else:
            return self.weighting_class['weighting_sampler'](nsamples)

    def uniform_weighting_function(self, t):
        return 1.0 + 0.0 * t

    def uniform_weighting_sampler(self, nsamples):
        """Sample uniform in [0, 1] for normalized mode, or [sigma_min, sigma_max] for sigma mode."""
        if self.scheduler.is_sigma_space:
            # Uniform in log-sigma space
            log_min = np.log(self.scheduler.sigma_min)
            log_max = np.log(self.scheduler.sigma_max)
            log_sigma = torch.rand(nsamples) * (log_max - log_min) + log_min
            return torch.exp(log_sigma)
        else:
            return torch.rand(nsamples)

    def edm_weighting_function(self, t):
        return self.uniform_weighting_function(t)
        # sigma = self.scheduler.sigma_fn(t)
        # sigma_dot = self.scheduler.sigma_fn_dot(t)
        # sigma_data = self.kwargs.get("sigma_data", 1.0)
        # lambd = (sigma_data**2 + sigma**2) / ((sigma * sigma_data)**2)
        # weight = lambd * sigma_dot**2 / sigma**2
        # return weight

    def edm_weighting_sampler(self, nsamples):
        """EDM sampler for normalized time mode - samples sigma then converts to t."""
        pmean = self.kwargs.get("pmean", -1.2)
        pstd = self.kwargs.get("pstd", 1.2)
        logsigma = pstd * torch.randn(nsamples) + pmean
        sigma = torch.exp(logsigma)
        t = self.scheduler.sigma_fn_inv(sigma)
        return t

    def edm_sigma_weighting_function(self, sigma):
        """
        EDM loss weighting function for sigma-space mode.

        lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2

        This is the weighting from the EDM paper (Karras et al. 2022).
        """
        sigma_data = self.kwargs.get("sigma_data", 0.5)
        return (sigma**2 + sigma_data**2) / ((sigma * sigma_data)**2)

    def edm_sigma_weighting_sampler(self, nsamples):
        """
        EDM sampler for sigma-space mode - samples sigma directly.

        Samples from log-normal: log(sigma) ~ N(pmean, pstd^2)
        This is the training distribution from the EDM paper.
        """
        pmean = self.kwargs.get("pmean", -1.2)
        pstd = self.kwargs.get("pstd", 1.2)
        logsigma = pstd * torch.randn(nsamples) + pmean
        sigma = torch.exp(logsigma)
        return sigma  # Return sigma directly, not converted to t


class SIModuleConfig(torch.nn.Module):
    """
    Configuration for Stochastic Interpolant Module.

    Supports two time parameterizations:
    - Normalized mode (default): t in [0, 1], scheduler maps to noise levels
    - Sigma-space mode: t IS sigma directly, t in [sigma_min, sigma_max]

    Supports two loss formulations:
    - 'flow_field': Train on flow field matching ||v(x_t, t) - v_target||
    - 'denoiser': Train on denoiser matching ||D(x_t, sigma) - x_clean|| (Karras EDM style)

    Use the factory methods for common configurations:
    - SIModuleConfig.from_edm_sigma_space() for EDM paper-style behavior
    """

    def __init__(self,
                 scheduler: SIScheduler | str = 'linear',
                 scheduler_args: dict[str, Any] = {},
                 num_channels: int | None = None,
                 initial_norm: bool | float = False,
                 autonomous_flow: bool = False,
                 precondition_fn: Callable | str | None = None,
                 preconditioner_kwargs: dict[str, Any] = {},
                 loss_weighting: Literal['edm', 'uniform', 'edm_sigma'] | dict[str, Any] = 'uniform',
                 loss_weighting_kwargs: dict[str, Any] = {},
                 loss_metric: Literal['mse', 'huber'] = 'huber',
                 loss_formulation: Literal['flow_field', 'denoiser'] = 'flow_field',
                 autoencoder_is_conditional: bool = False,
                 encode_condition: bool = False):
        super().__init__()
        if isinstance(scheduler, str):
            scheduler = SIScheduler.get_interpolator(scheduler, **scheduler_args)
        else:
            scheduler = scheduler
        self.scheduler = scheduler
        self.num_channels = num_channels
        self.initial_norm = initial_norm
        self.autonomous_flow = autonomous_flow
        self._loss_weighting_config = loss_weighting
        self._loss_weighting_kwargs = loss_weighting_kwargs
        self.loss_metric = loss_metric
        self.loss_formulation = loss_formulation
        self.precondition_fn = precondition_fn
        self.preconditioner_kwargs = preconditioner_kwargs
        self.autoencoder_is_conditional = autoencoder_is_conditional
        self.encode_condition = encode_condition
        self.set_scheduling_functions()
        self.set_loss_metric_module()
        self.set_preconditioner()
        self.set_loss_weighting()

    @classmethod
    def from_edm_sigma_space(
        cls,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        pmean: float = -1.2,
        pstd: float = 1.2,
        num_channels: int | None = None,
        initial_norm: bool | float = False,
        loss_metric: Literal['mse', 'huber'] = 'huber',
        loss_formulation: Literal['flow_field', 'denoiser'] = 'flow_field',
        autoencoder_is_conditional: bool = False,
        encode_condition: bool = False
    ):
        """
        Create an EDM configuration that operates directly in sigma-space.

        This replicates the EDM paper behavior where:
        - Time variable t IS sigma directly (not normalized to [0, 1])
        - Model receives 0.5 * log(sigma) as noise conditioning
        - Training samples sigma from log-normal distribution
        - Integration happens in sigma space

        Args:
            sigma_min: Minimum noise level (default 0.002)
            sigma_max: Maximum noise level (default 80.0)
            sigma_data: Data standard deviation for preconditioning (default 0.5)
            pmean: Log-normal prior mean for training (default -1.2)
            pstd: Log-normal prior std for training (default 1.2)
            num_channels: Number of channels for batch norm
            initial_norm: Whether to use initial normalization
            loss_metric: Loss function ('mse' or 'huber')
            loss_formulation: Loss formulation to use:
                - 'flow_field': Train on flow field matching (default)
                - 'denoiser': Train on denoiser matching ||D(x_t, sigma) - x_clean||
                  This replicates the original Karras EDM paper loss function.
            autoencoder_is_conditional: Whether autoencoder uses conditioning
            encode_condition: Whether to encode the condition

        Returns:
            SIModuleConfig configured for EDM sigma-space operation
        """
        scheduler = SIScheduler.sigma_space(sigma_min=sigma_min, sigma_max=sigma_max)

        return cls(
            scheduler=scheduler,
            num_channels=num_channels,
            initial_norm=initial_norm,
            autonomous_flow=False,
            precondition_fn='edm',
            preconditioner_kwargs={'sigma_data': sigma_data},
            loss_weighting='edm_sigma',
            loss_weighting_kwargs={'sigma_data': sigma_data, 'pmean': pmean, 'pstd': pstd},
            loss_metric=loss_metric,
            loss_formulation=loss_formulation,
            autoencoder_is_conditional=autoencoder_is_conditional,
            encode_condition=encode_condition
        )

    @property
    def is_sigma_space(self) -> bool:
        """Check if operating in sigma-space mode."""
        return self.scheduler.is_sigma_space

    def get_time_bounds(self) -> tuple[float, float]:
        """Get the bounds of the time variable."""
        return self.scheduler.get_time_bounds()

    def set_scheduling_functions(self):
        self.alpha_fn = self.scheduler.alpha_fn
        self.sigma_fn = self.scheduler.sigma_fn
        self.alpha_fn_dot = self.scheduler.alpha_fn_dot
        self.sigma_fn_dot = self.scheduler.sigma_fn_dot
        self.sigma_fn_inv = self.scheduler.sigma_fn_inv

    def set_loss_metric_module(self):
        if self.loss_metric == 'mse':
            self.loss_metric_module = torch.nn.MSELoss(reduction="none")
        elif self.loss_metric == 'huber':
            self.loss_metric_module = torch.nn.HuberLoss(reduction="none")
        else:
            raise ValueError(f"Invalid loss metric: {self.loss_metric}")

    def set_preconditioner(self):
        self.preconditioner = Preconditioner(
            self.scheduler,
            self.precondition_fn,
            self.autonomous_flow,
            **self.preconditioner_kwargs
        )

    def set_loss_weighting(self):
        if isinstance(self._loss_weighting_config, str):
            self.loss_weighting = LossWeighting(
                self.scheduler,
                self._loss_weighting_config,
                **self._loss_weighting_kwargs
            )
        else:
            self.loss_weighting = LossWeighting(self.scheduler, **self._loss_weighting_config)


class SIModule(lightning.LightningModule):
    def __init__(
        self,
        config: SIModuleConfig,
        model: nn.Module,
        autoencoder: nn.Module | None = None
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.set_initial_norm()
        self.autoencoder = autoencoder
        if self.autoencoder:
            self.freeze_autoencoder()

    def freeze_autoencoder(self):
        """
        Freezes the autoencoder to prevent its weights from being updated
        during training.
        """
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def encode(self, x, y=None):
        if not self.autoencoder:
            return x, y
        if not self.config.autoencoder_is_conditional and not self.config.encode_condition:
            x = self.autoencoder.encode(x)
        elif self.config.autoencoder_is_conditional and not self.config.encode_condition:
            x = self.autoencoder.encode(x, y)
        elif not self.config.autoencoder_is_conditional and self.config.encode_condition:
            raise ValueError("Cannot encode condition if autoencoder is not conditional")
        else:
            x, y = self.autoencoder.encode(x, y)
        if isinstance(x, dict):  # Handle the case where the autoencoder returns a dict
            x = x['zsample']
        return x, y

    def decode(self, x, y=None):
        if not self.autoencoder:
            return x, y
        if not self.config.autoencoder_is_conditional:
            x = self.autoencoder.decode(x)
        else:
            x = self.autoencoder.decode(x, y)
        return x, y

    def set_initial_norm(self):
        if isinstance(self.config.initial_norm, bool):
            if self.config.initial_norm:
                self.initial_norm = DimensionAgnosticBatchNorm(self.config.num_channels)
            else:
                self.initial_norm = IdentityBatchNorm()
        elif isinstance(self.config.initial_norm, float) or isinstance(self.config.initial_norm, int):
            self.initial_norm = ConstantBatchNorm(self.config.initial_norm)
        else:
            raise ValueError(f"Invalid initial norm: {self.config.initial_norm}")

    def loss_fn(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
                t: Float[Tensor, "batch"],  # noqa: F821, typing
                y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821, typing
                mask: None | Float[Tensor, "batch *shape"] = None  # noqa: F821, typing
                ) -> Float[Tensor, ""]:  # noqa: F821, F722
        x, y = self.encode(x, y)
        x = self.initial_norm(x)
        noise = torch.randn_like(x)
        t_broadcasted = broadcast_from_below(t, x)
        alpha, sigma = self.config.alpha_fn(t_broadcasted), self.config.sigma_fn(t_broadcasted)
        x_noised = alpha * x + sigma * noise

        if self.config.loss_formulation == 'denoiser':
            # Denoiser loss (Karras EDM style):
            # loss = lambda(sigma) * ||D(x_noised, sigma) - x_clean||^2
            # This matches the original karrasmodule.py behavior
            denoiser_output = self.get_denoiser_output(x_noised, t, y=y)
            target = x  # Clean data is the target
            loss = self.config.loss_metric_module(denoiser_output, target)
        else:
            # Flow field loss (default):
            # loss = w(t) * ||v(x_noised, t) - (alpha_dot * x + sigma_dot * noise)||^2
            flow_field = self.get_flow_field(x_noised, t, y=y, guidance=1.0)
            alpha_dot = self.config.alpha_fn_dot(t_broadcasted)
            sigma_dot = self.config.sigma_fn_dot(t_broadcasted)
            target = (alpha_dot * x + sigma_dot * noise)
            loss = self.config.loss_metric_module(flow_field, target)

        loss_weighting = self.config.loss_weighting.weighting_function(t_broadcasted)
        loss = loss * loss_weighting

        if mask is not None:
            # mask=1 where loss should be computed, 0 where it should be skipped
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return loss

    def sample_timestep(self, nsamples):
        # Sample from uniform 0 and 1
        t = self.config.loss_weighting.weighting_sampler(nsamples)
        return t

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch.get('y', None)
        mask = batch.get('mask', None)
        t = self.sample_timestep(x.shape[0]).to(x)
        loss = self.loss_fn(x, t, y, mask)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch.get('y', None)
        mask = batch.get('mask', None)
        t = self.sample_timestep(x.shape[0]).to(x)
        loss = self.loss_fn(x, t, y, mask)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def set_optimizer_and_scheduler(self,
                                    optimizer=None,
                                    scheduler=None,
                                    scheduler_interval="step"):
        """
        Parameters
        ----------
        optimizer : None | torch.optim.Optimizer
            if None, use the default optimizer AdamW,
            with learning rate 1e-3, betas=(0.9, 0.999),
            and weight decay 1e-4
        scheduler : None | torch.optim.lr_scheduler._LRScheduler
            if None, use the default scheduler CosineAnnealingWarmRestarts,
            with T_0=10.
        scheduler_interval : str
            "epoch" or "step", whether the scheduler should be called at the
            end of each epoch or each step.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=1e-4,
                                               betas=(0.9, 0.999),
                                               weight_decay=1e-4)
        if scheduler is not None:
            self.lr_scheduler = scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0 + 0 * step  # noqa
            )  # Neutral scheduler
        self.lr_scheduler_interval = scheduler_interval

    def configure_optimizers(self):
        lr_scheduler_config = {"scheduler": self.lr_scheduler,
                               "interval": self.lr_scheduler_interval}
        # self.hp_manager.add_runtime_optimizer_info(self.optimizer, self.lr_scheduler)
        # self.hp_manager.log_to_wandb()

        return [self.optimizer], [lr_scheduler_config]

    def get_flow_field(
        self,
        x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t: Float[Tensor, "batch"],  # noqa: F821, typing
        guidance: float = 1.0,
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing,
        integrate_on_sigma: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        if guidance == 1.0 or y is None:  # Implictly no guidance
            flow_field = self.config.preconditioner(self.model, x_noised, t, y=y)
        else:
            flow_field = self.config.preconditioner(self.model, x_noised, t, y=y)
            unconditioned_flow_field = self.config.preconditioner(self.model, x_noised, t, y=None)
            flow_field = guidance * flow_field + (1 - guidance) * unconditioned_flow_field
        if integrate_on_sigma:
            sigma_dot = self.config.sigma_fn_dot(t)
            flow_field = flow_field / sigma_dot
        return flow_field

    def get_denoiser_output(
        self,
        x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t: Float[Tensor, "batch"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        """
        Get the denoiser output D(x_noised, sigma) for denoiser loss computation.

        This returns the preconditioned denoiser output, used for computing
        the Karras EDM-style denoiser loss:
            loss = lambda(sigma) * ||D(x_noised, sigma) - x_clean||^2

        Args:
            x_noised: Noisy input [batch, *shape]
            t: Time/sigma values [batch]
            y: Optional conditioning
            guidance: Classifier-free guidance scale (default 1.0, no guidance)

        Returns:
            Denoised estimate D(x_noised, sigma) [batch, *shape]
        """
        if guidance == 1.0 or y is None:
            denoiser = self.config.preconditioner.get_denoiser_output(
                self.model, x_noised, t, y=y
            )
        else:
            denoiser = self.config.preconditioner.get_denoiser_output(
                self.model, x_noised, t, y=y
            )
            unconditioned_denoiser = self.config.preconditioner.get_denoiser_output(
                self.model, x_noised, t, y=None
            )
            denoiser = guidance * denoiser + (1 - guidance) * unconditioned_denoiser
        return denoiser

    def get_score_field(
        self,
        x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t: Float[Tensor, "batch"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        integrate_on_sigma: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        flow_field = self.get_flow_field(x_noised, t, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
        (alpha, sigma, alpha_dot, sigma_dot) = (
            self.config.alpha_fn(t),
            self.config.sigma_fn(t),
            self.config.alpha_fn_dot(t),
            self.config.sigma_fn_dot(t)
        )
        alpha = broadcast_from_below(alpha, x_noised)
        sigma = broadcast_from_below(sigma, x_noised)
        alpha_dot = broadcast_from_below(alpha_dot, x_noised)
        sigma_dot = broadcast_from_below(sigma_dot, x_noised)
        score_field = ((alpha * flow_field - alpha_dot * x_noised) /
                       (sigma * (alpha_dot * sigma - alpha * sigma_dot)))
        return score_field

    def get_denoised_estimate(
        self,
        x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t: Float[Tensor, "batch"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        integrate_on_sigma: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        score_field = self.get_score_field(x_noised, t, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
        alpha, sigma = self.config.alpha_fn(t), self.config.sigma_fn(t)
        alpha = broadcast_from_below(alpha, x_noised)
        sigma = broadcast_from_below(sigma, x_noised)
        epsilon = 1e-8
        denoised_estimate = (x_noised + sigma**2 * score_field) / (alpha + epsilon)
        return denoised_estimate

    def get_score_field_from_flow_field(
        self,
        flow_field: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t: Float[Tensor, "batch"],  # noqa: F821, typing
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        (alpha, sigma, alpha_dot, sigma_dot) = (
            self.config.alpha_fn(t),
            self.config.sigma_fn(t),
            self.config.alpha_fn_dot(t),
            self.config.sigma_fn_dot(t)
        )
        alpha = broadcast_from_below(alpha, flow_field)
        sigma = broadcast_from_below(sigma, flow_field)
        alpha_dot = broadcast_from_below(alpha_dot, flow_field)
        sigma_dot = broadcast_from_below(sigma_dot, flow_field)
        score_field = ((alpha * flow_field - alpha_dot * x_noised) /
                       (sigma * (alpha_dot * sigma - alpha * sigma_dot)))
        return score_field

    def create_time_schedule(
        self,
        nsteps: int,
        rho: float = 7.0
    ) -> Float[Tensor, "nsteps"]:  # noqa: F821
        """
        Create a time schedule for sampling.

        In normalized mode: returns t from 1 to 0 (linear spacing)
        In sigma-space mode: returns sigma from sigma_max to sigma_min (power-law spacing)

        Args:
            nsteps: Number of steps
            rho: Exponent for power-law spacing in sigma-space mode (default 7.0, from EDM)

        Returns:
            Time schedule tensor of shape [nsteps]
        """
        if self.config.is_sigma_space:
            # EDM-style power-law spacing in sigma space
            sigma_min = self.config.scheduler.sigma_min
            sigma_max = self.config.scheduler.sigma_max
            step_indices = torch.arange(nsteps)
            # sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
            sigma_min_inv_rho = sigma_min ** (1 / rho)
            sigma_max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (sigma_max_inv_rho + step_indices / (nsteps - 1) * (sigma_min_inv_rho - sigma_max_inv_rho)) ** rho
            return sigmas
        else:
            # Linear spacing in normalized time
            return torch.linspace(1, 0, nsteps)

    def sample(self,
               nsamples: int,
               shape: list[int],
               y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
               guidance: float = 1.0,
               nsteps: int = 30,
               is_latent_shape: bool = False,
               integrate_on_sigma: bool = False,
               noise_injection: bool = False,
               return_latents: bool = False,
               orig_noise: Float[Tensor, "batch *shape"] | None = None,  # noqa: F821, typing
               rho: float = 7.0  # Exponent for sigma-space step schedule
               ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        if torch.inference_mode():
            with torch.no_grad():
                if orig_noise is None:
                    x = torch.randn(nsamples, *shape).to(self.device)
                else:
                    assert orig_noise.shape[0] == nsamples, "Number of samples must match"
                    assert list(orig_noise.shape[1:]) == list(shape), "Shape of noise must match"
                    x = orig_noise.to(self.device)
                if y is not None:
                    warnings.warn("Moving y to device: {}".format(self.device))
                    y = dict_to(y, self.device)
                if not is_latent_shape and self.autoencoder:
                    # Need to do a stupid hack for getting correct shape
                    x, _ = self.encode(x, y)
                    x = torch.randn_like(x)
                if y is not None:
                    y = dict_unsqueeze(y, 0)
                time_schedule = self.create_time_schedule(nsteps, rho=rho).to(x)
                sigma_init = self.config.sigma_fn(time_schedule[0])
                x = x * sigma_init
                x = self.integrate_flow_field(
                    x,
                    time_schedule,
                    y,
                    guidance,
                    integrate_on_sigma=integrate_on_sigma,
                    noise_injection=noise_injection)
                if not return_latents:
                    x, _ = self.decode(x, y)
        return x  # noqa: F821, F722

    def inpaint(
        self,
        x_orig: Float[Tensor, "*shape"],  # noqa: F821, typing,
        mask: Float[Tensor, "*shape"],  # noqa: F821, typing,
        nsamples: int = 1,
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        nsteps: int = 30,
        integrate_on_sigma: bool = False,
        noise_injection: bool = False,
        orig_noise: Float[Tensor, "batch *shape"] | None = None,  # noqa: F821, typing
        # New parameters for improved inpainting:
        mask_falloff: int = 0,  # Soft mask gradient width (0 = hard mask)
        resample_steps: int = 0,  # Number of resample iterations (RePaint-style)
        jump_length: int = 1,  # Steps to jump back when resampling
        rho: float = 7.0  # Exponent for sigma-space step schedule
    ) -> Float[Tensor, "batch *shape"]:  # noqa F821, typing
        # mask: 1 for where data is present, 0 for where data is absent
        warnings.warn("We are assuming we are in latent space for inpainting")
        with torch.inference_mode():
            with torch.no_grad():
                if y is not None:
                    warnings.warn("Moving y to device: {}".format(self.device))
                    y = dict_to(y, self.device)
                x_orig = x_orig.to(self.device)
                mask = mask.to(self.device)
                shape = x_orig.shape

                # Create soft mask if falloff is specified
                if mask_falloff > 0:
                    soft_mask = self._create_soft_mask(mask, mask_falloff)
                else:
                    soft_mask = mask

                x_orig = x_orig.unsqueeze(0)
                x_orig = self.initial_norm(x_orig)
                if orig_noise is None:
                    x = torch.randn(nsamples, *shape).to(self.device)
                else:
                    assert orig_noise.shape[0] == nsamples, "Number of samples must match"
                    assert orig_noise.shape[1:] == shape, "Shape of noise must match"
                    x = orig_noise.to(self.device)
                time_schedule = self.create_time_schedule(nsteps, rho=rho).to(x)
                sigma_init = self.config.sigma_fn(time_schedule[0])
                x = x * sigma_init

                for i in range(len(time_schedule) - 1):
                    t_curr = time_schedule[i] * torch.ones(x.shape[0]).to(x)
                    t_next = time_schedule[i + 1] * torch.ones(x.shape[0]).to(x)

                    # Resample loop (RePaint-style jump back)
                    for r in range(resample_steps + 1):
                        x = self.integration_step(
                            x,
                            t_curr,
                            t_next,
                            y,
                            guidance,
                            method='euler_maruyama',
                            integrate_on_sigma=integrate_on_sigma,
                            noise_injection=True
                        )
                        sigma = broadcast_from_below(self.config.sigma_fn(t_next), x_orig)
                        alpha = broadcast_from_below(self.config.alpha_fn(t_next), x_orig)

                        x_patch = alpha * x_orig + sigma * torch.randn_like(x_orig)
                        x = (1 - soft_mask) * x + soft_mask * x_patch

                        # Jump back if not last resample iteration and not at final timestep
                        if r < resample_steps and i + jump_length < len(time_schedule) - 1:
                            # Jump back by adding noise
                            t_jump = time_schedule[i]  # Jump back to current timestep
                            sigma_jump = broadcast_from_below(
                                self.config.sigma_fn(t_jump), x
                            )
                            alpha_jump = broadcast_from_below(
                                self.config.alpha_fn(t_jump), x
                            )
                            # Re-noise the sample
                            x = alpha_jump * x + sigma_jump * torch.randn_like(x)
                            # Also update the patch for the jumped state
                            x_patch_jump = alpha_jump * x_orig + sigma_jump * torch.randn_like(x_orig)
                            x = (1 - soft_mask) * x + soft_mask * x_patch_jump

                x = self.initial_norm.unnorm(x)
                return x

    def _create_soft_mask(
        self,
        mask: Float[Tensor, "*shape"],  # noqa: F821
        falloff: int
    ) -> Float[Tensor, "*shape"]:  # noqa: F821
        """
        Create a soft mask with cosine falloff at the boundary.

        Args:
            mask: Binary mask (1 = known, 0 = unknown)
            falloff: Width of the gradient transition zone in voxels

        Returns:
            Soft mask with smooth transition at boundaries
        """
        if falloff <= 0:
            return mask

        # Use average pooling to create distance-like field, then rescale
        # This is a simple approximation that works for 3D data
        import torch.nn.functional as F

        # Determine spatial dimensions (assume first dim is channels)
        ndim = mask.dim() - 1  # Number of spatial dimensions

        # Expand mask for pooling (need batch dim)
        m = mask.unsqueeze(0).float()  # [1, C, ...]

        # Apply average pooling to get approximate distance field
        kernel_size = 2 * falloff + 1
        padding = falloff

        if ndim == 3:
            # 3D case
            m_dilated = F.avg_pool3d(
                m, kernel_size=kernel_size, stride=1, padding=padding
            )
            m_eroded = F.avg_pool3d(
                1 - m, kernel_size=kernel_size, stride=1, padding=padding
            )
        elif ndim == 2:
            # 2D case
            m_dilated = F.avg_pool2d(
                m, kernel_size=kernel_size, stride=1, padding=padding
            )
            m_eroded = F.avg_pool2d(
                1 - m, kernel_size=kernel_size, stride=1, padding=padding
            )
        else:
            # Fallback: no soft mask for other dimensions
            return mask

        # Create soft mask: 1 in known region, 0 in unknown, gradient at boundary
        # Use the pooled values to create smooth transition
        soft_mask = m_dilated / (m_dilated + m_eroded + 1e-8)

        # Apply cosine smoothing for nicer transition
        soft_mask = (1 - torch.cos(soft_mask * np.pi)) / 2

        return soft_mask.squeeze(0)  # Remove batch dim

    # =========================================================================
    # Diffusion Posterior Sampling (DPS)
    # Based on: "Diffusion Posterior Sampling for General Noisy Inverse Problems"
    # (Chung et al., ICLR 2023)
    #
    # General formulation for linear inverse problems: y = Ax + ε
    # =========================================================================

    def _compute_dps_gradient(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821
        t: Float[Tensor, "batch"],  # noqa: F821
        y: Float[Tensor, "*mshape"],  # noqa: F821
        forward_operator: Callable[[Tensor], Tensor],
        rho: float,
        y_cond: None | dict = None,
        guidance: float = 1.0
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Compute DPS gradient via backpropagation through denoiser.

        DPS gradient: -ρ · ∇_{x_t} ||y - A(x̂_0(x_t))||²

        This requires backprop through the denoiser since x̂_0 is a function of x_t.

        Args:
            x: Current noisy state x_t [batch, *shape]
            t: Current time [batch]
            y: Observed measurement [*mshape]
            forward_operator: Callable A(x) computing forward measurement
            rho: Likelihood scaling (typically 1/η² where η is measurement noise)
            y_cond: Optional conditioning for the model
            guidance: Classifier-free guidance scale for denoiser

        Returns:
            DPS gradient: -ρ · ∇_{x_t} ||y - A(x̂_0)||² [batch, *shape]
        """
        # Must be called inside torch.enable_grad() context
        x_for_grad = x.detach().requires_grad_(True)

        # Compute denoised estimate x̂_0 = D(x_t, t)
        x_hat = self.get_denoised_estimate(x_for_grad, t, y=y_cond, guidance=guidance)

        # Apply forward operator: A(x̂_0)
        pred = forward_operator(x_hat)

        # Expand y to match pred shape if needed
        if y.dim() < pred.dim():
            y_expanded = y.unsqueeze(0).expand_as(pred)
        else:
            y_expanded = y

        # Measurement loss: ||y - A(x̂_0)||²
        loss = ((y_expanded - pred) ** 2).sum()

        # Backprop to get gradient w.r.t. x_t
        grad_x = torch.autograd.grad(loss, x_for_grad)[0]

        # Return -ρ * gradient (negative because we want score, not loss gradient)
        return -rho * grad_x

    def sample_posterior_dps(
        self,
        measurement: Float[Tensor, "*mshape"],  # noqa: F821
        forward_operator: Callable[[Tensor], Tensor],
        shape: list[int],
        nsamples: int = 1,
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
        guidance: float = 1.0,
        nsteps: int = 50,
        # DPS parameters
        zeta: float = 1.0,
        eta: float = 1.0,
        jitter: bool = True,
        grad_clip: float = 1.0,
        # Integration parameters
        method: Literal['euler', 'heun'] = 'heun',
        # Schedule parameters
        rho: float = 7.0,
        # Output options
        return_trajectory: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Sample from posterior p(x|y) via Diffusion Posterior Sampling (DPS).

        General method for inverse problems: y = A(x) + ε

        DPS modifies the reverse diffusion process by adding a guidance term:
            x_{t-1} = x_{t-1}^{uncond} + ζ · (-ρ · ∇_{x_t} ||y - A(x̂_0)||²)

        where x̂_0 is the denoised estimate, and ρ = 1/η².

        Args:
            measurement: Observed measurement y [*mshape] (should be normalized)
            forward_operator: Callable A(x) computing forward measurement
            shape: Shape of x to generate (without batch dimension)
            nsamples: Number of samples to generate
            y: Optional conditioning for the diffusion model
            guidance: Classifier-free guidance scale

            nsteps: Number of diffusion steps

            zeta: DPS guidance strength (ζ). Higher = stronger constraint.
            eta: Measurement noise std (η). ρ = 1/η² scales the likelihood.
            jitter: If True, add η noise to measurement each step.
            grad_clip: Max norm for DPS gradient clipping. Prevents explosion.

            method: Integration method ('euler' or 'heun')

            rho: Power-law exponent for time schedule

            return_trajectory: If True, return (samples, trajectory) tuple

        Returns:
            Samples from approximate posterior p(x|y) [batch, *shape]
        """
        device = self.device
        measurement = measurement.to(device)
        if y is not None:
            y = dict_to(y, device)

        # Likelihood scaling: ρ = 1/η²
        rho_likelihood = 1.0 / (eta ** 2)

        # Initialize with random noise
        x = torch.randn(nsamples, *shape, device=device)

        # Get time schedule and scale initial noise
        time_schedule = self.create_time_schedule(nsteps, rho=rho).to(device)
        sigma_init = self.config.sigma_fn(time_schedule[0])
        x = x * sigma_init

        trajectory = [x.clone()] if return_trajectory else None

        # DPS Integration Loop
        for i in range(len(time_schedule) - 1):
            t_curr = time_schedule[i] * torch.ones(nsamples, device=device)
            t_next = time_schedule[i + 1] * torch.ones(nsamples, device=device)

            # Compute dt
            dt = t_next - t_curr
            dt = broadcast_from_below(dt, x)

            # Standard flow field step - no gradients needed
            with torch.no_grad():
                if method == 'euler':
                    v = self.get_flow_field(x, t_curr, y=y, guidance=guidance)
                    x_next = x + dt * v
                elif method == 'heun':
                    v1 = self.get_flow_field(x, t_curr, y=y, guidance=guidance)
                    x_euler = x + dt * v1
                    v2 = self.get_flow_field(x_euler, t_next, y=y, guidance=guidance)
                    x_next = x + dt * (v1 + v2) / 2
                else:
                    raise ValueError(f"Invalid method: {method}")

            # DPS gradient - requires backprop through denoiser
            if jitter:
                y_jittered = measurement + eta * torch.randn_like(measurement)
            else:
                y_jittered = measurement

            with torch.enable_grad():
                dps_grad = self._compute_dps_gradient(
                    x, t_curr, y_jittered, forward_operator,
                    rho=rho_likelihood, y_cond=y, guidance=guidance
                )

            # Clip gradient to prevent explosion
            grad_norm = dps_grad.norm()
            if grad_norm > grad_clip:
                dps_grad = dps_grad * (grad_clip / grad_norm)

            # Update
            x = x_next + zeta * dps_grad

            if return_trajectory:
                trajectory.append(x.clone())

        # Denormalize output
        x = self.initial_norm.unnorm(x)

        if return_trajectory:
            trajectory = [self.initial_norm.unnorm(t) for t in trajectory]
            return x, trajectory

        return x

    def inpaint_dps(
        self,
        x_orig: Float[Tensor, "*shape"],  # noqa: F821
        mask: Float[Tensor, "*shape"],  # noqa: F821
        nsamples: int = 1,
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
        guidance: float = 1.0,
        nsteps: int = 50,
        # DPS parameters
        zeta: float = 1.0,
        eta: float = 1.0,
        jitter: bool = True,
        grad_clip: float = 1.0,
        # Integration parameters
        method: Literal['euler', 'heun'] = 'heun',
        # Mask parameters
        mask_falloff: int = 0,
        # Schedule parameters
        rho: float = 7.0,
        # Output options
        return_trajectory: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Inpainting via Diffusion Posterior Sampling (DPS).

        Convenience wrapper around sample_posterior_dps for the inpainting
        case where the forward operator is mask projection.

        Args:
            x_orig: Original data with known regions [*shape]
            mask: Binary mask, 1=known, 0=unknown [*shape]
            nsamples: Number of samples to generate
            y: Optional conditioning for the diffusion model
            guidance: Classifier-free guidance scale

            nsteps: Number of diffusion steps

            zeta: DPS guidance strength (ζ). Higher = stronger constraint.
            eta: Measurement noise std (η) in y = Mx + ε, ε ~ N(0, η²I).
                 Used in likelihood score scaling (ρ = 1/η²).
            jitter: If True, add η noise to known pixels each step.
            grad_clip: Max norm for DPS gradient clipping. Prevents explosion.

            method: Integration method ('euler' or 'heun')

            mask_falloff: Soft mask gradient width (0 = hard mask)

            rho: Power-law exponent for time schedule

            return_trajectory: If True, return (samples, trajectory) tuple

        Returns:
            Inpainted samples [batch, *shape]
        """
        warnings.warn("Assuming latent space operation for DPS inpainting")

        device = self.device
        x_orig = x_orig.to(device)
        mask = mask.to(device)

        # Create soft mask if specified
        if mask_falloff > 0:
            soft_mask = self._create_soft_mask(mask, mask_falloff)
        else:
            soft_mask = mask

        # Normalize x_orig and create measurement
        x_orig_normed = self.initial_norm(x_orig.unsqueeze(0)).squeeze(0)  # [*shape]
        measurement = soft_mask * x_orig_normed

        # Forward operator for inpainting: A(x) = mask * x
        def forward_operator(x):
            mask_expanded = soft_mask.unsqueeze(0).expand_as(x)
            return mask_expanded * x

        # Call general DPS method
        return self.sample_posterior_dps(
            measurement=measurement,
            forward_operator=forward_operator,
            shape=list(x_orig.shape),
            nsamples=nsamples,
            y=y,
            guidance=guidance,
            nsteps=nsteps,
            zeta=zeta,
            eta=eta,
            jitter=jitter,
            grad_clip=grad_clip,
            method=method,
            rho=rho,
            return_trajectory=return_trajectory
        )

    # =========================================================================
    # LanPaint: Training-Free Diffusion Inpainting via Langevin Dynamics
    # Based on: "Training-Free Diffusion Model Inpainting via Langevin Sampling"
    #
    # Key insight: Sample from a tractable surrogate distribution
    # q_{λ,σ}(x,y) ∝ p_σ(x,y) · exp(-λ/(2σ²)||y - α(σ)y₀||²)
    # using Langevin dynamics with the "BiG score".
    #
    # Unlike DPS, no autograd through denoiser is needed.
    # =========================================================================

    def _compute_big_score(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821
        t: Float[Tensor, "batch"],  # noqa: F821
        y_0: Float[Tensor, "*shape"],  # noqa: F821
        unknown_mask: Float[Tensor, "*shape"],  # noqa: F821
        lambda_: float,
        sigma_min: float = 0.02,
        y_cond: None | dict = None,
        guidance: float = 1.0,
        debug: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Compute BiG (Bidirectional Guidance) score for LanPaint.

        For unknown pixels (M=1): s_model
        For known pixels (M=0): -λ * s_model + (1 + λ) * s_prior

        Where s_prior = ∇log p(y_t|y_0) = (α·y₀ - x) / σ² is the score of
        the transition kernel N(α·y₀, σ²·I).

        Args:
            x: Current noisy state [batch, *shape]
            t: Time [batch]
            y_0: Clean known pixel values (normalized) [*shape]
            unknown_mask: 1=unknown (to inpaint), 0=known (observed) [*shape]
            lambda_: Guidance strength
            sigma_min: (unused, kept for API compatibility)
            y_cond: Optional conditioning dict for the model
            guidance: Classifier-free guidance scale

        Returns:
            BiG score [batch, *shape]
        """
        if debug:
            print(f"  [BiG] x: min={x.min():.4f}, max={x.max():.4f}, nan={x.isnan().any()}")

        # Get model score (no autograd needed)
        with torch.no_grad():
            s_model = self.get_score_field(x, t, y=y_cond, guidance=guidance)

        if debug:
            print(f"  [BiG] s_model: min={s_model.min():.4f}, max={s_model.max():.4f}, nan={s_model.isnan().any()}")

        # Get sigma and alpha
        sigma = self.config.sigma_fn(t)
        alpha = self.config.alpha_fn(t)
        sigma = broadcast_from_below(sigma, x)
        alpha = broadcast_from_below(alpha, x)

        if debug:
            print(f"  [BiG] sigma: min={sigma.min():.4f}, max={sigma.max():.4f}")
            print(f"  [BiG] alpha: min={alpha.min():.4f}, max={alpha.max():.4f}")

        # Prior score: ∇log p(y_t|y_0) = (α·y₀ - x) / σ²
        # This is the score of the transition kernel N(α·y₀, σ²·I)
        s_prior = (alpha * y_0 - x) / (sigma ** 2)

        if debug:
            print(f"  [BiG] s_prior: min={s_prior.min():.4f}, max={s_prior.max():.4f}, nan={s_prior.isnan().any()}")

        # Known mask (1-M where M=unknown_mask)
        known_mask = 1.0 - unknown_mask

        # BiG score:
        # - Unknown (M=1): s_model
        # - Known (M=0): -λ * s_model + (1 + λ) * s_prior
        score_unknown = s_model
        score_known = -lambda_ * s_model + (1 + lambda_) * s_prior

        result = unknown_mask * score_unknown + known_mask * score_known

        if debug:
            print(f"  [BiG] score_unknown: min={score_unknown.min():.4f}, max={score_unknown.max():.4f}")
            print(f"  [BiG] score_known: min={score_known.min():.4f}, max={score_known.max():.4f}")
            print(f"  [BiG] result: min={result.min():.4f}, max={result.max():.4f}, nan={result.isnan().any()}")

        return result

    # MCMC step methods are now in diffsci2/models/karras/mcmc.py
    # See: OverdampedLangevin, TamedULA, BAOAB, HMC, FLD, MALA

    def _diffusion_step_lanpaint(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821
        t_curr: Float[Tensor, "batch"],  # noqa: F821
        t_next: Float[Tensor, "batch"],  # noqa: F821
        y_cond: None | dict = None,
        guidance: float = 1.0,
        debug: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        DDIM-like diffusion step from t_curr to t_next.

        This is a heuristic warm start for the next Langevin correction,
        NOT exact transport of q_{λ,σ} to q_{λ,σ_next}.

        Args:
            x: Current state at t_curr [batch, *shape]
            t_curr: Current time [batch]
            t_next: Next time [batch] (t_next < t_curr, moving toward 0)
            y_cond: Optional conditioning dict
            guidance: Classifier-free guidance scale

        Returns:
            State at t_next [batch, *shape]
        """
        if debug:
            print(f"  [Diff] x input: min={x.min():.4f}, max={x.max():.4f}, nan={x.isnan().any()}")

        with torch.no_grad():
            # Get denoised estimate x̂_0
            x_hat = self.get_denoised_estimate(x, t_curr, y=y_cond, guidance=guidance)

        if debug:
            print(f"  [Diff] x_hat: min={x_hat.min():.4f}, max={x_hat.max():.4f}, nan={x_hat.isnan().any()}")

        # Get sigma and alpha at current and next times
        sigma_curr = self.config.sigma_fn(t_curr)
        sigma_next = self.config.sigma_fn(t_next)
        alpha_curr = self.config.alpha_fn(t_curr)
        alpha_next = self.config.alpha_fn(t_next)

        sigma_curr = broadcast_from_below(sigma_curr, x)
        sigma_next = broadcast_from_below(sigma_next, x)
        alpha_curr = broadcast_from_below(alpha_curr, x)
        alpha_next = broadcast_from_below(alpha_next, x)

        if debug:
            print(f"  [Diff] sigma_curr={sigma_curr.mean():.4f}, sigma_next={sigma_next.mean():.4f}")

        # Estimate noise: x = α·x̂_0 + σ·ε → ε = (x - α·x̂_0) / σ
        eps = (x - alpha_curr * x_hat) / (sigma_curr + 1e-8)

        if debug:
            print(f"  [Diff] eps: min={eps.min():.4f}, max={eps.max():.4f}, nan={eps.isnan().any()}")

        # Reconstruct at next noise level
        x_next = alpha_next * x_hat + sigma_next * eps

        if debug:
            print(f"  [Diff] x_next: min={x_next.min():.4f}, max={x_next.max():.4f}, nan={x_next.isnan().any()}")

        return x_next

    def _langevin_correction(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821
        t: Float[Tensor, "batch"],  # noqa: F821
        y_0: Float[Tensor, "*shape"],  # noqa: F821
        unknown_mask: Float[Tensor, "*shape"],  # noqa: F821
        lambda_: float,
        K: int,
        mcmc_method: str = "tamed_ula",
        mcmc_args: dict | None = None,
        max_score: float | None = None,
        y_cond: None | dict = None,
        guidance: float = 1.0,
        debug: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Run K MCMC steps at fixed noise level to sample from q_{λ,σ}.

        Uses the mcmc module for clean, modular MCMC stepping.

        Args:
            x: Initial state [batch, *shape]
            t: Time (fixed during MCMC steps) [batch]
            y_0: Clean known pixel values (normalized) [*shape]
            unknown_mask: 1=unknown, 0=known [*shape]
            lambda_: Guidance strength
            K: Number of MCMC steps (or trajectories for HMC)
            mcmc_method: MCMC method. One of:
                - "overdamped", "ula": Overdamped Langevin
                - "tamed_ula", "tamed": Tamed ULA (default, stable)
                - "mala": Metropolis-adjusted Langevin
                - "baoab": BAOAB splitting (stable underdamped)
                - "fld": Fast Langevin Dynamics
                - "hmc": Hamiltonian Monte Carlo
            mcmc_args: Method-specific arguments dict
            max_score: If set, clamp score magnitude to prevent explosion
            y_cond: Optional conditioning dict
            guidance: Classifier-free guidance scale

        Returns:
            State after K MCMC steps [batch, *shape]
        """
        mcmc_args = mcmc_args or {}

        # For FLD, we need to pass sigma
        if mcmc_method.lower() == "fld":
            sigma = self.config.sigma_fn(t)
            mcmc_args = {**mcmc_args, "sigma": sigma.mean().item()}

        # Create stepper
        stepper = get_stepper(mcmc_method, **mcmc_args)

        # Create score function closure
        def score_fn(x_: Tensor) -> Tensor:
            score = self._compute_big_score(
                x_, t, y_0, unknown_mask, lambda_,
                y_cond=y_cond, guidance=guidance, debug=False
            )
            # Optionally clamp score
            if max_score is not None:
                score = torch.clamp(score, -max_score, max_score)
            return score

        # Run K MCMC steps
        for k in range(K):
            if debug:
                print(f"  [MCMC:{mcmc_method}] Step {k}: x min={x.min():.4f}, max={x.max():.4f}, nan={x.isnan().any()}")

            x = stepper.step(x, score_fn)

            if debug:
                print(f"  [MCMC:{mcmc_method}] After step {k}: x min={x.min():.4f}, max={x.max():.4f}")

        return x

    def sample_posterior_lanpaint(
        self,
        y_0: Float[Tensor, "*shape"],  # noqa: F821
        unknown_mask: Float[Tensor, "*shape"],  # noqa: F821
        nsamples: int = 1,
        y: None | dict = None,
        guidance: float = 1.0,
        nsteps: int = 50,
        # LanPaint parameters
        lambda_: float = 6.0,
        K: int = 5,
        # MCMC parameters
        mcmc_method: str = "tamed_ula",
        mcmc_args: dict | None = None,
        # Stability parameters
        max_score: float | None = None,
        # Schedule parameters
        rho: float = 7.0,
        # Output options
        return_trajectory: bool = False,
        debug: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Sample from posterior p(x|y₀) via LanPaint with MCMC sampling.

        At each noise level σ:
        1. Run K MCMC steps with BiG score to sample from q_{λ,σ}
        2. Take diffusion step to next noise level (warm start)

        The surrogate distribution q_{λ,σ} converges to true posterior as σ→0.

        Args:
            y_0: Known pixel values (normalized) [*shape]
            unknown_mask: 1=unknown (to inpaint), 0=known (observed) [*shape]
            nsamples: Number of samples to generate
            y: Optional conditioning dict for the model
            guidance: Classifier-free guidance scale

            nsteps: Number of noise levels in schedule

            lambda_: Guidance strength (λ). Higher = stronger constraint on known.
                     Recommended: 4.0 - 10.0
            K: MCMC steps per noise level. More = closer to q_{λ,σ}.
               Recommended: 5 - 10

            mcmc_method: MCMC method to use. One of:
                - "tamed_ula" (default): Tamed ULA, stable gradient bounding
                - "overdamped", "ula": Overdamped Langevin
                - "baoab": BAOAB splitting (stable underdamped)
                - "hmc": Hamiltonian Monte Carlo
                - "fld": Fast Langevin Dynamics
                - "mala": Metropolis-Adjusted Langevin
            mcmc_args: Method-specific arguments dict:
                - tamed_ula: {"h": step_size (default 0.1)}
                - overdamped: {"h": step_size (default 0.1)}
                - baoab: {"h": step_size, "gamma": friction (default 1.0)}
                - hmc: {"epsilon": leapfrog_step (default 0.01),
                        "n_leapfrog": num_steps (default 10)}
                - fld: {"h": step_size, "gamma": friction (default 15.0)}

            max_score: If set, clamp score magnitude to this value.
                       Prevents explosion from extreme gradients.

            rho: Power-law exponent for time schedule

            return_trajectory: If True, return (samples, trajectory)

        Returns:
            Samples from approximate posterior [batch, *shape]
        """
        mcmc_args = mcmc_args or {}
        device = self.device
        y_0 = y_0.to(device)
        unknown_mask = unknown_mask.to(device)
        if y is not None:
            y = dict_to(y, device)

        shape = list(y_0.shape)

        # Get time schedule
        time_schedule = self.create_time_schedule(nsteps, rho=rho).to(device)
        sigma_max = self.config.sigma_fn(time_schedule[0])
        alpha_max = self.config.alpha_fn(time_schedule[0])

        if debug:
            print(f"[LanPaint] sigma_max={sigma_max:.4f}, alpha_max={alpha_max:.4f}")
            print(f"[LanPaint] lambda={lambda_}, K={K}, mcmc_method={mcmc_method}, mcmc_args={mcmc_args}")
            print(f"[LanPaint] y_0: min={y_0.min():.4f}, max={y_0.max():.4f}")
            print(f"[LanPaint] unknown_mask: sum={unknown_mask.sum()}, total={unknown_mask.numel()}")

        # Initialize from noise at sigma_max
        x = torch.randn(nsamples, *shape, device=device) * sigma_max

        # Set known pixels to noisy version of y_0: α(σ_max)·y_0 + σ_max·ξ
        known_mask = 1.0 - unknown_mask
        y_noisy = alpha_max * y_0 + sigma_max * torch.randn_like(y_0)
        x = unknown_mask * x + known_mask * y_noisy

        if debug:
            print(f"[LanPaint] Initial x: min={x.min():.4f}, max={x.max():.4f}")

        trajectory = [x.clone()] if return_trajectory else None

        # Main loop: iterate through noise levels
        for i in range(len(time_schedule) - 1):
            t_curr = time_schedule[i] * torch.ones(nsamples, device=device)
            t_next = time_schedule[i + 1] * torch.ones(nsamples, device=device)
            sigma_curr = self.config.sigma_fn(t_curr[0])

            if debug:
                print(f"\n[LanPaint] === Step {i}/{len(time_schedule)-1}, t={t_curr[0]:.4f}, sigma={sigma_curr:.4f} ===")
                print(f"[LanPaint] x before Langevin: min={x.min():.4f}, max={x.max():.4f}, nan={x.isnan().any()}")

            # Step 1: MCMC correction at current noise level
            x = self._langevin_correction(
                x, t_curr, y_0, unknown_mask, lambda_, K,
                mcmc_method=mcmc_method, mcmc_args=mcmc_args, max_score=max_score,
                y_cond=y, guidance=guidance, debug=debug
            )

            if debug:
                print(f"[LanPaint] x after Langevin: min={x.min():.4f}, max={x.max():.4f}, nan={x.isnan().any()}")

            # Step 2: Diffusion step to next noise level (warm start)
            x = self._diffusion_step_lanpaint(x, t_curr, t_next, y_cond=y, guidance=guidance, debug=debug)

            if debug:
                print(f"[LanPaint] x after diffusion: min={x.min():.4f}, max={x.max():.4f}, nan={x.isnan().any()}")

            if return_trajectory:
                trajectory.append(x.clone())

            # Early termination if NaN detected
            if x.isnan().any():
                print(f"[LanPaint] ERROR: NaN detected at step {i}, terminating early")
                break

        # # Final Langevin correction (skip if sigma is already at minimum)
        # if not x.isnan().any():
        #     t_final = time_schedule[-1] * torch.ones(nsamples, device=device)
        #     sigma_final = self.config.sigma_fn(t_final[0]).item()

        #     if sigma_final >= sigma_min:
        #         if debug:
        #             print(f"\n[LanPaint] === Final Langevin correction, t={t_final[0]:.4f}, sigma={sigma_final:.4f} ===")
        #         x = self._langevin_correction(
        #             x, t_final, y_0, unknown_mask, lambda_,
        #             K, h, Gamma, use_fld, sigma_min=sigma_min, max_score=max_score,
        #             y_cond=y, guidance=guidance, debug=debug
        #         )
        #     else:
        #         if debug:
        #             print(f"\n[LanPaint] === Skipping final Langevin (sigma={sigma_final:.4f} < sigma_min={sigma_min:.4f}) ===")

        # Denormalize output
        x = self.initial_norm.unnorm(x)

        if return_trajectory:
            trajectory.append(x.clone())
            trajectory = [self.initial_norm.unnorm(t) for t in trajectory[:-1]] + [trajectory[-1]]
            return x, trajectory

        return x

    def inpaint_lanpaint(
        self,
        x_orig: Float[Tensor, "*shape"],  # noqa: F821
        mask: Float[Tensor, "*shape"],  # noqa: F821
        nsamples: int = 1,
        y: None | dict = None,
        guidance: float = 1.0,
        nsteps: int = 50,
        # LanPaint parameters
        lambda_: float = 6.0,
        K: int = 5,
        # MCMC parameters
        mcmc_method: str = "tamed_ula",
        mcmc_args: dict | None = None,
        # Stability parameters
        max_score: float | None = None,
        # Mask parameters
        mask_falloff: int = 0,
        # Schedule parameters
        rho: float = 7.0,
        # Output options
        return_trajectory: bool = False,
        debug: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Inpainting via LanPaint with MCMC sampling.

        Convenience wrapper around sample_posterior_lanpaint.

        Args:
            x_orig: Original data with known regions [*shape]
            mask: Binary mask, 1=known, 0=unknown [*shape]
                  (Same convention as inpaint_dps)
            nsamples: Number of samples to generate
            y: Optional conditioning dict for the model
            guidance: Classifier-free guidance scale

            nsteps: Number of noise levels

            lambda_: Guidance strength (λ). Recommended: 4.0 - 10.0
            K: MCMC steps per noise level. Recommended: 5 - 10

            mcmc_method: MCMC method to use. One of:
                - "tamed_ula" (default): Tamed ULA, stable gradient bounding
                - "overdamped", "ula": Overdamped Langevin
                - "baoab": BAOAB splitting (stable underdamped)
                - "hmc": Hamiltonian Monte Carlo
                - "fld": Fast Langevin Dynamics
                - "mala": Metropolis-Adjusted Langevin
            mcmc_args: Method-specific arguments dict:
                - tamed_ula: {"h": step_size (default 0.1)}
                - overdamped: {"h": step_size (default 0.1)}
                - baoab: {"h": step_size, "gamma": friction (default 1.0)}
                - hmc: {"epsilon": leapfrog_step, "n_leapfrog": num_steps}
                - fld: {"h": step_size, "gamma": friction}

            max_score: If set, clamp score magnitude to this value.
                       Prevents explosion. Try 10.0 - 100.0.

            mask_falloff: Soft mask gradient width (0 = hard mask)

            rho: Power-law exponent for time schedule

            return_trajectory: If True, return (samples, trajectory)

        Returns:
            Inpainted samples [batch, *shape]
        """
        warnings.warn("Assuming latent space operation for LanPaint inpainting")

        device = self.device
        x_orig = x_orig.to(device)
        mask = mask.to(device)

        # Create soft mask if specified
        if mask_falloff > 0:
            soft_mask = self._create_soft_mask(mask, mask_falloff)
        else:
            soft_mask = mask

        # Normalize x_orig
        x_orig_normed = self.initial_norm(x_orig.unsqueeze(0)).squeeze(0)

        # y_0 = known pixel values (where mask=1)
        # We pass the full normalized image; the algorithm uses known_mask to select
        y_0 = x_orig_normed

        # unknown_mask: 1=unknown, 0=known
        # Our mask convention: 1=known, 0=unknown
        # So unknown_mask = 1 - mask
        unknown_mask = 1.0 - soft_mask

        return self.sample_posterior_lanpaint(
            y_0=y_0,
            unknown_mask=unknown_mask,
            nsamples=nsamples,
            y=y,
            guidance=guidance,
            nsteps=nsteps,
            lambda_=lambda_,
            K=K,
            mcmc_method=mcmc_method,
            mcmc_args=mcmc_args,
            max_score=max_score,
            rho=rho,
            return_trajectory=return_trajectory,
            debug=debug
        )

    def integrate_flow_field(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        time_schedule: Float[Tensor, "nsteps"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        return_history: bool = False,
        integrate_on_sigma: bool = False,
        noise_injection: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        # Integrate the flow field x' = v(x, t) using the Heun method
        self.model.eval()
        if return_history:
            history = [(time_schedule[0], x)]

        for i in range(len(time_schedule) - 1):
            t_curr = time_schedule[i] * torch.ones(x.shape[0]).to(x)
            t_next = time_schedule[i + 1] * torch.ones(x.shape[0]).to(x)

            if noise_injection:
                method = 'euler_maruyama'
            else:
                method = 'euler' if i == len(time_schedule) - 2 else 'heun'

            x = self.integration_step(
                x,
                t_curr,
                t_next,
                y,
                guidance,
                method=method,
                integrate_on_sigma=integrate_on_sigma,
                noise_injection=noise_injection
            )

            if return_history:
                history.append((time_schedule[i + 1], x))

        if not return_history:
            x = self.initial_norm.unnorm(x)
            return x
        else:
            history = list(map(lambda tx: (tx[0], self.initial_norm.unnorm(tx[1])), history))
            return history

    def integration_step(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        t_curr: Float[Tensor, "batch"],  # noqa: F821, typing
        t_next: Float[Tensor, "batch"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        method: Literal['euler', 'heun', 'euler_maruyama'] = 'euler',
        integrate_on_sigma: bool = False,
        noise_injection: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing

        if not integrate_on_sigma:
            dt = t_next - t_curr
        else:
            dt = self.config.sigma_fn(t_next) - self.config.sigma_fn(t_curr)
        dt = broadcast_from_below(dt, x)

        # Euler method
        if method in ['euler', 'heun']:
            assert not noise_injection, "Noise injection is not supported for Euler and Heun methods"

        if method == 'euler':
            v = self.get_flow_field(x, t_curr, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            return x + dt * v
        elif method == 'heun':
            # Heun method
            # First step - Euler
            v1 = self.get_flow_field(x, t_curr, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            x_euler = x + dt * v1

            # Second step - correction
            v2 = self.get_flow_field(x_euler, t_next, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            return x + dt * (v1 + v2) / 2
        elif method == 'euler_maruyama':
            if not noise_injection:
                raise ValueError("Noise injection is required for Euler-Maruyama method")
            v = self.get_flow_field(x, t_curr, y=y, guidance=guidance, integrate_on_sigma=integrate_on_sigma)
            score_field = self.get_score_field_from_flow_field(v, x, t_curr)
            omega = self.config.sigma_fn(t_curr)  # TODO: Allow for more complex integration methods
            omega = broadcast_from_below(omega, x)
            x = x + dt * (v - 0.5 * omega * score_field)
            noise = torch.sqrt(omega * torch.abs(dt)) * torch.randn_like(x)
            x = x + noise
            return x
        else:
            raise ValueError(f"Invalid integration method: {method}")

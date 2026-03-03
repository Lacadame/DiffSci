from typing import Literal, Callable, Any
from jaxtyping import Float
from torch import Tensor

import warnings

import torch
import torch.nn as nn
import numpy as np
import lightning

from diffsci.torchutils import broadcast_from_below, dict_unsqueeze, dict_to
from diffsci.models.aux_scripts import DimensionAgnosticBatchNorm, IdentityBatchNorm


SampleType = Float[Tensor, "batch *shape"]
ConditionType = Float[Tensor, "batch *yshape"]
TimeType = Float[Tensor, "batch"]


class SIScheduler(object):
    def __init__(
        self,
        alpha_fn: Callable[[float], float],
        sigma_fn: Callable[[float], float],
        alpha_fn_dot: Callable[[float], float],
        sigma_fn_dot: Callable[[float], float],
        sigma_fn_inv: Callable[[float], float]
    ):
        self.alpha_fn = alpha_fn
        self.sigma_fn = sigma_fn
        self.alpha_fn_dot = alpha_fn_dot
        self.sigma_fn_dot = sigma_fn_dot
        self.sigma_fn_inv = sigma_fn_inv

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

    @classmethod
    def named_interpolators(cls):
        return ['linear', 'cosine', 'edm', 'finterpolation']


class Preconditioner(object):
    def __init__(
        self,
        scheduler: SIScheduler,
        precondition_fn: Literal['identity', 'edm'] | Callable | None = 'identity',
        is_autonomous: bool = False,
        **kwargs
    ):
        self.scheduler = scheduler
        self.precondition_fn = precondition_fn
        self.is_autonomous = is_autonomous
        self.kwargs = kwargs

    def __call__(self, model, x, t=None, y=None):
        return self.get_flow_field(model, x, t, y)

    def get_flow_field(self, model, x, t=None, y=None):
        if self.precondition_fn is None:
            v = self.identity(model, x, t, y)
        elif isinstance(self.precondition_fn, str):
            if self.precondition_fn == 'identity':
                v = self.identity(model, x, t, y)
            elif self.precondition_fn == 'edm':
                v = self.edm(model, x, t, y)
            else:
                raise ValueError(f"Invalid condition function: {self.precondition_fn}")
        else:
            if self.is_autonomous:
                v = self.precondition_fn(model, x, y=y)
            else:
                v = self.precondition_fn(model, x, t, y=y)
        return v

    def identity(self, model, x, t=None, y=None):
        if self.is_autonomous:
            return model(x, y=y)
        else:
            return model(x, t, y=y)

    def edm(self, model, x, t=None, y=None):
        sigma_data = self.kwargs.get("sigma_data", 0.5)
        sigma = self.scheduler.sigma_fn(t)
        sigma_dot = self.scheduler.sigma_fn_dot(t)
        sigma = broadcast_from_below(sigma, x)
        sigma_dot = broadcast_from_below(sigma_dot, x)
        cin = 1 / torch.sqrt(sigma_data**2 + sigma**2)
        cout = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        cskip = sigma_data ** 2 / (sigma_data**2 + sigma**2)
        if self.is_autonomous:
            flow_field = cskip * x + cout * model(x / cin, y=y)
        else:
            cnoise = 0.5 * torch.log(self.scheduler.sigma_fn(t))
            denoiser = cskip * x + cout * model(cin * x, cnoise, y=y)

            flow_field = sigma_dot / sigma * (x - denoiser)
        return flow_field


class LossWeighting(object):
    def __init__(
        self,
        scheduler: SIScheduler,
        weighting_class: Literal['edm', 'uniform'] | dict[str, Any] = 'uniform',
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
            elif self.weighting_class == 'uniform':
                return self.uniform_weighting_sampler(nsamples)
            else:
                raise ValueError(f"Invalid weighting class: {self.weighting_class}")
        else:
            return self.weighting_class['weighting_sampler'](nsamples)

    def uniform_weighting_function(self, t):
        return 1.0 + 0.0 * t

    def uniform_weighting_sampler(self, nsamples):
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
        pmean = self.kwargs.get("pmean", -1.2)
        pstd = self.kwargs.get("pstd", 1.2)
        logsigma = pstd * torch.randn(nsamples) + pmean
        sigma = torch.exp(logsigma)
        t = self.scheduler.sigma_fn_inv(sigma)
        return t


class SIModuleConfig(torch.nn.Module):
    def __init__(self,
                 scheduler: SIScheduler | str = 'linear',
                 scheduler_args: dict[str, Any] = {},
                 num_channels: int | None = None,
                 initial_norm: bool = False,
                 autonomous_flow: bool = False,
                 precondition_fn: Callable | str | None = None,
                 loss_weighting: Literal['edm', 'uniform'] | dict[str, Any] = 'uniform',
                 loss_metric: Literal['mse', 'huber'] = 'huber',
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
        self.loss_weighting = loss_weighting
        self.loss_metric = loss_metric
        self.precondition_fn = precondition_fn
        self.autoencoder_is_conditional = autoencoder_is_conditional
        self.encode_condition = encode_condition
        self.set_scheduling_functions()
        self.set_loss_metric_module()
        self.set_preconditioner()
        self.set_loss_weighting()

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
        self.preconditioner = Preconditioner(self.scheduler, self.precondition_fn, self.autonomous_flow)

    def set_loss_weighting(self):
        if isinstance(self.loss_weighting, str):
            self.loss_weighting = LossWeighting(self.scheduler, self.loss_weighting)
        else:
            self.loss_weighting = LossWeighting(self.scheduler, **self.loss_weighting)


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
        if self.config.initial_norm:
            self.initial_norm = DimensionAgnosticBatchNorm(self.config.num_channels)
        else:
            self.initial_norm = IdentityBatchNorm()

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
        flow_field = self.get_flow_field(x_noised, t, y=y, guidance=1.0)

        alpha_dot, sigma_dot = self.config.alpha_fn_dot(t_broadcasted), self.config.sigma_fn_dot(t_broadcasted)
        target = (alpha_dot * x + sigma_dot * noise)

        loss = self.config.loss_metric_module(flow_field, target)
        loss_weighting = self.config.loss_weighting.weighting_function(t_broadcasted)
        loss = loss * loss_weighting

        if mask is not None:
            # Apply the mask if it is provided
            # We assume that the mask is 1 where the data is absent
            mask = mask.expand_as(loss)
            loss = loss * (1 - mask)
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

    def sample(self,
               nsamples: int,
               shape: list[int],
               y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
               guidance: float = 1.0,
               nsteps: int = 30,
               is_latent_shape: bool = False,
               integrate_on_sigma: bool = False,
               noise_injection: bool = False,
               return_latents: bool = False
               ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        if torch.inference_mode():
            with torch.no_grad():
                x = torch.randn(nsamples, *shape).to(self.device)
                if y is not None:
                    warnings.warn("Moving y to device: {}".format(self.device))
                    y = dict_to(y, self.device)
                if not is_latent_shape and self.autoencoder:
                    # Need to do a stupid hack for getting correct shape
                    x, _ = self.encode(x, y)
                    x = torch.randn_like(x)
                if y is not None:
                    y = dict_unsqueeze(y, 0)
                time_schedule = torch.linspace(1, 0, nsteps).to(x)
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

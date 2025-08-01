from typing import Callable, Any

import torch
import lightning
from torch import Tensor
from jaxtyping import Float, Bool
from typing import Tuple, Union, Dict, Any, Callable, Optional

from diffsci.torchutils import (broadcast_from_below,
                                linear_interpolation,
                                dict_unsqueeze,
                                dict_to)
from diffsci.utils import get_minibatch_sizes
from diffsci.custom_losses import GaussianWeightedMSELoss, MultiSpaceLoss, MultiThresholdSmoothIndicatorLoss
from . import preconditioners
from . import noisesamplers
from . import schedulers
from . import edmbatchnorm
from . import integrators


Scaler = Callable[[Float[Tensor, '*shape']],  # noqa: F821
                  Float[Tensor, '*shape']]  # noqa: F821
Sampler = Callable[[list[int]], Float[Tensor, '...']]


class KarrasModuleConfig(object):
    def __init__(self,
                 preconditioner: preconditioners.KarrasPreconditioner,
                 noisesampler: noisesamplers.NoiseSampler,
                 noisescheduler: schedulers.Scheduler,
                 loss_metric: Union[str, Dict[str, Any]] = "huber",
                 tag: str = "custom",
                 has_edm_batch_norm: bool = False,
                 dynamic_loss_weight: int | None = None,
                 extra_args: None | dict[str, Any] = None,
                 # Legacy parameters for backward compatibility
                 spatial_shape: tuple = None,
                 focus_radius: float = None):
        """
        Args:
            preconditioner: Karras preconditioner
            noisesampler: Noise sampler
            noisescheduler: Scheduler
            loss_metric: Loss configuration - can be:
                - str: "mse", "huber", etc. (backward compatible)
                - dict: {"smoothed_indicator": {"thresholds": [0.5, 1.0], ...}}
                - dict: {"losses": [{"name": "loss1", "type": "mse", ...}, ...]}
            tag: Configuration tag
            has_edm_batch_norm: Whether to use EDM batch norm
            dynamic_loss_weight: Dynamic loss weight configuration
            extra_args: Extra arguments for reconstruction
            spatial_shape: Legacy parameter for weighted_gaussian loss
            focus_radius: Legacy parameter for weighted_gaussian loss
        """
        self.preconditioner = preconditioner
        self.noisesampler = noisesampler
        self.noisescheduler = noisescheduler
        self.loss_metric = loss_metric
        self.tag = tag
        self.has_edm_batch_norm = has_edm_batch_norm
        self.dynamic_loss_weight = dynamic_loss_weight
        self.spatial_shape = spatial_shape
        self.focus_radius = focus_radius
        
        if extra_args is None:
            self.extra_args = dict()
        else:
            self.extra_args = extra_args

    @classmethod
    def from_edm(cls,
                 sigma_data: float = 0.5,
                 prior_mean: float = -1.2,
                 prior_std: float = 1.2,
                 has_edm_batch_norm: bool = False,
                 dynamic_loss_weight: int | None = None,
                 loss_metric: Union[str, Dict[str, Any]] = "huber",
                 spatial_shape: tuple = None,
                 focus_radius: float = None):
        """
        Create EDM configuration with flexible loss support.
        
        Args:
            sigma_data: Sigma data parameter
            prior_mean: Prior mean
            prior_std: Prior standard deviation
            has_edm_batch_norm: Whether to use EDM batch norm
            dynamic_loss_weight: Dynamic loss weight
            loss_metric: Loss configuration (new flexible format)
            spatial_shape: For weighted_gaussian loss (legacy)
            focus_radius: For weighted_gaussian loss (legacy)
        """
        preconditioner = preconditioners.EDMPreconditioner(
                            sigma_data=sigma_data
                        )
        noisesampler = noisesamplers.EDMNoiseSampler(
                            sigma_data=sigma_data,
                            prior_mean=prior_mean,
                            prior_std=prior_std
                        )
        noisescheduler = schedulers.EDMScheduler()
        tag = "edm"
        extra_args = {
            "sigma_data": sigma_data,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "loss_metric": loss_metric,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   tag=tag,
                   has_edm_batch_norm=has_edm_batch_norm,
                   dynamic_loss_weight=dynamic_loss_weight,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius)

    @classmethod
    def from_vp(cls,
                beta_data: float = 19.9,
                beta_min: float = 0.1,
                epsilon_min: float = 1e-3,
                epsilon_sampler: float = 1e-5,
                M: int = 1000,
                loss_metric: Union[str, Dict[str, Any]] = "huber",
                spatial_shape: tuple = None,
                focus_radius: float = None):
        """
        Create VP configuration with flexible loss support.
        """
        noisescheduler = schedulers.VPScheduler(epsilon_min=epsilon_min,
                                                beta_data=beta_data,
                                                beta_min=beta_min)
        preconditioner = preconditioners.VPPreconditioner(
                            scheduler=noisescheduler,
                            M=M
                            )
        noisesampler = noisesamplers.VPNoiseSampler(
            noise_scheduler=noisescheduler,
            epsilon=epsilon_sampler
        )
        tag = "vp"
        extra_args = {
            "beta_data": beta_data,
            "beta_min": beta_min,
            "epsilon_min": epsilon_min,
            "epsilon_sampler": epsilon_sampler,
            "M": M,
            "loss_metric": loss_metric,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   tag=tag,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius)

    @classmethod
    def from_ve(cls,
                sigma_min: float = 0.02,
                sigma_max: float = 100,
                loss_metric: Union[str, Dict[str, Any]] = "huber",
                spatial_shape: tuple = None,
                focus_radius: float = None):
        """
        Create VE configuration with flexible loss support.
        """
        noisescheduler = schedulers.VEScheduler(sigma_min=sigma_min,
                                                sigma_max=sigma_max)
        preconditioner = preconditioners.VEPreconditioner()
        noisesampler = noisesamplers.VENoiseSampler(
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        tag = "ve"
        extra_args = {
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "loss_metric": loss_metric,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   tag=tag,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius)

    @classmethod
    def conditionalSR3(cls,
                       sigma_min: float = 0.02,
                       sigma_max: float = 100,
                       loss_metric: Union[str, Dict[str, Any]] = "huber",
                       spatial_shape: tuple = None,
                       focus_radius: float = None):
        """
        Create conditional SR3 configuration with flexible loss support.
        """
        noisescheduler = schedulers.EDMScheduler(sigma_min=sigma_min,
                                                 sigma_max=sigma_max)
        preconditioner = preconditioners.SR3Preconditioner()
        noisesampler = noisesamplers.EDMNoiseSampler(
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        tag = "conditionalSR3"
        extra_args = {
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "loss_metric": loss_metric,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   tag=tag,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius)

    def export_description(self) -> dict[str, Any]:
        """Export configuration for saving/loading."""
        return dict(tag=self.tag,
                    extra_args=self.extra_args)

    @classmethod
    def load_from_description_with_tag(cls,
                                       description: dict[str, Any]):
        """Load configuration from saved description."""
        tag = description["tag"]
        extra_args = description["extra_args"]
        if tag == "custom":
            raise ValueError("Cannot load from a custom tag")
        elif tag == "edm":
            return cls.from_edm(**extra_args)
        elif tag == "vp":
            return cls.from_vp(**extra_args)
        elif tag == "ve":
            return cls.from_ve(**extra_args)
        elif tag == "conditionalSR3":
            return cls.conditionalSR3(**extra_args)
        else:
            raise ValueError(f"Unknown tag: {tag}")

    @property
    def has_dynamic_loss_weight(self):
        """Check if dynamic loss weight is enabled."""
        return self.dynamic_loss_weight is not None

    def update_loss_metric(self, loss_config: Union[str, Dict[str, Any]]):
        """
        Update the loss metric configuration.
        
        Useful for programmatically changing loss configuration after creation.
        
        Args:
            loss_config: New loss configuration
        """
        self.loss_metric = loss_config
        # Update extra_args for proper serialization
        if 'loss_metric' in self.extra_args:
            self.extra_args['loss_metric'] = loss_config

    def get_loss_summary(self) -> str:
        """
        Get a human-readable summary of the loss configuration.
        """
        if isinstance(self.loss_metric, str):
            return f"Single loss: {self.loss_metric}"
        elif isinstance(self.loss_metric, dict):
            if "losses" in self.loss_metric:
                n_losses = len(self.loss_metric["losses"])
                loss_names = [loss["name"] for loss in self.loss_metric["losses"]]
                return f"Multi-space loss: {n_losses} losses ({', '.join(loss_names)})"
            else:
                loss_name = list(self.loss_metric.keys())[0]
                return f"Single advanced loss: {loss_name}"
        else:
            return f"Unknown loss type: {type(self.loss_metric)}"



class KarrasModule(lightning.LightningModule):
    """Updated KarrasModule with multi-space loss support"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 config: 'KarrasModuleConfig',
                 conditional: bool = False,
                 masked: bool = False,
                 autoencoder: None | torch.nn.Module = None,
                 autoencoder_conditional: bool = False,
                 encode_y: bool = False,
                 decode_original_y: bool = False):
        super().__init__()
        self.model = model
        self.config = config
        self.conditional = conditional
        self.masked = masked
        self.autoencoder = autoencoder
        if self.autoencoder:
            self.freeze_autoencoder()
        self.autoencoder_conditional = autoencoder_conditional
        self.encode_y = encode_y
        self.decode_original_y = decode_original_y
        self.set_optimizer_and_scheduler()
        self.set_loss_metric()
        self.start_edm_batch_norm()
        self.start_dynamic_loss_weight()
        self.norm = 1.0

    def freeze_autoencoder(self):
        """Freezes the autoencoder to prevent its weights from being updated during training."""
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def export_description(self) -> dict[str, Any]:
        config_description = self.config.export_description()
        conditional = self.conditional
        masked = self.masked
        autoencoder = True if self.autoencoder else False
        autoencoder_conditional = self.autoencoder_conditional
        encode_y = self.encode_y
        return dict(config_description=config_description,
                    conditional=conditional,
                    masked=masked,
                    autoencoder=autoencoder,
                    autoencoder_conditional=autoencoder_conditional,
                    encode_y=encode_y)

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
            self.optimizer = torch.optim.AdamW(self.parameters(),
                                               lr=1e-3,
                                               betas=(0.9, 0.999),
                                               weight_decay=1e-4)
        if scheduler is not None:
            self.lr_scheduler = scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                                    self.optimizer,
                                    lr_lambda=lambda step: 1.0 + 0*step
                                )  # Neutral scheduler
        self.lr_scheduler_interval = scheduler_interval

    def set_loss_metric(self):
        """
        Set the loss function(s) to be used.
        
        Supports three formats:
        1. String: "mse" (backward compatible)
        2. Dict with single loss: {"smoothed_indicator": {...}} (single loss)
        3. Dict with multiple losses: {"losses": [...]} (multi-space losses)
        """
        
        loss_config = self.config.loss_metric
        
        if isinstance(loss_config, str):
            # Original string format - create single loss
            self._set_single_loss_string(loss_config)
            
        elif isinstance(loss_config, dict):
            if "losses" in loss_config:
                # Multi-space loss configuration
                self.multi_space_loss = MultiSpaceLoss(loss_config, self.autoencoder)
                self.loss_metric = None  # Will use multi_space_loss instead
            else:
                # Single loss dict format
                self._set_single_loss_dict(loss_config)
        else:
            raise ValueError(f"loss_metric must be string or dict, got {type(loss_config)}")

    def _set_single_loss_string(self, loss_name: str):
        """Handle single loss string format (backward compatible)"""
        if loss_name == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif loss_name == "huber":
            self.loss_metric = torch.nn.HuberLoss(reduction="none")
        elif loss_name == "weighted_gaussian":
            if self.config.spatial_shape is None or self.config.focus_radius is None:
                raise AttributeError("config must have shape tuple and focus radius")
            self.loss_metric = GaussianWeightedMSELoss(shape=self.config.spatial_shape,
                                                       focus_radius=self.config.focus_radius)
        elif loss_name == "smoothed_indicator":
            self.loss_metric = MultiThresholdSmoothIndicatorLoss()
        else:
            raise ValueError(f"loss_type {loss_name} not recognized")

    def _set_single_loss_dict(self, loss_config: Dict[str, Any]):
        """Handle single loss dict format"""
        loss_name = list(loss_config.keys())[0]
        loss_params = loss_config[loss_name]
        
        if loss_name == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif loss_name == "huber":
            delta = loss_params.get('delta', 1.0)
            self.loss_metric = torch.nn.HuberLoss(reduction="none", delta=delta)
        elif loss_name == "smoothed_indicator":
            self.loss_metric = MultiThresholdSmoothIndicatorLoss(**loss_params)
        # ... add other loss types as needed
        else:
            raise ValueError(f"loss_name '{loss_name}' not recognized")

    def loss_fn(self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821
            mask: None | Float[Tensor, "batch *shape"] = None  # noqa: F821
            ) -> Float[Tensor, ""]:  # noqa: F821, F722
        """
        Loss function with support for multi-space losses and smart mask handling.
        """
        
        # Store original pixel space data
        x_pixel = x.clone()
        mask_pixel = mask.clone() if mask is not None else None
        
        # Encode to latent space
        if self.encode_y:
            x_latent, y = self.encode(x, y)
        else:
            x_latent = self.encode(x, y)
        
        # Add noise and get denoiser output
        broadcasted_sigma = broadcast_from_below(sigma, x_latent)
        noise = broadcasted_sigma * torch.randn_like(x_latent)
        x_noised = x_latent + noise
        denoiser_latent, cond_noise = self.get_denoiser(x_noised, sigma, y)
        
        # Compute loss weighting
        weight = self.config.noisesampler.loss_weighting(broadcasted_sigma)
        bias = torch.zeros_like(weight)
        if self.config.has_dynamic_loss_weight:
            modifier = self.dynamic_loss_weight(cond_noise)
            modifier = broadcast_from_below(modifier, x_latent)
            weight = weight / torch.exp(modifier)
            bias = bias + modifier

        # Check if using multi-space loss system
        if hasattr(self, 'multi_space_loss') and self.multi_space_loss is not None:
            # Multi-space loss computation
            loss_results = self.multi_space_loss.compute_loss(
                denoiser_latent=denoiser_latent,
                target_latent=x_latent,
                target_pixel=x_pixel,
                mask_latent=mask,
                mask_pixel=mask_pixel
            )
            
            total_loss = loss_results["total"]
            
            # Check if total_loss is scalar (mask already handled) or tensor (needs mask handling)
            if total_loss.dim() == 0:
                # Scalar loss - mask already handled internally
                final_loss = weight.mean() * total_loss + bias.mean()
            else:
                # Tensor loss - apply mask externally
                if mask is not None:
                    mask_expanded = mask.expand_as(total_loss)
                    adjusted_loss = total_loss * (1 - mask_expanded)
                    final_loss = (weight * adjusted_loss + bias).mean()
                else:
                    final_loss = (weight * total_loss + bias).mean()
            
            return final_loss
        
        else:
            # Single loss computation (backward compatible)
            # Check if the loss function accepts mask parameter
            loss = self._compute_single_loss(denoiser_latent, x_latent, mask)
            
            # Check if loss is scalar (mask handled internally) or tensor (needs external mask)
            if loss.dim() == 0:
                # Scalar loss - mask already handled internally (e.g., smoothed_indicator)
                final_loss = weight.mean() * loss + bias.mean()
            else:
                # Tensor loss - apply mask externally (e.g., MSE, Huber)
                if mask is not None:
                    mask_expanded = mask.expand_as(loss)
                    adjusted_loss = loss * (1 - mask_expanded)
                    final_loss = (weight * adjusted_loss + bias).mean()
                else:
                    final_loss = (weight * loss + bias).mean()
            
            return final_loss        
    
    def _compute_single_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simpler version - just try with mask first, fallback without mask.
        """
        try:
            # Try calling with mask parameter first
            return self.loss_metric(pred, target, mask)
        except TypeError:
            # If that fails, call without mask parameter
            return self.loss_metric(pred, target)

    def get_denoiser(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821,
            guidance: float = 1.0
            ) -> tuple[Float[Tensor, "batch *shape"],  # noqa: F821
                       Float[Tensor, "batch"]]:  # noqa: F821
        """
        Parameters
        ---------
        x : torch.Tensor of shape [B, *[shapex]], the original noise
        sigma : torch.Tensor of shape [B]
        y : None or torch.Tensor of shape [B, *[yshape]], the conditional data,
        depending on whether we are dealing with a conditional or unconditional
        model
        """
        input_scale_factor = self.config.preconditioner.input_scaling(
            sigma)  # [nbatch]
        input_scale_factor = broadcast_from_below(input_scale_factor,
                                                  x)  # [nbatch, *1]
        output_scale_factor = self.config.preconditioner.output_scaling(
            sigma)  # [nbatch]
        output_scale_factor = broadcast_from_below(output_scale_factor,
                                                   x)  # [nbatch, *1]
        skip_scale_factor = self.config.preconditioner.skip_scaling(
            sigma)  # [nbatch]
        skip_scale_factor = broadcast_from_below(skip_scale_factor,
                                                 x)  # [nbatch, *1]
        scaled_input = input_scale_factor*x  # [nbatch, *shapex]
        cond_noise = self.config.preconditioner.noise_conditioner(
            sigma)  # [nbatch]
        if self.conditional and guidance != 0.0:
            base_score = self.model(scaled_input,
                                    cond_noise,
                                    y)  # [nbatch, *shape]
            if guidance != 1.0:
                uncond_base_score = self.model(scaled_input,
                                               cond_noise)
                base_score = ((1 - guidance)*uncond_base_score +
                              guidance*base_score)
        else:
            base_score = self.model(scaled_input,
                                    cond_noise)  # [nbatch, *shape]
        scaled_output = output_scale_factor*base_score  # [nbatch, *shape]
        denoiser = scaled_output + skip_scale_factor*x  # [nbatch, *shape]
        return denoiser, cond_noise

    def get_score(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821
            guidance: float = 1.0
            ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        denoiser, _ = self.get_denoiser(x,
                                        sigma,
                                        y,
                                        guidance)  # [nbatch, *shapex]
        sigma_broadcasted = broadcast_from_below(sigma, x)  # [nbatch, *shapex]
        return (denoiser - x)/(sigma_broadcasted**2)

    def sample_and_filter(
            self,
            nsamples: int,
            shape: list[int],
            filter_fn: Callable[[Float[Tensor, "nsamples *shape"]],  # noqa: F821
                                Bool[Tensor, "*shape"]],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            guidance: float = 1.0,
            nsteps: int = 100,
            record_history: bool = False,
            maximum_batch_size: None | int = None,
            integrator: None | str | integrators.Integrator = None,
            move_to_cpu: bool = False,
            return_only_positives: bool = False
            ) -> dict[str, Any]:  # TODO: Put the actual type
        if record_history:
            raise ValueError("record_history is not supported for filtering at the moment")
        if maximum_batch_size is not None:
            batch_sizes = get_minibatch_sizes(nsamples, maximum_batch_size)
            samples = []
            filters = []
            num_positive = 0
            for batch_size in batch_sizes:
                result = self.sample_and_filter(
                    batch_size,
                    shape,
                    filter_fn,
                    y,
                    guidance,
                    nsteps,
                    record_history,
                    maximum_batch_size=None,
                    integrator=integrator,
                    return_only_positives=return_only_positives,
                    move_to_cpu=move_to_cpu
                )
                samples.append(result["samples"])
                filters.append(result["filter"])
                num_positive += result['filter'].sum().item()
            hit_rate = num_positive / nsamples
            catdim = 1 if record_history else 0
            samples = torch.cat(samples, dim=catdim)
            filters = torch.cat(filters, dim=catdim)
            return dict(samples=samples, filter=filters, hit_rate=hit_rate)
        else:
            samples = self.sample(
                nsamples,
                shape,
                y=y,
                guidance=guidance,
                nsteps=nsteps,
                record_history=record_history,
                maximum_batch_size=maximum_batch_size,
                integrator=integrator,
                move_to_cpu=False  # Moving to CPU will be done after encoding
            )
            with torch.inference_mode():
                filter = filter_fn(self.encode(samples, y, record_history))
            if return_only_positives:
                samples = samples[filter]
                filter = filter[filter]
            if move_to_cpu:
                samples = samples.detach().cpu()
            hit_rate = filter.sum()/nsamples
            return dict(samples=samples, filter=filter, hit_rate=hit_rate)

    def sample(
            self,
            nsamples: int,
            shape: list[int],
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            guidance: float = 1.0,
            nsteps: int = 100,
            record_history: bool = False,
            maximum_batch_size: None | int = None,
            integrator: None | str | integrators.Integrator = None,
            move_to_cpu: bool = False,
            is_latent_shape: bool = False,
            squeeze_memory_efficiency: bool = False,
            return_in_latent_space: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        with torch.inference_mode():
            if maximum_batch_size is not None:
                batch_sizes = get_minibatch_sizes(nsamples, maximum_batch_size)
                result = []
                for batch_size in batch_sizes:
                    result.append(self.sample(batch_size,
                                              shape,
                                              y,
                                              guidance,
                                              nsteps,
                                              record_history,
                                              maximum_batch_size=None,
                                              integrator=integrator,
                                              move_to_cpu=move_to_cpu,
                                              is_latent_shape=is_latent_shape,
                                              squeeze_memory_efficiency=squeeze_memory_efficiency,
                                              return_in_latent_space=return_in_latent_space))
                catdim = 1 if record_history else 0
                result = torch.cat(result, dim=catdim)
                return result
            else:
                batched_shape = [nsamples] + list(shape)
                white_noise = torch.randn(*batched_shape).to(self.device)
                if y is not None:
                    y = dict_to(y, self.device)
                # Ideally we do not enter here and is_latent_shape is True
                if self.latent_model and not is_latent_shape:  # TODO: A stupid hack. Should be improved
                    if self.encode_y:  # FIXME: What the fuck is this code?
                        if self.decode_original_y:
                            original_y = y.copy()
                        white_noise, y = self.encode(white_noise, y)
                        y['y'] = y['y'].squeeze(0)
                    else:
                        white_noise = self.encode(white_noise, y)
                    white_noise = torch.randn_like(white_noise)
                    # print(white_noise.shape)
                    # raise ValueError("Stop here")
                result = self.propagate_white_noise(
                            white_noise,
                            y,
                            guidance,
                            nsteps,
                            record_history,
                            integrator=integrator,
                            original_y=original_y if self.decode_original_y else None,
                            move_to_cpu=move_to_cpu,
                            latent_shape=is_latent_shape,
                            squeeze_memory_efficiency=squeeze_memory_efficiency,
                            return_in_latent_space=return_in_latent_space)
                return result

    def propagate_white_noise(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            guidance: float = 1.0,
            nsteps: int = 100,
            record_history: bool = False,
            integrator: None | str | integrators.Integrator = None,
            original_y: None | dict[str, Float[Tensor, "*yshape"]] = None,  # noqa: F821
            move_to_cpu: bool = False,
            latent_shape: bool = False,
            squeeze_memory_efficiency: bool = False,
            return_in_latent_space: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        x = x*self.config.noisescheduler.maximum_scale
        with torch.inference_mode():
            result = self.propagate_toward_sample(x,
                                                  y,
                                                  guidance,
                                                  nsteps,
                                                  record_history,
                                                  integrator=integrator)
        if squeeze_memory_efficiency:
            torch.cuda.empty_cache()
            self.model.to("cpu")
            self.autoencoder.encoder.to("cpu")
        with torch.inference_mode():
            if not return_in_latent_space:
                if original_y is not None:
                    result = self.decode(result, original_y, record_history)
                else:
                    result = self.decode(result, y, record_history)
        if move_to_cpu:
            result = result.detach().cpu()
        if squeeze_memory_efficiency:
            self.model.to(self.device)
            self.autoencoder.encoder.to(self.device)

        return result

    def propagate_toward_sample(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            guidance: float = 1.0,
            nsteps: int = 100,
            record_history: bool = False,
            integrator: None | str | integrators.Integrator = None
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        if y is not None:
            y = dict_unsqueeze(y, 0)  # Broadcasting will take care of the rest

        def rhs(x, sigma):
            with torch.inference_mode():
                return self.get_score(x, sigma, y, guidance)
        if integrator is not None:
            self.config.noisescheduler.set_temporary_integrator(integrator)
        result = self.config.noisescheduler.propagate_backward(
                                            x,
                                            rhs,
                                            nsteps,
                                            record_history=record_history)
        if integrator is not None:
            self.config.noisescheduler.unset_temporary_integrator()
        return result

    def propagate_partial_toward_sample(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            initial_step: int,
            final_step: int = None,
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            nsteps: int = 100,
            record_history: bool = False,
            integrator: None | str | integrators.Integrator = None,
            analytical_score=None,
            interp_fn=None
            ) -> Float[Tensor, "nsamples *shape"]:  # noqa: F821
        # TODO: Add the option of custom integration
        if y is not None:
            y = y.unsqueeze(0)  # Broadcasting will take care of the rest

        def rhs(x, sigma):
            with torch.inference_mode():
                trained_score = self.get_score(x, sigma, y)
            if interp_fn is not None:        # for interpolating between trained and analytical scores
                assert analytical_score is not None
                alpha = interp_fn(sigma).unsqueeze(-1).to(trained_score.device)
                x_ = x.cpu().detach()
                sigma_ = sigma.cpu().detach()
                analytic_score = analytical_score(x_, sigma_).to(trained_score.device)
                score = alpha * trained_score + (1 - alpha) * analytic_score
                return score
            else:
                return trained_score
        if final_step is None:
            final_step = nsteps

        if integrator is not None:
            self.config.noisescheduler.set_temporary_integrator(integrator)
        result = self.config.noisescheduler.propagate_partial(
                                            x,
                                            rhs,
                                            nsteps,
                                            initial_step,
                                            final_step,
                                            record_history=record_history)
        if integrator is not None:
            self.config.noisescheduler.unset_temporary_integrator()
        return result

    def inpaint(
            self,
            x_orig: Float[Tensor, "nsamples *shape"],  # noqa: F821
            mask: Float[Tensor, "*shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            nsteps: int = 100,
            record_history: bool = False,
            maximum_batch_size: None | int = None,
            mode: str = "inpaint"
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        # TODO: Implement stochastic integration for inpainting
        # (Should be easy, but still, not done yet)
        if maximum_batch_size is not None:
            batch_sizes = get_minibatch_sizes(x_orig.shape[0],
                                              maximum_batch_size)
            result = []
            x_orig_chunks = x_orig.chunk(len(batch_sizes))
            mask_chunks = mask.chunk(len(batch_sizes))
            for i, _ in enumerate(batch_sizes):
                result.append(
                    self.inpaint(x_orig_chunks[i],
                                 mask_chunks[i],
                                 y,
                                 nsteps,
                                 record_history,
                                 maximum_batch_size=None)
                            )
            catdim = 1 if record_history else 0
            result = torch.cat(result, dim=catdim)
            return result
        else:
            x_orig_history = self.propagate_toward_noise(
                                        x_orig,
                                        nsteps=nsteps,
                                        y=y,
                                        record_history=True,
                                        stochastic_integration=True
                                        )
            noise = (torch.randn_like(x_orig) *
                     self.config.noisescheduler.maximum_scale)
            inpaint_fn = (self.propagate_inpaint_toward_sample
                          if mode == "inpaint"
                          else self.propagate_repaint_toward_sample)
            x_inpainted = inpaint_fn(
                noise,
                x_orig_history,
                mask,
                y=y,
                record_history=record_history)
            return x_inpainted

    def repaint(
            self,
            x_orig: Float[Tensor, "nsamples *shape"],  # noqa: F821
            mask: Float[Tensor, "*shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            nsteps: int = 100,
            record_history: bool = False,
            maximum_batch_size: None | int = None
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        return self.inpaint(x_orig,
                            mask,
                            y,
                            nsteps,
                            record_history,
                            maximum_batch_size,
                            mode="repaint")

    def propagate_inpaint_toward_sample(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            x_inpaint: Float[Tensor, "nsteps+1 nsamples *shape"],  # noqa: F722
            mask: Float[Tensor, "*shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            record_history: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        if y is not None:
            y = dict_unsqueeze(y, 0)  # Broadcasting will take care of the rest

        def rhs(x, sigma):
            with torch.inference_mode():
                return self.get_score(x, sigma, y)
        nsteps = x_inpaint.shape[0] - 1
        result = self.config.noisescheduler.inpaint(
                                            x,
                                            x_inpaint,
                                            mask,
                                            rhs,
                                            nsteps,
                                            record_history=record_history)
        return result

    def propagate_repaint_toward_sample(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            x_inpaint: Float[Tensor, "nsteps+1 nsamples *shape"],  # noqa: F722
            mask: Float[Tensor, "*shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            record_history: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        if y is not None:
            y = dict_unsqueeze(y, 0)  # Broadcasting will take care of the rest

        def rhs(x, sigma):
            with torch.inference_mode():
                return self.get_score(x, sigma, y)
        nsteps = x_inpaint.shape[0] - 1
        result = self.config.noisescheduler.repaint(
                                            x,
                                            x_inpaint,
                                            mask,
                                            rhs,
                                            nsteps,
                                            record_history=record_history)
        return result

    def propagate_toward_noise(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            nsteps: int = 100,
            record_history: bool = False,
            stochastic_integration: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        if y is not None:
            y = dict_unsqueeze(y, 0)  # Broadcasting will take care of the rest

        def rhs(x, sigma):
            with torch.inference_mode():
                return self.get_score(x, sigma, y)

        result = self.config.noisescheduler.propagate_forward(
                                            x,
                                            rhs,
                                            nsteps,
                                            record_history=record_history,
                                            stochastic=stochastic_integration)
        return result

    def interpolate_images(
            self,
            x1: Float[Tensor, "*shape"],  # noqa: F821
            x2: Float[Tensor, "*shape"],  # noqa: F821
            ninterp: int,
            jitter: None | float = 1e-2,
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            nsteps: int = 100,
            record_history: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        x = torch.stack([x1, x2], dim=0)  # [2, *shape]
        if jitter is not None:
            x = x + jitter*torch.randn_like(x)
        if y is not None:
            y = dict_unsqueeze(y, 0)
        x_noised = self.propagate_toward_noise(x,
                                               y,
                                               nsteps)
        x1_noised, x2_noised = x_noised[0], x_noised[1]
        x_noised_interp = linear_interpolation(x1_noised,
                                               x2_noised,
                                               ninterp)
        x_interp = self.propagate_toward_sample(
                            x_noised_interp,
                            y=y,
                            nsteps=nsteps,
                            record_history=record_history)
        return x_interp

    def training_step(self, batch, batch_idx):
        x, y, mask = self.select_batch(batch)
        sigma = self.config.noisesampler.sample(x.shape[0]).to(x)  # [nbatch]
        loss = self.loss_fn(x, sigma, y, mask)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = self.select_batch(batch)
        sigma = self.config.noisesampler.sample(x.shape[0]).to(x)  # [nbatch]
        loss = self.loss_fn(x, sigma, y, mask)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)  # For compat
        return loss

    def configure_optimizers(self):
        if self.lr_scheduler is not None:
            lr_scheduler_config = {"scheduler": self.lr_scheduler,
                                   "interval": self.lr_scheduler_interval}
            return [self.optimizer], [lr_scheduler_config]
        else:  # Just fo backward compatibility for some examples
            return self.optimizer

    def select_batch(self, batch):
        if (self.conditional and self.masked):
            x, y, mask = batch
        elif ((not self.conditional) and self.masked):
            x, mask = batch
            y = None
        elif (self.conditional and (not self.masked)):
            x, y = batch
            mask = None
        else:
            x = batch
            y = None
            mask = None
        return x, y, mask

    def encode(self, x, y=None, record_history=False):
        if record_history:
            xlist = []
            for xx in x:
                out = self.encode(xx, y, record_history=False)
                xlist.append(out)
            return torch.stack(xlist, dim=0)
        if self.latent_model:
            if self.autoencoder_conditional:
                if self.encode_y:
                    x, y = self.autoencoder.encode(x, y)
                else:
                    x = self.autoencoder.encode(x, y)
            else:
                x = self.autoencoder.encode(x)
        else:
            x = x
        if self.config.has_edm_batch_norm:
            x = self.edm_batch_norm.normalize(x)
        if self.encode_y:
            return x/self.norm, y
        else:
            return x / self.norm

    def decode(self, x, y=None, record_history=False):
        if record_history:
            xlist = []
            for xx in x:
                out = self.decode(xx, y, record_history=False)
                xlist.append(out)
            return torch.stack(xlist, dim=0)
        else:
            x = x * self.norm
            if self.config.has_edm_batch_norm:
                x = self.edm_batch_norm.unnormalize(x)
            if self.latent_model:
                if self.autoencoder_conditional:
                    x = self.autoencoder.decode(x, y)
                else:
                    x = self.autoencoder.decode(x)
            else:
                x = x
            return x

    def start_edm_batch_norm(self):
        if not self.config.has_edm_batch_norm:
            self.edm_batch_norm = None
        else:
            sigma_data = self.config.extra_args.get("sigma_data", 0.5)
            self.edm_batch_norm = edmbatchnorm.DimensionAgnosticBatchNorm(sigma=sigma_data)

    def start_dynamic_loss_weight(self):
        if self.config.has_dynamic_loss_weight:
            self.dynamic_loss_weight = DynamicLossWeight(
                self.config.dynamic_loss_weight
            )
        else:
            self.dynamic_loss_weight = None

    @property
    def latent_model(self):
        return self.autoencoder is not None


class DynamicLossWeight(torch.nn.Module):
    def __init__(self, nhidden: int, scale: float = 1.0):
        super().__init__()
        self.nhidden = nhidden
        self.register_buffer(
            "fourier_weights",
            torch.randn(nhidden)*scale
        )  # [nhidden]
        self.register_buffer(
            "fourier_bias",
            torch.rand(nhidden)*scale
        )  # [nhidden]
        self.linear = torch.nn.Linear(nhidden, 1)

    def forward(self, x):
        # x : [batch]
        # returns : [batch]
        x = x.unsqueeze(1)  # [batch, 1]
        h = x * self.fourier_weights + self.fourier_bias  # [batch, nhidden]
        h = torch.cos(h)  # [batch, nhidden]
        h = self.linear(h)  # [batch, 1]
        h = h.squeeze(1)  # [batch]
        return h

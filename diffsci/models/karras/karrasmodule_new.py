import math
from contextlib import contextmanager
from fnmatch import fnmatchcase
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
from diffsci.custom_losses import EnsembleAwareMSELoss, EnsembleAwareHuberLoss, EnsembleAwareGaussianWeightedMSELoss, EnsembleAwareSmoothedIndicatorLoss, MultiSpaceLoss, EnsembleAwareCRPSLoss
from . import preconditioners
from . import noisesamplers
from . import schedulers
from . import edmbatchnorm
from . import integrators
from .autoregressivesample import LatentSpaceAutoregressive
from .autoregressiveloss import AutoregressiveLossMixin
from .ema import ModelEMA

Scaler = Callable[[Float[Tensor, '*shape']],  # noqa: F821
                  Float[Tensor, '*shape']]  # noqa: F821
Sampler = Callable[[list[int]], Float[Tensor, '...']]


class EnsembleKarrasModuleConfig(object):
    def __init__(self,
                 preconditioner: preconditioners.KarrasPreconditioner,
                 noisesampler: noisesamplers.NoiseSampler,
                 noisescheduler: schedulers.Scheduler,
                 loss_metric: Union[str, Dict[str, Any]] = "huber",
                 tag: str = "custom",
                 has_edm_batch_norm: bool = False,
                 dynamic_loss_weight: int | None = None,
                 extra_args: None | dict[str, Any] = None,
                 ensemble_size_train: int = 1,
                 ensemble_size_val: int = 1,
                 ensemble_size_test: int = 1,
                 autoregressive_loss_steps: int = 1,
                 autoregressive_loss_diffusion_steps: int = 100,
                 autoregressive_loss_guidance: float = 1.0,
                 autoregressive_loss_weights: None | list[float] = None,
                 autoregressive_loss_maximum_batch_size: None | int = None,
                 autoregressive_loss_integrator: None | str | integrators.Integrator = None,
                 ema_enabled: bool = False,
                 ema_type: str = "traditional",
                 ema_decay: float = 0.999,
                 ema_halflife_steps: None | float = None,
                 ema_rampup_ratio: None | float = None,
                 ema_power_function_stds: None | list[float] = None,
                 ema_use_for_validation: bool = True,
                 ema_use_for_sampling: bool = True,
                 ema_device: None | str = None,
                 ema_profile_index: int = 0,
                 freeze_layer_patterns: None | str | list[str] = None,
                 freeze_layer_strict: bool = True,
                 replay_enabled: bool = False,
                 replay_loss_weight: float = 0.1,
                 replay_loss_schedule: None | dict[str, Any] = None,
                 replay_validation_enabled: bool = False,
                 pretrained_weight_regularization: None | dict[str, Any] = None,
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
            ensemble_size_train: Ensemble size for training
            ensemble_size_val: Ensemble size for validation
            ensemble_size_test: Ensemble size for testing
            autoregressive_loss_steps: Number of successive forecast targets
                to train on. Values > 1 enable autoregressive loss.
            autoregressive_loss_diffusion_steps: Diffusion steps used when
                generating intermediate autoregressive conditioning samples.
            autoregressive_loss_guidance: Guidance used for intermediate
                autoregressive samples.
            autoregressive_loss_weights: Optional per-step loss weights.
            autoregressive_loss_maximum_batch_size: Optional sampling minibatch
                size for intermediate autoregressive samples.
            autoregressive_loss_integrator: Optional sampling integrator for
                intermediate autoregressive samples.
            ema_enabled: Enables EMA shadow weights during training.
            ema_type: "traditional" or "power". "power" follows the EDM2
                power-function EMA profile.
            ema_decay: Traditional EMA decay when ema_halflife_steps is None.
            ema_halflife_steps: Optional traditional EMA half-life in optimizer
                updates.
            ema_rampup_ratio: Optional traditional EMA half-life ramp-up ratio.
            ema_power_function_stds: Relative std profile(s) for power EMA.
            ema_use_for_validation: Temporarily validate with EMA weights.
            ema_use_for_sampling: Use EMA weights for eval-time sampling.
            ema_device: Optional device for EMA tensors, e.g. "cpu".
            ema_profile_index: Which tracked EMA profile to apply.
            freeze_layer_patterns: Optional glob-style patterns matching
                `model.named_modules()` or `model.named_parameters()` entries
                to freeze before the optimizer is created. A module match
                freezes all parameters below that module.
            freeze_layer_strict: Raise an error when any freeze pattern has no
                matches.
            replay_enabled: Enables loss replay from a secondary dataloader.
            replay_loss_weight: Weight applied to the replay loss.
            replay_loss_schedule: Optional schedule overriding replay_loss_weight
                during training.
            replay_validation_enabled: Enables separate replay validation logging
                when the trainer provides a replay validation dataloader.
            pretrained_weight_regularization: Optional L2-SP regularization
                config against the initial pretrained weights.
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
        self.ensemble_size_train = ensemble_size_train
        self.ensemble_size_val = ensemble_size_val
        self.ensemble_size_test = ensemble_size_test
        self.autoregressive_loss_steps = autoregressive_loss_steps
        self.autoregressive_loss_diffusion_steps = autoregressive_loss_diffusion_steps
        self.autoregressive_loss_guidance = autoregressive_loss_guidance
        self.autoregressive_loss_weights = autoregressive_loss_weights
        self.autoregressive_loss_maximum_batch_size = autoregressive_loss_maximum_batch_size
        self.autoregressive_loss_integrator = autoregressive_loss_integrator
        self.ema_enabled = ema_enabled
        self.ema_type = ema_type
        self.ema_decay = ema_decay
        self.ema_halflife_steps = ema_halflife_steps
        self.ema_rampup_ratio = ema_rampup_ratio
        self.ema_power_function_stds = ema_power_function_stds
        self.ema_use_for_validation = ema_use_for_validation
        self.ema_use_for_sampling = ema_use_for_sampling
        self.ema_device = ema_device
        self.ema_profile_index = ema_profile_index
        self.freeze_layer_patterns = freeze_layer_patterns
        self.freeze_layer_strict = freeze_layer_strict
        self.replay_enabled = replay_enabled
        self.replay_loss_weight = float(replay_loss_weight)
        self.replay_loss_schedule = replay_loss_schedule
        self.replay_validation_enabled = replay_validation_enabled
        self.pretrained_weight_regularization = pretrained_weight_regularization
        if extra_args is None:
            self.extra_args = dict()
        else:
            self.extra_args = extra_args

    @staticmethod
    def ema_config_keys() -> set[str]:
        return {
            "ema_enabled",
            "ema_type",
            "ema_decay",
            "ema_halflife_steps",
            "ema_rampup_ratio",
            "ema_power_function_stds",
            "ema_use_for_validation",
            "ema_use_for_sampling",
            "ema_device",
            "ema_profile_index",
        }

    @staticmethod
    def freeze_config_keys() -> set[str]:
        return {
            "freeze_layer_patterns",
            "freeze_layer_strict",
        }

    @staticmethod
    def replay_config_keys() -> set[str]:
        return {
            "replay_enabled",
            "replay_loss_weight",
            "replay_loss_schedule",
            "replay_validation_enabled",
        }

    @staticmethod
    def pretrained_regularization_config_keys() -> set[str]:
        return {
            "pretrained_weight_regularization",
        }

    @classmethod
    def _normalize_ema_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        unknown = set(kwargs) - cls.ema_config_keys()
        if unknown:
            raise TypeError(
                f"Unexpected EMA config key(s): {', '.join(sorted(unknown))}"
            )
        return kwargs

    def _current_ema_extra_args(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.ema_config_keys()
        }

    def _current_freeze_extra_args(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.freeze_config_keys()
        }

    def _current_replay_extra_args(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.replay_config_keys()
        }

    def _current_pretrained_regularization_extra_args(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.pretrained_regularization_config_keys()
        }

    @classmethod
    def from_edm(cls,
                 sigma_data: float = 0.5,
                 prior_mean: float = -1.2,
                 prior_std: float = 1.2,
                 has_edm_batch_norm: bool = False,
                 dynamic_loss_weight: int | None = None,
                 loss_metric: Union[str, Dict[str, Any]] = "huber",
                 autoregressive_loss_steps: int = 1,
                 autoregressive_loss_diffusion_steps: int = 100,
                 autoregressive_loss_guidance: float = 1.0,
                 autoregressive_loss_weights: None | list[float] = None,
                 autoregressive_loss_maximum_batch_size: None | int = None,
                 autoregressive_loss_integrator: None | str | integrators.Integrator = None,
                 freeze_layer_patterns: None | str | list[str] = None,
                 freeze_layer_strict: bool = True,
                 replay_enabled: bool = False,
                 replay_loss_weight: float = 0.1,
                 replay_loss_schedule: None | dict[str, Any] = None,
                 replay_validation_enabled: bool = False,
                 pretrained_weight_regularization: None | dict[str, Any] = None,
                 spatial_shape: tuple = None,
                 focus_radius: float = None,
                 **ema_kwargs):
        """
        Create EDM configuration with flexible loss support.
        
        Args:
            sigma_data: Sigma data parameter
            prior_mean: Prior mean
            prior_std: Prior standard deviation
            has_edm_batch_norm: Whether to use EDM batch norm
            dynamic_loss_weight: Dynamic loss weight
            loss_metric: Loss configuration (new flexible format)
            autoregressive_loss_steps: Number of successive forecast targets
                to train on. Values > 1 enable autoregressive loss.
            autoregressive_loss_diffusion_steps: Diffusion steps used when
                generating intermediate autoregressive conditioning samples.
            autoregressive_loss_guidance: Guidance used for intermediate
                autoregressive samples.
            autoregressive_loss_weights: Optional per-step loss weights.
            autoregressive_loss_maximum_batch_size: Optional sampling minibatch
                size for intermediate autoregressive samples.
            autoregressive_loss_integrator: Optional sampling integrator for
                intermediate autoregressive samples.
            freeze_layer_patterns: Optional glob-style layer/parameter
                patterns to freeze before optimizer creation.
            freeze_layer_strict: Raise an error when any freeze pattern has no
                matches.
            replay_enabled: Enables loss replay from a secondary dataloader.
            replay_loss_weight: Weight applied to the replay loss.
            replay_loss_schedule: Optional schedule overriding replay_loss_weight.
            replay_validation_enabled: Enables separate replay validation logging.
            pretrained_weight_regularization: Optional L2-SP config.
            spatial_shape: For weighted_gaussian loss (legacy)
            focus_radius: For weighted_gaussian loss (legacy)
        """
        ema_kwargs = cls._normalize_ema_kwargs(ema_kwargs)
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
            "autoregressive_loss_steps": autoregressive_loss_steps,
            "autoregressive_loss_diffusion_steps": autoregressive_loss_diffusion_steps,
            "autoregressive_loss_guidance": autoregressive_loss_guidance,
            "autoregressive_loss_weights": autoregressive_loss_weights,
            "autoregressive_loss_maximum_batch_size": autoregressive_loss_maximum_batch_size,
            "autoregressive_loss_integrator": autoregressive_loss_integrator,
            "freeze_layer_patterns": freeze_layer_patterns,
            "freeze_layer_strict": freeze_layer_strict,
            "replay_enabled": replay_enabled,
            "replay_loss_weight": replay_loss_weight,
            "replay_loss_schedule": replay_loss_schedule,
            "replay_validation_enabled": replay_validation_enabled,
            "pretrained_weight_regularization": pretrained_weight_regularization,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius,
            **ema_kwargs
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   tag=tag,
                   has_edm_batch_norm=has_edm_batch_norm,
                   dynamic_loss_weight=dynamic_loss_weight,
                   autoregressive_loss_steps=autoregressive_loss_steps,
                   autoregressive_loss_diffusion_steps=autoregressive_loss_diffusion_steps,
                   autoregressive_loss_guidance=autoregressive_loss_guidance,
                   autoregressive_loss_weights=autoregressive_loss_weights,
                   autoregressive_loss_maximum_batch_size=autoregressive_loss_maximum_batch_size,
                   autoregressive_loss_integrator=autoregressive_loss_integrator,
                   freeze_layer_patterns=freeze_layer_patterns,
                   freeze_layer_strict=freeze_layer_strict,
                   replay_enabled=replay_enabled,
                   replay_loss_weight=replay_loss_weight,
                   replay_loss_schedule=replay_loss_schedule,
                   replay_validation_enabled=replay_validation_enabled,
                   pretrained_weight_regularization=pretrained_weight_regularization,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius,
                   **ema_kwargs)

    @classmethod
    def from_vp(cls,
                beta_data: float = 19.9,
                beta_min: float = 0.1,
                epsilon_min: float = 1e-3,
                epsilon_sampler: float = 1e-5,
                M: int = 1000,
                loss_metric: Union[str, Dict[str, Any]] = "huber",
                autoregressive_loss_steps: int = 1,
                autoregressive_loss_diffusion_steps: int = 100,
                autoregressive_loss_guidance: float = 1.0,
                autoregressive_loss_weights: None | list[float] = None,
                autoregressive_loss_maximum_batch_size: None | int = None,
                autoregressive_loss_integrator: None | str | integrators.Integrator = None,
                freeze_layer_patterns: None | str | list[str] = None,
                freeze_layer_strict: bool = True,
                replay_enabled: bool = False,
                replay_loss_weight: float = 0.1,
                replay_loss_schedule: None | dict[str, Any] = None,
                replay_validation_enabled: bool = False,
                pretrained_weight_regularization: None | dict[str, Any] = None,
                spatial_shape: tuple = None,
                focus_radius: float = None,
                **ema_kwargs):
        """
        Create VP configuration with flexible loss support.
        """
        ema_kwargs = cls._normalize_ema_kwargs(ema_kwargs)
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
            "autoregressive_loss_steps": autoregressive_loss_steps,
            "autoregressive_loss_diffusion_steps": autoregressive_loss_diffusion_steps,
            "autoregressive_loss_guidance": autoregressive_loss_guidance,
            "autoregressive_loss_weights": autoregressive_loss_weights,
            "autoregressive_loss_maximum_batch_size": autoregressive_loss_maximum_batch_size,
            "autoregressive_loss_integrator": autoregressive_loss_integrator,
            "freeze_layer_patterns": freeze_layer_patterns,
            "freeze_layer_strict": freeze_layer_strict,
            "replay_enabled": replay_enabled,
            "replay_loss_weight": replay_loss_weight,
            "replay_loss_schedule": replay_loss_schedule,
            "replay_validation_enabled": replay_validation_enabled,
            "pretrained_weight_regularization": pretrained_weight_regularization,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius,
            **ema_kwargs
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   autoregressive_loss_steps=autoregressive_loss_steps,
                   autoregressive_loss_diffusion_steps=autoregressive_loss_diffusion_steps,
                   autoregressive_loss_guidance=autoregressive_loss_guidance,
                   autoregressive_loss_weights=autoregressive_loss_weights,
                   autoregressive_loss_maximum_batch_size=autoregressive_loss_maximum_batch_size,
                   autoregressive_loss_integrator=autoregressive_loss_integrator,
                   freeze_layer_patterns=freeze_layer_patterns,
                   freeze_layer_strict=freeze_layer_strict,
                   replay_enabled=replay_enabled,
                   replay_loss_weight=replay_loss_weight,
                   replay_loss_schedule=replay_loss_schedule,
                   replay_validation_enabled=replay_validation_enabled,
                   pretrained_weight_regularization=pretrained_weight_regularization,
                   tag=tag,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius,
                   **ema_kwargs)

    @classmethod
    def from_ve(cls,
                sigma_min: float = 0.02,
                sigma_max: float = 100,
                loss_metric: Union[str, Dict[str, Any]] = "huber",
                autoregressive_loss_steps: int = 1,
                autoregressive_loss_diffusion_steps: int = 100,
                autoregressive_loss_guidance: float = 1.0,
                autoregressive_loss_weights: None | list[float] = None,
                autoregressive_loss_maximum_batch_size: None | int = None,
                autoregressive_loss_integrator: None | str | integrators.Integrator = None,
                freeze_layer_patterns: None | str | list[str] = None,
                freeze_layer_strict: bool = True,
                replay_enabled: bool = False,
                replay_loss_weight: float = 0.1,
                replay_loss_schedule: None | dict[str, Any] = None,
                replay_validation_enabled: bool = False,
                pretrained_weight_regularization: None | dict[str, Any] = None,
                spatial_shape: tuple = None,
                focus_radius: float = None,
                **ema_kwargs):
        """
        Create VE configuration with flexible loss support.
        """
        ema_kwargs = cls._normalize_ema_kwargs(ema_kwargs)
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
            "autoregressive_loss_steps": autoregressive_loss_steps,
            "autoregressive_loss_diffusion_steps": autoregressive_loss_diffusion_steps,
            "autoregressive_loss_guidance": autoregressive_loss_guidance,
            "autoregressive_loss_weights": autoregressive_loss_weights,
            "autoregressive_loss_maximum_batch_size": autoregressive_loss_maximum_batch_size,
            "autoregressive_loss_integrator": autoregressive_loss_integrator,
            "freeze_layer_patterns": freeze_layer_patterns,
            "freeze_layer_strict": freeze_layer_strict,
            "replay_enabled": replay_enabled,
            "replay_loss_weight": replay_loss_weight,
            "replay_loss_schedule": replay_loss_schedule,
            "replay_validation_enabled": replay_validation_enabled,
            "pretrained_weight_regularization": pretrained_weight_regularization,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius,
            **ema_kwargs
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   autoregressive_loss_steps=autoregressive_loss_steps,
                   autoregressive_loss_diffusion_steps=autoregressive_loss_diffusion_steps,
                   autoregressive_loss_guidance=autoregressive_loss_guidance,
                   autoregressive_loss_weights=autoregressive_loss_weights,
                   autoregressive_loss_maximum_batch_size=autoregressive_loss_maximum_batch_size,
                   autoregressive_loss_integrator=autoregressive_loss_integrator,
                   freeze_layer_patterns=freeze_layer_patterns,
                   freeze_layer_strict=freeze_layer_strict,
                   replay_enabled=replay_enabled,
                   replay_loss_weight=replay_loss_weight,
                   replay_loss_schedule=replay_loss_schedule,
                   replay_validation_enabled=replay_validation_enabled,
                   pretrained_weight_regularization=pretrained_weight_regularization,
                   tag=tag,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius,
                   **ema_kwargs)

    @classmethod
    def conditionalSR3(cls,
                       sigma_min: float = 0.02,
                       sigma_max: float = 100,
                       loss_metric: Union[str, Dict[str, Any]] = "huber",
                       autoregressive_loss_steps: int = 1,
                       autoregressive_loss_diffusion_steps: int = 100,
                       autoregressive_loss_guidance: float = 1.0,
                       autoregressive_loss_weights: None | list[float] = None,
                       autoregressive_loss_maximum_batch_size: None | int = None,
                       autoregressive_loss_integrator: None | str | integrators.Integrator = None,
                       freeze_layer_patterns: None | str | list[str] = None,
                       freeze_layer_strict: bool = True,
                       replay_enabled: bool = False,
                       replay_loss_weight: float = 0.1,
                       replay_loss_schedule: None | dict[str, Any] = None,
                       replay_validation_enabled: bool = False,
                       pretrained_weight_regularization: None | dict[str, Any] = None,
                       spatial_shape: tuple = None,
                       focus_radius: float = None,
                       **ema_kwargs):
        """
        Create conditional SR3 configuration with flexible loss support.
        """
        ema_kwargs = cls._normalize_ema_kwargs(ema_kwargs)
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
            "autoregressive_loss_steps": autoregressive_loss_steps,
            "autoregressive_loss_diffusion_steps": autoregressive_loss_diffusion_steps,
            "autoregressive_loss_guidance": autoregressive_loss_guidance,
            "autoregressive_loss_weights": autoregressive_loss_weights,
            "autoregressive_loss_maximum_batch_size": autoregressive_loss_maximum_batch_size,
            "autoregressive_loss_integrator": autoregressive_loss_integrator,
            "freeze_layer_patterns": freeze_layer_patterns,
            "freeze_layer_strict": freeze_layer_strict,
            "replay_enabled": replay_enabled,
            "replay_loss_weight": replay_loss_weight,
            "replay_loss_schedule": replay_loss_schedule,
            "replay_validation_enabled": replay_validation_enabled,
            "pretrained_weight_regularization": pretrained_weight_regularization,
            "spatial_shape": spatial_shape,
            "focus_radius": focus_radius,
            **ema_kwargs
        }
        return cls(preconditioner=preconditioner,
                   noisesampler=noisesampler,
                   noisescheduler=noisescheduler,
                   loss_metric=loss_metric,
                   autoregressive_loss_steps=autoregressive_loss_steps,
                   autoregressive_loss_diffusion_steps=autoregressive_loss_diffusion_steps,
                   autoregressive_loss_guidance=autoregressive_loss_guidance,
                   autoregressive_loss_weights=autoregressive_loss_weights,
                   autoregressive_loss_maximum_batch_size=autoregressive_loss_maximum_batch_size,
                   autoregressive_loss_integrator=autoregressive_loss_integrator,
                   freeze_layer_patterns=freeze_layer_patterns,
                   freeze_layer_strict=freeze_layer_strict,
                   replay_enabled=replay_enabled,
                   replay_loss_weight=replay_loss_weight,
                   replay_loss_schedule=replay_loss_schedule,
                   replay_validation_enabled=replay_validation_enabled,
                   pretrained_weight_regularization=pretrained_weight_regularization,
                   tag=tag,
                   extra_args=extra_args,
                   spatial_shape=spatial_shape,
                   focus_radius=focus_radius,
                   **ema_kwargs)

    def export_description(self) -> dict[str, Any]:
        """Export configuration for saving/loading."""
        extra_args = dict(self.extra_args)
        extra_args.update(self._current_ema_extra_args())
        extra_args.update(self._current_freeze_extra_args())
        extra_args.update(self._current_replay_extra_args())
        extra_args.update(self._current_pretrained_regularization_extra_args())
        return dict(tag=self.tag,
                    extra_args=extra_args)

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



class EnsembleKarrasModule(AutoregressiveLossMixin,
                           LatentSpaceAutoregressive,
                           lightning.LightningModule):
    """Updated KarrasModule with multi-space loss support"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 config: 'EnsembleKarrasModuleConfig',
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
        self.apply_freeze_layer_patterns()
        self.set_optimizer_and_scheduler()
        self.set_loss_metric()
        self.start_edm_batch_norm()
        self.start_dynamic_loss_weight()
        self.start_ema()
        self.norm = 1.0
        self._ema_scope_depth = 0
        self._ema_validation_backup = None
        self._ema_loaded_from_checkpoint = False
        self._pretrained_regularization_reference: dict[str, torch.Tensor] | None = None
        self._pretrained_regularization_parameter_names: list[str] = []

    def freeze_autoencoder(self):
        """Freezes the autoencoder to prevent its weights from being updated during training."""
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    @staticmethod
    def _normalize_freeze_layer_patterns(
            patterns: None | str | list[str]) -> list[str]:
        if patterns is None:
            return []
        if isinstance(patterns, str):
            return [patterns]
        return list(patterns)

    @staticmethod
    def _canonical_freeze_pattern(pattern: str) -> str:
        pattern = str(pattern).strip()
        if pattern.startswith("model."):
            pattern = pattern[len("model."):]
        return pattern

    @staticmethod
    def _freeze_pattern_matches(pattern: str, name: str) -> bool:
        return (
            name == pattern or
            name.startswith(f"{pattern}.") or
            fnmatchcase(name, pattern)
        )

    def apply_freeze_layer_patterns(self) -> None:
        """Freeze selected parameters from the denoising model by name pattern."""
        patterns = self._normalize_freeze_layer_patterns(
            getattr(self.config, "freeze_layer_patterns", None)
        )
        self.frozen_layer_matches: dict[str, list[str]] = {}
        self.frozen_parameter_names: list[str] = []
        self.frozen_parameter_count = 0

        if len(patterns) == 0:
            return

        named_modules = dict(self.model.named_modules())
        named_parameters = dict(self.model.named_parameters())
        matched_parameter_names: set[str] = set()

        for raw_pattern in patterns:
            pattern = self._canonical_freeze_pattern(raw_pattern)
            pattern_matches: set[str] = set()

            for module_name, module in named_modules.items():
                if module_name and self._freeze_pattern_matches(pattern, module_name):
                    pattern_matches.add(module_name)
                    for parameter_name, _ in module.named_parameters(recurse=True):
                        full_name = f"{module_name}.{parameter_name}"
                        matched_parameter_names.add(full_name)

            for parameter_name in named_parameters:
                if self._freeze_pattern_matches(pattern, parameter_name):
                    pattern_matches.add(parameter_name)
                    matched_parameter_names.add(parameter_name)

            self.frozen_layer_matches[raw_pattern] = sorted(pattern_matches)

        unmatched_patterns = [
            pattern for pattern, matches in self.frozen_layer_matches.items()
            if len(matches) == 0
        ]
        if unmatched_patterns and getattr(self.config, "freeze_layer_strict", True):
            raise ValueError(
                "The following freeze_layer_patterns did not match any model "
                f"module or parameter: {unmatched_patterns}"
            )

        for parameter_name, parameter in named_parameters.items():
            if parameter_name in matched_parameter_names:
                parameter.requires_grad = False
                self.frozen_parameter_names.append(parameter_name)
                self.frozen_parameter_count += parameter.numel()

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

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
            trainable_parameters = self.trainable_parameters()
            if len(trainable_parameters) == 0:
                raise ValueError("No trainable parameters remain after freezing.")
            self.optimizer = torch.optim.AdamW(trainable_parameters,
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
        
        Now supports ensemble inputs [B, E, C, H, W]!
        
        Supports three formats:
        1. String: "mse" (backward compatible)
        2. Dict with single loss: {"smoothed_indicator": {...}} (single loss)
        3. Dict with multiple losses: {"losses": [...]} (multi-space losses)
        """
        loss_config = self.config.loss_metric

        if self.config.ensemble_size_train == 1 and self.config.ensemble_size_train ==1 and self.config.ensemble_size_test == 1:
            if isinstance(loss_config, str):
                 # Original string format - create single loss
                 self._set_single_loss_string(loss_config)
            
            elif isinstance(loss_config, dict):
                if "losses" in loss_config:
                    self.multi_space_loss = MultiSpaceLoss(loss_config, self.autoencoder)
                    self.loss_metric = None
                else:
                    # Single loss dict format
                    self._set_single_loss_dict(loss_config)
            else:
                raise ValueError(f"loss_metric must be string or dict, got {type(loss_config)}")

        else:
 
            if isinstance(loss_config, str):
                # Original string format - create single loss
                self._set_enseble_single_loss_string(loss_config)
                
            elif isinstance(loss_config, dict):
                if "losses" in loss_config:
                    # Multi-space loss configuration (skip for now)
                    self.multi_space_loss = MultiSpaceLoss(loss_config, self.autoencoder)
                    self.loss_metric = None
                else:
                    # Single loss dict format
                    self._set_enseble_single_loss_dict(loss_config)
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
        elif loss_name == "CRPS":
            self.loss_metric = EnsembleAwareCRPSLoss()
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

    def _set_enseble_single_loss_string(self, loss_name: str):
        """Handle single loss string format (backward compatible + ensemble-aware)"""
        if loss_name == "mse":
            self.loss_metric = EnsembleAwareMSELoss(reduction="none")
        elif loss_name == "huber":
            self.loss_metric = EnsembleAwareHuberLoss(reduction="none")
        elif loss_name == "weighted_gaussian":
            if self.config.spatial_shape is None or self.config.focus_radius is None:
                raise AttributeError("config must have shape tuple and focus radius")
            self.loss_metric = EnsembleAwareGaussianWeightedMSELoss(
                shape=self.config.spatial_shape,
                focus_radius=self.config.focus_radius
            )
        elif loss_name == "smoothed_indicator":
            # Wrap your original smoothed indicator with ensemble support
            original_loss = MultiThresholdSmoothIndicatorLoss()
            self.loss_metric = EnsembleAwareSmoothedIndicatorLoss(original_loss)
        elif loss_name == "CRPS":
            self.loss_metric = EnsembleAwareCRPSLoss()
        else:
            raise ValueError(f"loss_type {loss_name} not recognized")


    def _set_enseble_single_loss_dict(self, loss_config: Dict[str, Any]):
        """Handle single loss dict format (with ensemble support)"""
        loss_name = list(loss_config.keys())[0]
        loss_params = loss_config[loss_name]
        
        if loss_name == "mse":
            self.loss_metric = EnsembleAwareMSELoss(reduction="none")
        elif loss_name == "huber":
            delta = loss_params.get('delta', 1.0)
            self.loss_metric = EnsembleAwareHuberLoss(delta=delta, reduction="none")
        elif loss_name == "weighted_gaussian":
            shape = loss_params.get('shape') or self.config.spatial_shape
            focus_radius = loss_params.get('focus_radius') or self.config.focus_radius
            self.loss_metric = EnsembleAwareGaussianWeightedMSELoss(
                shape=shape,
                focus_radius=focus_radius
            )
        elif loss_name == "smoothed_indicator":
            # Wrap your original smoothed indicator with ensemble support
            original_loss = MultiThresholdSmoothIndicatorLoss(**loss_params)
            self.loss_metric = EnsembleAwareSmoothedIndicatorLoss(original_loss)

        elif loss_name == "CRPS":
            self.loss_metric = EnsembleAwareCRPSLoss(*loss_params)
        
        else:
            raise ValueError(f"loss_name '{loss_name}' not recognized")


    def loss_fn(self,
            x: Float[Tensor, "batch *shape"],
            sigma: Float[Tensor, "batch"],
            y: None | Float[Tensor, "batch *yshape"] = None,
            mask: None | Float[Tensor, "batch *shape"] = None,
            n_ensemble: int = 1) -> Float[Tensor, ""]:
        """
        Loss function with vectorized ensemble generation.
        
        Args:
            x: Input tensor [B, C, H, W]
            sigma: Noise level/timestep [B]
            y: Conditioning
            mask: Optional mask
            n_ensemble: Number of ensemble members
        
        The key optimization here:
            1. Generate all ensemble noises at once: [B, E, C, H, W]
            2. Reshape to [B*E, C, H, W] for batch processing
            3. Call denoiser once with parallelism across B*E
            4. Reshape back to [B, E, C, H, W] for CRPS
        
        Returns:
            Scalar loss
        """
        if n_ensemble <= 1:
            # old loss function for backward compatibility
            return self.old_loss_fn(x, sigma, y, mask)
        # Store original pixel space data   
        device = x.device
        sigma = sigma.to(device)

        if y is not None:
            if isinstance(y, dict):
                y = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in y.items()}
            else:
                y = y.to(device)


        x_pixel = x.clone()
        
        mask_pixel = mask.clone().to(device) if mask is not None else None
        
        # Encode to latent space
        if self.encode_y:
            x_latent, y = self.encode(x, y)
        else:
            x_latent = self.encode(x, y)
        
        # === VECTORIZED ENSEMBLE GENERATION ===
        # This is the KEY optimization
        
        B, C, H, W = x_latent.shape
        
        # Broadcast sigma to spatial dimensions
        broadcasted_sigma = broadcast_from_below(sigma, x_latent)  # [B, 1, 1, 1]
        
        # Generate ALL ensemble noises at once
        # Shape: [B, n_ensemble, C, H, W]
        # Use torch.randn_like with expanded shape
        noise_shape = (B, n_ensemble, C, H, W)
        noise_ensemble = torch.randn(noise_shape, device=device, dtype=x_latent.dtype)
        
        # Scale by sigma: [B, 1, 1, 1] * [B, E, C, H, W] -> [B, E, C, H, W]
        # Need to expand sigma for broadcasting

        
        broadcasted_sigma_expanded = broadcasted_sigma.unsqueeze(1)  # [B, 1, 1, 1, 1]
        noise_ensemble = broadcasted_sigma_expanded * noise_ensemble  # [B, E, C, H, W]
        
        # Add noise to input: [B, 1, C, H, W] + [B, E, C, H, W] -> [B, E, C, H, W]
        x_latent_expanded = x_latent.view(B, 1, C, H, W)  # [B, 1, C, H, W]
        x_noised_ensemble = x_latent_expanded + noise_ensemble  # [B, E, C, H, W]
        
        # === RESHAPE TO BATCH DIMENSION FOR PARALLELISM ===
        # Reshape [B, E, C, H, W] -> [B*E, C, H, W] for parallel processing
        x_noised_flat = x_noised_ensemble.view(B * n_ensemble, C, H, W)
        x_noised_flat = x_noised_flat.to(device)
        # === EXPAND SIGMA FOR BATCH ===
        # We need sigma to also be [B*E] for the denoiser
        # Repeat sigma: [B] -> [B, E] -> [B*E]
        sigma_expanded = sigma.unsqueeze(1).expand(B, n_ensemble).reshape(B * n_ensemble)
        sigma_expanded = sigma_expanded.to(device)
        # === EXPAND Y FOR BATCH (if needed) ===
        # If y is [B, ...], expand to [B*E, ...]
        y_expanded = y
        if y is not None:
            # y shape might be [B, ...] or dict of [B, ...]
            if isinstance(y, dict):
                y_expanded = {}
                for key, val in y.items():
                    if val is not None:
                        # [B, ...] -> [B, E, ...] -> [B*E, ...]
                        if key in ['dates', 'latlon_grid', 'ssh']:
                            # TODO: Handle dates for ensemble generation and for models that uses it
                            continue
                        if key == 'bat':
                            val_expanded = val.expand(
                            B, n_ensemble, *val.shape[-2:])
                            val_expanded = val_expanded.reshape(B * n_ensemble, *val.shape[-2:])
                            y_expanded[key] = val_expanded.unsqueeze(1).to(device)

                        else:
                            val_expanded = val.unsqueeze(1).expand(
                                B, n_ensemble, *val.shape[1:]
                            ).reshape(B * n_ensemble, *val.shape[1:])
                            y_expanded[key] = val_expanded.to(device)
            else:
                y_expanded = y.unsqueeze(1).expand(
                    B, n_ensemble, *y.shape[1:]
                ).reshape(B * n_ensemble, *y.shape[1:]).to(device)
        
        # === SINGLE DENOISER CALL ===
        # All B*E samples processed in parallel!
        denoiser_flat, cond_noise = self.get_denoiser(
            x_noised_flat,      # [B*E, C, H, W]
            sigma_expanded,     # [B*E]
            y_expanded          # [B*E, ...]
        )  # Returns [B*E, C, H, W]
        
        # === RESHAPE BACK TO ENSEMBLE DIMENSION ===
        denoiser_latent = denoiser_flat.view(B, n_ensemble, C, H, W)  # [B, E, C, H, W]
        cond_noise_reshaped = cond_noise.view(B, n_ensemble)  # [B, E]

        # === AGGREGATE ENSEMBLE ===
        # Update so that we pass the whole ensemble to the loss function
        # For simple loss: average across ensemble
        #denoiser_mean = denoiser_latent.mean(dim=1)  # [B, C, H, W]
        
        # For CRPS and advanced metrics: keep [B, E, C, H, W] format
        # (see loss_fn_with_crps below)
        
        # === COMPUTE LOSS ===
        
        # Compute loss weighting
        weight = self.config.noisesampler.loss_weighting(broadcasted_sigma)  # [B, 1, 1, 1]
        bias = torch.zeros_like(weight)
        
        if self.config.has_dynamic_loss_weight:
            modifier = self.dynamic_loss_weight(cond_noise.mean(dim=1))  # Average across ensemble
            modifier = broadcast_from_below(modifier, x_latent)
            weight = weight / torch.exp(modifier)
            bias = bias + modifier
        
        # Multi-space loss
        #TODO: Update so that we pass the whole ensemble to the loss function
        if hasattr(self, 'multi_space_loss') and self.multi_space_loss is not None:
            raise NotImplementedError("Multi-space loss is not implemented for ensemble generation")
            loss_results = self.multi_space_loss.compute_loss(
                denoiser_latent=denoiser_mean,  # [B, C, H, W]
                target_latent=x_latent,         # [B, C, H, W]
                target_pixel=x_pixel,
                mask_latent=mask,
                mask_pixel=mask_pixel
            )
            
            total_loss = loss_results["total"]
            
            if total_loss.dim() == 0:
                final_loss = weight.mean() * total_loss + bias.mean()
            else:
                if mask is not None:
                    mask_expanded = mask.expand_as(total_loss)
                    adjusted_loss = total_loss * (1 - mask_expanded)
                    final_loss = (weight * adjusted_loss + bias).mean()
                else:
                    final_loss = (weight * total_loss + bias).mean()
            
            return final_loss
        
        else:
            # Single loss computation
            # DONE: Updated so that we can pass the whole ensemble to the loss function
            loss = self._compute_single_loss(denoiser_latent, x_latent, mask)
            
            if loss.dim() == 0:
                final_loss = weight.mean() * loss + bias.mean()
            else:
                if mask is not None:
                    mask_expanded = mask.expand_as(loss)
                    adjusted_loss = loss * (1 - mask_expanded)
                    final_loss = (weight * adjusted_loss + bias).mean()
                else:
                    final_loss = (weight * loss + bias).mean()
            
            return final_loss

    def old_loss_fn(self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821
            mask: None | Float[Tensor, "batch *shape"] = None  # noqa: F821
            ) -> Float[Tensor, ""]:  # noqa: F821, F722
        """
        Loss function with support for multi-space losses smart mask handling and ensemble.
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

    def _loss_fn_for_autoregressive_step(
            self,
            x: torch.Tensor,
            sigma: torch.Tensor,
            y: Optional[Any],
            mask: Optional[torch.Tensor],
            n_ensemble: int = 1) -> torch.Tensor:
        return self.loss_fn(x, sigma, y, mask, n_ensemble=n_ensemble)

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
            return_in_latent_space: bool = False,
            use_ema: Optional[bool] = None
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        if use_ema is None:
            use_ema = self._should_use_ema_for_sampling()
        if use_ema:
            with self.ema_scope(enabled=True):
                return self.sample(
                    nsamples=nsamples,
                    shape=shape,
                    y=y,
                    guidance=guidance,
                    nsteps=nsteps,
                    record_history=record_history,
                    maximum_batch_size=maximum_batch_size,
                    integrator=integrator,
                    move_to_cpu=move_to_cpu,
                    is_latent_shape=is_latent_shape,
                    squeeze_memory_efficiency=squeeze_memory_efficiency,
                    return_in_latent_space=return_in_latent_space,
                    use_ema=False,
                )
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
                                              return_in_latent_space=return_in_latent_space,
                                              use_ema=False))
                catdim = 1 if record_history else 0
                result = torch.cat(result, dim=catdim)
                return result
            else:
                batched_shape = [nsamples] + list(shape)
                white_noise = torch.randn(*batched_shape).to(self.device)
                if y is not None:
                    y = dict_to(y, self.device)
                # Ideally we do not enter here and is_latent_shape is True
                if self.latent_model and not is_latent_shape:  # TODO: A stupid n. Should be improved
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
            y : None | Float[Tensor, "*yshape"] = None,  # noqa: F821
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

    @property
    def has_replay_loss(self) -> bool:
        return bool(getattr(self.config, "replay_enabled", False))

    @property
    def has_pretrained_weight_regularization(self) -> bool:
        config = self._pretrained_weight_regularization_config()
        if config is None:
            return False
        return float(config.get("weight", 0.0)) > 0.0

    def _pretrained_weight_regularization_config(self) -> dict[str, Any] | None:
        config = getattr(self.config, "pretrained_weight_regularization", None)
        if config is True:
            config = {"enabled": True}
        if not isinstance(config, dict):
            return None
        if not bool(config.get("enabled", False)):
            return None
        return config

    @staticmethod
    def _pattern_list_matches(
            patterns: None | str | list[str],
            name: str,
            default: bool = False) -> bool:
        normalized_patterns = EnsembleKarrasModule._normalize_freeze_layer_patterns(
            patterns
        )
        if len(normalized_patterns) == 0:
            return default
        for raw_pattern in normalized_patterns:
            pattern = EnsembleKarrasModule._canonical_freeze_pattern(raw_pattern)
            if EnsembleKarrasModule._freeze_pattern_matches(pattern, name):
                return True
        return False

    def _selected_pretrained_regularization_parameters(
            self,
            config: dict[str, Any]) -> list[tuple[str, torch.nn.Parameter]]:
        include_patterns = config.get("include_patterns", ["*"])
        exclude_patterns = config.get("exclude_patterns", [])
        selected_parameters = []
        for name, parameter in self.model.named_parameters():
            if not self._pattern_list_matches(include_patterns, name):
                continue
            if self._pattern_list_matches(exclude_patterns, name):
                continue
            selected_parameters.append((name, parameter))
        if (
            len(selected_parameters) == 0 and
            bool(config.get("strict", True))
        ):
            raise ValueError(
                "pretrained_weight_regularization did not match any model "
                "parameters. Check include_patterns/exclude_patterns."
            )
        return selected_parameters

    def initialize_pretrained_weight_regularization_reference(self) -> None:
        if self._pretrained_regularization_reference is not None:
            return
        config = self._pretrained_weight_regularization_config()
        if config is None:
            return
        device = config.get("device", None)
        reference = {}
        for name, parameter in self._selected_pretrained_regularization_parameters(
                config):
            reference_parameter = parameter.detach().clone()
            if device is not None:
                reference_parameter = reference_parameter.to(device)
            reference[name] = reference_parameter
        self._pretrained_regularization_reference = reference
        self._pretrained_regularization_parameter_names = sorted(reference)

    def pretrained_weight_regularization_loss(self) -> torch.Tensor:
        config = self._pretrained_weight_regularization_config()
        if config is None:
            return next(self.model.parameters()).new_tensor(0.0)

        weight = float(config.get("weight", 0.0))
        if weight <= 0.0:
            return next(self.model.parameters()).new_tensor(0.0)

        self.initialize_pretrained_weight_regularization_reference()
        reference = self._pretrained_regularization_reference or {}
        named_parameters = dict(self.model.named_parameters())
        regularization = next(self.model.parameters()).new_tensor(0.0)
        parameter_count = 0

        for name in self._pretrained_regularization_parameter_names:
            parameter = named_parameters[name]
            if not parameter.requires_grad:
                continue
            reference_parameter = reference[name].to(parameter.device)
            regularization = regularization + (
                parameter - reference_parameter
            ).pow(2).sum()
            parameter_count += parameter.numel()

        if parameter_count == 0:
            return regularization
        if bool(config.get("normalize", True)):
            regularization = regularization / parameter_count
        return weight * regularization

    @staticmethod
    def _scheduled_replay_loss_weight(
            schedule: dict[str, Any],
            default_weight: float,
            position: float) -> float:
        if not bool(schedule.get("enabled", False)):
            return float(default_weight)

        start_weight = float(schedule.get("start_weight", default_weight))
        end_weight = float(schedule.get("end_weight", default_weight))
        duration = float(schedule.get("num_steps",
                                      schedule.get("num_epochs", 1)))
        if duration <= 0:
            progress = 1.0
        else:
            progress = min(max(float(position) / duration, 0.0), 1.0)

        schedule_type = str(schedule.get("type", "linear")).lower()
        if schedule_type == "constant":
            return start_weight
        if schedule_type == "linear":
            return start_weight + progress * (end_weight - start_weight)
        if schedule_type == "cosine":
            cosine_progress = 0.5 - 0.5 * math.cos(math.pi * progress)
            return start_weight + cosine_progress * (end_weight - start_weight)
        raise ValueError(f"Unknown replay_loss_schedule type: {schedule_type}")

    def current_replay_loss_weight(self) -> float:
        default_weight = float(getattr(self.config, "replay_loss_weight", 0.1))
        schedule = getattr(self.config, "replay_loss_schedule", None)
        if not isinstance(schedule, dict):
            return default_weight
        if "num_steps" in schedule:
            position = float(getattr(self, "global_step", 0))
        else:
            position = float(getattr(self, "current_epoch", 0))
        return self._scheduled_replay_loss_weight(
            schedule,
            default_weight,
            position
        )

    def _add_pretrained_regularization_to_loss(
            self,
            loss: torch.Tensor) -> torch.Tensor:
        if not self.has_pretrained_weight_regularization:
            return loss
        regularization_loss = self.pretrained_weight_regularization_loss()
        self.log("train_pretrained_l2_loss", regularization_loss,
                 prog_bar=False, sync_dist=True)
        return loss + regularization_loss

    def _unwrap_replay_batch(self, batch: Any) -> Any:
        if (
            isinstance(batch, (list, tuple)) and
            len(batch) > 0 and
            self._is_replay_batch(batch[0])
        ):
            return batch[0]
        return batch

    def _is_replay_batch(self, batch: Any) -> bool:
        return isinstance(batch, dict) and {"finetune", "replay"} <= set(batch)

    def _require_replay_batch(self, batch: Any) -> dict[str, Any]:
        batch = self._unwrap_replay_batch(batch)
        if not self._is_replay_batch(batch):
            raise ValueError(
                "Replay is enabled, so training_step expects a dict batch with "
                "keys 'finetune' and 'replay'. Check the replay dataloader "
                "wrapper in the trainer."
            )
        return batch

    def _training_loss_from_batch(
            self,
            batch: Any,
            n_ensemble: int,
            autoregressive_log_prefix: Optional[str] = None) -> torch.Tensor:
        x, y, mask = self.select_batch(batch)
        if self.has_autoregressive_loss():
            loss = self.autoregressive_loss_fn(
                x,
                y,
                mask,
                n_ensemble=n_ensemble
            )
            if autoregressive_log_prefix is not None:
                self.log_autoregressive_step_losses(autoregressive_log_prefix)
        else:
            sigma = self.config.noisesampler.sample(x.shape[0]).to(x)
            loss = self.loss_fn(
                x,
                sigma,
                y,
                mask,
                n_ensemble=n_ensemble
            )
        return loss

    def training_step(self, batch, batch_idx):
        if self.has_replay_loss:
            batch = self._require_replay_batch(batch)
            finetune_loss = self._training_loss_from_batch(
                batch["finetune"],
                n_ensemble=self.config.ensemble_size_train,
                autoregressive_log_prefix="train_finetune"
            )
            replay_loss = self._training_loss_from_batch(
                batch["replay"],
                n_ensemble=self.config.ensemble_size_train,
                autoregressive_log_prefix="train_replay"
            )
            replay_weight = self.current_replay_loss_weight()
            loss = finetune_loss + replay_weight * replay_loss
            loss = self._add_pretrained_regularization_to_loss(loss)
            self.log("train_loss_finetune", finetune_loss,
                     prog_bar=False, sync_dist=True)
            self.log("train_loss_replay", replay_loss,
                     prog_bar=False, sync_dist=True)
            self.log("train_replay_loss_weight",
                     torch.as_tensor(replay_weight, device=loss.device),
                     prog_bar=False,
                     sync_dist=True)
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)
            return loss

        if self._is_replay_batch(self._unwrap_replay_batch(batch)):
            raise ValueError(
                "Received a replay-style batch, but config.replay_enabled is "
                "False. Disable the replay dataloader wrapper or enable replay."
            )

        loss = self._training_loss_from_batch(
            batch,
            n_ensemble=self.config.ensemble_size_train,
            autoregressive_log_prefix="train"
        )
        loss = self._add_pretrained_regularization_to_loss(loss)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss
    
    def _validation_dataloader_name(self, dataloader_idx: int) -> str:
        names = getattr(self, "validation_dataloader_names", None)
        if names is None:
            names = ("finetune", "replay")
        if dataloader_idx < len(names):
            return str(names[dataloader_idx])
        return f"dataloader_{dataloader_idx}"

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        dataloader_name = self._validation_dataloader_name(dataloader_idx)
        has_multiple_validation_loaders = (
            len(getattr(self, "validation_dataloader_names", ("finetune",))) > 1
        )
        primary_validation = dataloader_idx == 0
        validation_prefix = (
            f"valid_{dataloader_name}"
            if has_multiple_validation_loaders
            else "valid"
        )

        loss = self._training_loss_from_batch(
            batch,
            n_ensemble=self.config.ensemble_size_val,
            autoregressive_log_prefix=None
        )

        if self.has_autoregressive_loss():
            self.log_autoregressive_step_losses(validation_prefix)
            if primary_validation and validation_prefix != "valid":
                self.log_autoregressive_step_losses("valid")

        if has_multiple_validation_loaders:
            self.log(f"valid_loss_{dataloader_name}", loss,
                     prog_bar=False, sync_dist=True)
            self.log(f"val_loss_{dataloader_name}", loss,
                     prog_bar=False, sync_dist=True)

        if primary_validation:
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

    def start_ema(self):
        if not getattr(self.config, "ema_enabled", False):
            self.ema_tracker = None
            return
        self.ema_tracker = ModelEMA(
            self.model,
            ema_type=getattr(self.config, "ema_type", "traditional"),
            decay=getattr(self.config, "ema_decay", 0.999),
            halflife_steps=getattr(self.config, "ema_halflife_steps", None),
            rampup_ratio=getattr(self.config, "ema_rampup_ratio", None),
            power_function_stds=getattr(
                self.config,
                "ema_power_function_stds",
                None,
            ),
            device=getattr(self.config, "ema_device", None),
            profile_index=getattr(self.config, "ema_profile_index", 0),
        )

    @property
    def has_ema(self):
        return self.ema_tracker is not None

    def on_fit_start(self):
        if self.has_ema and not self._ema_loaded_from_checkpoint:
            self.ema_tracker.reset(self.model)
        self.initialize_pretrained_weight_regularization_reference()

    def on_before_zero_grad(self, optimizer):
        if self.has_ema:
            self.ema_tracker.update(self.model)

    def on_save_checkpoint(self, checkpoint):
        if self.has_ema:
            checkpoint["model_ema"] = self.ema_tracker.state_dict()
        config = self._pretrained_weight_regularization_config()
        if (
            config is not None and
            bool(config.get("save_reference_in_checkpoint", True)) and
            self._pretrained_regularization_reference is not None
        ):
            checkpoint["pretrained_weight_regularization_reference"] = {
                name: parameter.detach().cpu()
                for name, parameter
                in self._pretrained_regularization_reference.items()
            }
            checkpoint["pretrained_weight_regularization_parameter_names"] = list(
                self._pretrained_regularization_parameter_names
            )

    def on_load_checkpoint(self, checkpoint):
        if self.has_ema and "model_ema" in checkpoint:
            self.ema_tracker.load_state_dict(checkpoint["model_ema"])
            self._ema_loaded_from_checkpoint = True
        if "pretrained_weight_regularization_reference" in checkpoint:
            self._pretrained_regularization_reference = checkpoint[
                "pretrained_weight_regularization_reference"
            ]
            self._pretrained_regularization_parameter_names = checkpoint.get(
                "pretrained_weight_regularization_parameter_names",
                sorted(self._pretrained_regularization_reference)
            )

    def on_validation_epoch_start(self):
        if self._should_use_ema_for_validation():
            self._ema_validation_backup = self.ema_tracker.apply_to(self.model)
            self._ema_scope_depth += 1

    def on_validation_epoch_end(self):
        if self._ema_validation_backup is not None:
            self.ema_tracker.restore(self.model, self._ema_validation_backup)
            self._ema_validation_backup = None
            self._ema_scope_depth = max(self._ema_scope_depth - 1, 0)

    def _should_use_ema_for_validation(self):
        return (
            self.has_ema and
            getattr(self.config, "ema_use_for_validation", True) and
            self._ema_scope_depth == 0
        )

    def _should_use_ema_for_sampling(self):
        return (
            self.has_ema and
            getattr(self.config, "ema_use_for_sampling", True) and
            not self.training and
            self._ema_scope_depth == 0
        )

    @contextmanager
    def ema_scope(self, enabled: bool = True):
        if not enabled or not self.has_ema or self._ema_scope_depth > 0:
            yield
            return
        backup = self.ema_tracker.apply_to(self.model)
        self._ema_scope_depth += 1
        try:
            yield
        finally:
            self.ema_tracker.restore(self.model, backup)
            self._ema_scope_depth = max(self._ema_scope_depth - 1, 0)

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

from typing import Callable, Any

import torch
import lightning
from torch import Tensor
from jaxtyping import Float

from diffsci.torchutils import (broadcast_from_below,
                                linear_interpolation,
                                dict_unsqueeze,
                                dict_to)
from diffsci.utils import get_minibatch_sizes
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
                 loss_metric: str = "huber",
                 tag: str = "custom",
                 has_edm_batch_norm: bool = False,
                 dynamic_loss_weight: int | None = None,
                 extra_args: None | dict[str, Any] = None):
        self.preconditioner = preconditioner
        self.noisesampler = noisesampler
        self.noisescheduler = noisescheduler
        self.loss_metric = loss_metric
        self.tag = tag
        self.has_edm_batch_norm = has_edm_batch_norm
        self.dynamic_loss_weight = dynamic_loss_weight
        if extra_args is None:
            self.extra_args = dict()
        else:
            self.extra_args = extra_args

    @classmethod
    def from_edm(self,
                 sigma_data: float = 0.5,
                 prior_mean: float = -1.2,
                 prior_std: float = 1.2,
                 has_edm_batch_norm: bool = False,
                 dynamic_loss_weight: int | None = None):

        preconditioner = preconditioners.EDMPreconditioner(
                            sigma_data=sigma_data
                        )
        noisesampler = noisesamplers.EDMNoiseSampler(
                            sigma_data=sigma_data,
                            prior_mean=prior_mean,
                            prior_std=prior_std
                        )
        noisescheduler = schedulers.EDMScheduler()
        loss_metric = "huber"
        tag = "edm"
        extra_args = {"sigma_data": sigma_data,
                      "prior_mean": prior_mean,
                      "prior_std": prior_std}
        return KarrasModuleConfig(preconditioner=preconditioner,
                                  noisesampler=noisesampler,
                                  noisescheduler=noisescheduler,
                                  loss_metric=loss_metric,
                                  tag=tag,
                                  has_edm_batch_norm=has_edm_batch_norm,
                                  dynamic_loss_weight=dynamic_loss_weight,
                                  extra_args=extra_args)

    @classmethod
    def from_vp(self,
                beta_data: float = 19.9,
                beta_min: float = 0.1,
                epsilon_min: float = 1e-3,
                epsilon_sampler: float = 1e-5,
                M: int = 1000):
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
        loss_metric = "huber"
        tag = "vp"
        extra_args = {"beta_data": beta_data,
                      "beta_min": beta_min,
                      "epsilon_min": epsilon_min,
                      "epsilon_sampler": epsilon_sampler,
                      "M": M}
        return KarrasModuleConfig(preconditioner=preconditioner,
                                  noisesampler=noisesampler,
                                  noisescheduler=noisescheduler,
                                  loss_metric=loss_metric,
                                  tag=tag,
                                  extra_args=extra_args)

    @classmethod
    def from_ve(self,
                sigma_min: float = 0.02,
                sigma_max: float = 100):
        noisescheduler = schedulers.VEScheduler(sigma_min=sigma_min,
                                                sigma_max=sigma_max)
        preconditioner = preconditioners.VEPreconditioner()
        noisesampler = noisesamplers.VENoiseSampler(
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        loss_metric = "huber"
        tag = "ve"
        extra_args = {"sigma_min": sigma_min,
                      "sigma_max": sigma_max}
        return KarrasModuleConfig(preconditioner=preconditioner,
                                  noisesampler=noisesampler,
                                  noisescheduler=noisescheduler,
                                  loss_metric=loss_metric,
                                  tag=tag,
                                  extra_args=extra_args)

    @classmethod
    def conditionalSR3(self,
                       sigma_min: float = 0.02,
                       sigma_max: float = 100):
        noisescheduler = schedulers.EDMScheduler(sigma_min=sigma_min,
                                                 sigma_max=sigma_max)
        preconditioner = preconditioners.SR3Preconditioner()
        noisesampler = noisesamplers.EDMNoiseSampler(
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
        loss_metric = "huber"
        tag = "conditionalSR3"
        extra_args = {"sigma_min": sigma_min,
                      "sigma_max": sigma_max}
        return KarrasModuleConfig(preconditioner=preconditioner,
                                  noisesampler=noisesampler,
                                  noisescheduler=noisescheduler,
                                  loss_metric=loss_metric,
                                  tag=tag,
                                  extra_args=extra_args)

    def export_description(self) -> dict[str, Any]:
        return dict(tag=self.tag,
                    extra_args=self.extra_args)

    @classmethod
    def load_from_description_with_tag(self,
                                       description: dict[str, Any]):
        tag = description["tag"]
        extra_args = description["extra_args"]
        if tag == "custom":
            raise ValueError("Cannot load from a custom tag")
        elif tag == "edm":
            return KarrasModuleConfig.from_edm(**extra_args)
        elif tag == "vp":
            return KarrasModuleConfig.from_vp(**extra_args)
        elif tag == "ve":
            return KarrasModuleConfig.from_ve(**extra_args)
        elif tag == "conditionalSR3":
            return KarrasModuleConfig.conditionalSR3(**extra_args)

    @property
    def has_dynamic_loss_weight(self):
        return self.dynamic_loss_weight is not None


class KarrasModule(lightning.LightningModule):
    """
    A diffusion model using the framework found in
    "Elucidating the Design Space of Diffusion-Based
     Generative Models", by Karras et al, 2022.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 config: KarrasModuleConfig,
                 conditional: bool = False,
                 masked: bool = False,
                 autoencoder: None | torch.nn.Module = None,
                 autoencoder_conditional: bool = False):
        super().__init__()
        self.model = model
        self.config = config
        self.conditional = conditional
        self.masked = masked
        self.autoencoder = autoencoder
        if self.autoencoder:
            self.freeze_autoencoder()
        self.autoencoder_conditional = autoencoder_conditional
        self.set_optimizer_and_scheduler()
        self.set_loss_metric()
        self.start_edm_batch_norm()
        self.start_dynamic_loss_weight()
        self.norm = 1.0    # TODO: find better way to normalize latent space

    def export_description(self) -> dict[str, Any]:
        config_description = self.config.export_description()
        conditional = self.conditional
        masked = self.masked
        autoencoder = True if self.autoencoder else False
        autoencoder_conditional = self.autoencoder_conditional
        return dict(config_description=config_description,
                    conditional=conditional,
                    masked=masked,
                    autoencoder=autoencoder,
                    autoencoder_conditional=autoencoder_conditional)

    def freeze_autoencoder(self):
        """
        Freezes the autoencoder to prevent its weights from being updated
        during training.
        """
        for param in self.autoencoder.parameters():
            param.requires_grad = False

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
        Set the loss function to be used.
        """
        if self.config.loss_metric == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif self.config.loss_metric == "huber":
            self.loss_metric = torch.nn.HuberLoss(reduction="none")
        # elif self.config.loss_metric == "sinkhorn":
            # self.config.loss_metric = SinkhornLoss()
        else:
            raise ValueError(f"loss_type {self.loss_metric} not recognized")

    def loss_fn(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821
                sigma: Float[Tensor, "batch"],  # noqa: F821
                y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821
                mask: None | Float[Tensor, "batch *shape"] = None  # noqa: F821
                ) -> Float[Tensor, ""]:  # noqa: F821, F722

        """
        Parameters
        ---------
        x : torch.Tensor of shape [B, *[shapex]], the original noise
        sigma : torch.Tensor of shape [B]
        y : None or torch.Tensor of shape [B, *[yshape]], the conditional data,
        depending on whether we are dealing with a conditional or unconditional
        model
        """
        x = self.encode(x, y)
        broadcasted_sigma = broadcast_from_below(sigma, x)  # [nbatch, *1]
        noise = broadcasted_sigma*torch.randn_like(x)  # [nbatch, *shapex]
        x_noised = x + noise  # [nbatch, *shapex]
        denoiser, cond_noise = self.get_denoiser(x_noised, sigma, y)  # [nbatch, *shapex]
        weight = self.config.noisesampler.loss_weighting(
                    broadcasted_sigma
                )  # [nbatch, *1]
        bias = torch.zeros_like(weight)
        if self.config.has_dynamic_loss_weight:
            modifier = self.dynamic_loss_weight(cond_noise)  # [nbatch]
            modifier = broadcast_from_below(modifier, x)  # [nbatch, *1]
            weight = weight/torch.exp(modifier)
            bias = bias + modifier
        # Compute the loss
        loss = self.loss_metric(denoiser, x)  # [nbatch]

        if mask is not None:
            # Apply the mask if it is provided
            # We assume that the mask is 1 where the data is absent
            mask = mask.expand_as(loss)
            adjusted_loss = loss * (1 - mask)
            loss = (weight * adjusted_loss + bias).mean()
        else:
            # Compute mean loss as usual if no mask is provided
            loss = (weight * loss + bias).mean()

        return loss

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

    def sample(
            self,
            nsamples: int,
            shape: list[int],
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            guidance: float = 1.0,
            nsteps: int = 100,
            record_history: bool = False,
            maximum_batch_size: None | int = None,
            integrator: None | str | integrators.Integrator = None
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
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
                                          integrator=integrator))
            catdim = 1 if record_history else 0
            result = torch.cat(result, dim=catdim)
            return result
        else:
            batched_shape = [nsamples] + list(shape)
            white_noise = torch.randn(*batched_shape).to(self.device)
            if y is not None:
                y = dict_to(y, self.device)
            if self.latent_model:  # TODO: A stupid hack. Should be improved
                white_noise = self.encode(white_noise, y)
                white_noise = torch.randn_like(white_noise)
            result = self.propagate_white_noise(
                        white_noise,
                        y,
                        guidance,
                        nsteps,
                        record_history,
                        integrator=integrator)
            return result

    def propagate_white_noise(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            guidance: float = 1.0,
            nsteps: int = 100,
            record_history: bool = False,
            integrator: None | str | integrators.Integrator = None
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        x = x*self.config.noisescheduler.maximum_scale
        result = self.propagate_toward_sample(x,
                                              y,
                                              guidance,
                                              nsteps,
                                              record_history,
                                              integrator=integrator)
        result = self.decode(result, y)
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

    def encode(self, x, y=None):
        if self.latent_model:
            if self.autoencoder_conditional:
                x = self.autoencoder.encode(x, y)
            else:
                x = self.autoencoder.encode(x)
        else:
            x = x
        if self.config.has_edm_batch_norm:
            x = self.edm_batch_norm.normalize(x)
        return x / self.norm

    def decode(self, x, y=None):
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
            self.edm_batch_norm = edmbatchnorm.EDMBatchNorm(sigma=sigma_data)

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
        h = x*self.fourier_weights + self.fourier_bias  # [batch, nhidden]
        h = torch.cos(h)  # [batch, nhidden]
        h = self.linear(h)  # [batch, 1]
        h = h.squeeze(1)  # [batch]
        return h

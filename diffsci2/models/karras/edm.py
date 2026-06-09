from typing import Literal
from jaxtyping import Float
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np
import lightning

from diffsci2.torchutils import broadcast_from_below, dict_unsqueeze
from diffsci2.models.aux_scripts import DimensionAgnosticBatchNorm, IdentityBatchNorm, HyperparameterManager


SampleType = Float[Tensor, "batch *shape"]
ConditionType = Float[Tensor, "batch *yshape"]
TimeType = Float[Tensor, "batch"]


# TODO: Finish this module

class EDMModuleConfig(torch.nn.Module):
    def __init__(self,
                 num_channels: int | None = None,
                 initial_norm: bool = False,
                 loss_metric: Literal['mse', 'huber'] = 'huber',
                 sigma_data: float = 0.5,
                 prior_mean: float = -1.2,
                 prior_std: float = 1.2,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 expoent_steps: float = 7.0):

        super().__init__()
        self.num_channels = num_channels
        self.initial_norm = initial_norm
        self.loss_metric = loss_metric

        self.sigma_data = sigma_data
        self.prior_mean = prior_mean
        self.prior_std = prior_std

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.expoent_steps = expoent_steps

        self.set_edm_functions()
        self.set_loss_metric_module()

    def set_edm_functions(self):
        self.loss_weighting = lambda sigma: (sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2)
        self.noise_conditioner = lambda sigma: 0.5*torch.log(sigma)
        self.input_scaling = lambda sigma: 1/torch.sqrt(sigma**2 + self.sigma_data**2)
        self.output_scaling = lambda sigma: sigma*self.sigma_data/torch.sqrt(sigma**2 + self.sigma_data**2)
        self.skip_scaling = lambda sigma: self.sigma_data**2/(sigma**2 + self.sigma_data**2)

    def sample_sigma(self, shape: list[int]) -> Float[Tensor, "batch"]:
        white_noise = torch.randn(shape)
        logsigma = white_noise*self.prior_std + self.prior_mean
        sigma = torch.exp(logsigma)
        return sigma

    def create_sigma_steps(self, n: int) -> Float[Tensor, "n"]:
        s = torch.arange(n) / (n)
        start = self.sigma_max**(1/self.expoent_steps)
        end = self.sigma_min**(1/self.expoent_steps)
        steps = (start + s*(end - start))**(self.expoent_steps) + 1e-6
        return steps

    def set_loss_metric_module(self):
        if self.loss_metric == 'mse':
            self.loss_metric_module = torch.nn.MSELoss(reduction="none")
        elif self.loss_metric == 'huber':
            self.loss_metric_module = torch.nn.HuberLoss(reduction="none")
        else:
            raise ValueError(f"Invalid loss metric: {self.loss_metric}")


class EDMModule(lightning.LightningModule):
    def __init__(self, config: EDMModuleConfig, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.set_initial_norm()

    def set_initial_norm(self):
        if self.config.initial_norm:
            self.initial_norm = DimensionAgnosticBatchNorm(self.config.num_channels)
        else:
            self.initial_norm = IdentityBatchNorm()

    def encode(self, x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
               y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821, typing
               ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        return self.initial_norm(x)

    def decode(self, x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
               y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821, typing
               ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        return self.initial_norm.unnorm(x)

    def loss_fn(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
                sigma: Float[Tensor, "batch"],  # noqa: F821, typing
                y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821, typing
                mask: None | Float[Tensor, "batch *shape"] = None  # noqa: F821, typing
                ) -> Float[Tensor, ""]:  # noqa: F821, F722
        x = self.encode(x, y)
        noise = torch.randn_like(x)
        broadcasted_sigma = broadcast_from_below(sigma, x)  # [nbatch, *1]
        x_noised = x + broadcasted_sigma * noise
        denoised = self.evaluate_denoiser(x_noised, sigma, y)
        loss = self.config.loss_metric_module(denoised, x)
        if mask is not None:
            # Apply the mask if it is provided
            # We assume that the mask is 1 where the data is absent
            mask = mask.expand_as(loss)
            loss = loss * (1 - mask)
        loss = loss.mean()
        return loss

    def evaluate_denoiser(self,
                          x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
                          sigma: Float[Tensor, "batch"],  # noqa: F821, typing
                          y: None | Float[Tensor, "batch *yshape"] = None,  # noqa: F821, typing
                          guidance: float = 1.0
                          ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        input_scale_factor = self.config.input_scaling(
            sigma)  # [nbatch]
        input_scale_factor = broadcast_from_below(input_scale_factor,
                                                  x_noised)  # [nbatch, *1]
        output_scale_factor = self.config.output_scaling(
            sigma)  # [nbatch]
        output_scale_factor = broadcast_from_below(output_scale_factor,
                                                   x_noised)  # [nbatch, *1]
        skip_scale_factor = self.config.skip_scaling(
            sigma)  # [nbatch]
        skip_scale_factor = broadcast_from_below(skip_scale_factor,
                                                 x_noised)  # [nbatch, *1]
        scaled_input = input_scale_factor*x_noised  # [nbatch, *shapex]
        cond_noise = self.config.noise_conditioner(
            sigma)  # [nbatch]
        if y is not None and guidance != 0.0:
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
        denoiser = scaled_output + skip_scale_factor*x_noised  # [nbatch, *shape]
        return denoiser

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch.get('y', None)
        mask = batch.get('mask', None)
        sigma = self.sample_sigmastep(x.shape[0]).to(x)
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

    def sample(self,
               nsamples: int,
               shape: list[int],
               y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
               guidance: float = 1.0,
               nsteps: int = 30
               ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        if torch.inference_mode():
            x = torch.randn(nsamples, *shape).to(self.device)
            if y is not None:
                y = dict_unsqueeze(y, 0)
            sigma_steps = self.config.create_sigma_steps(nsteps).to(x)
            x = self.integrate_probability_flow(x, sigma_steps, y, guidance)
        return x  # noqa: F821, F722

    def get_probability_flow_rhs(self,
                                 x_noised: Float[Tensor, "batch *shape"],  # noqa: F821, typing
                                 sigma: Float[Tensor, "batch"],  # noqa: F821, typing
                                 y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
                                 guidance: float = 1.0
                                 ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, F722
        denoised = self.evaluate_denoiser(x_noised, sigma, y, guidance)
        return -(denoised - x_noised)/(sigma)

    def integrate_probability_flow(
        self,
        x: Float[Tensor, "batch *shape"],  # noqa: F821, typing
        time_schedule: Float[Tensor, "nsteps"],  # noqa: F821, typing
        y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821, typing
        guidance: float = 1.0,
        return_history: bool = False
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        # Integrate the flow field x' = v(x, t) using the Heun method
        self.model.eval()
        if return_history:
            history = [(time_schedule[0], x)]
        for i in range(len(time_schedule) - 1):
            t_curr = time_schedule[i]
            t_next = time_schedule[i + 1]
            dt = t_next - t_curr

            t_curr = t_curr * (torch.ones(x.shape[0]).to(x))
            t_next = t_next * (torch.ones(x.shape[0]).to(x))

            # Heun method
            # First step - Euler
            v1 = self.get_flow_field(x, t_curr, y=y, guidance=guidance)
            x_euler = x + dt * v1

            # Second step - correction
            v2 = self.get_flow_field(x_euler, t_next, y=y, guidance=guidance)
            x = x + dt * (v1 + v2) / 2

            if return_history:
                history.append((time_schedule[i + 1], x))

        if not return_history:
            x = self.decode(x)
            return x
        else:
            history = list(map(lambda tx: (tx[0], self.decode(tx[1])), history))
            return history

from typing import Literal
from jaxtyping import Float
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np
import lightning

from diffsci.torchutils import broadcast_from_below, dict_unsqueeze
from diffsci.models.aux_scripts import DimensionAgnosticBatchNorm, IdentityBatchNorm, HyperparameterManager


SampleType = Float[Tensor, "batch *shape"]
ConditionType = Float[Tensor, "batch *yshape"]
TimeType = Float[Tensor, "batch"]


class SIModuleConfig(torch.nn.Module):
    def __init__(self,
                 scheduler: Literal['linear', 'cosine'] = 'linear',
                 num_channels: int | None = None,
                 initial_norm: bool = False,
                 loss_metric: Literal['mse', 'huber'] = 'huber'):
        super().__init__()
        self.scheduler = scheduler
        self.num_channels = num_channels
        self.initial_norm = initial_norm
        self.loss_metric = loss_metric
        self.set_scheduling_functions()
        self.set_loss_metric_module()

    def set_scheduling_functions(self):
        if self.scheduler == 'linear':
            self.alpha_fn = lambda t: 1 - t  # noqa
            self.sigma_fn = lambda t: t  # noqa
            self.alpha_fn_dot = lambda t: -1 * torch.ones_like(t)  # noqa
            self.sigma_fn_dot = lambda t: torch.ones_like(t)  # noqa
        elif self.scheduler == 'cosine':
            self.alpha_fn = lambda t: torch.cos(t * np.pi / 2)  # noqa
            self.sigma_fn = lambda t: torch.sin(t * np.pi / 2)  # noqa
            self.alpha_fn_dot = lambda t: -1 * torch.pi / 2 * torch.sin(t * np.pi / 2)  # noqa
            self.sigma_fn_dot = lambda t: torch.pi / 2 * torch.cos(t * np.pi / 2)  # noqa
        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")

    def set_loss_metric_module(self):
        if self.loss_metric == 'mse':
            self.loss_metric_module = torch.nn.MSELoss(reduction="none")
        elif self.loss_metric == 'huber':
            self.loss_metric_module = torch.nn.HuberLoss(reduction="none")
        else:
            raise ValueError(f"Invalid loss metric: {self.loss_metric}")


class SIModule(lightning.LightningModule):
    def __init__(self, config: SIModuleConfig, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.set_initial_norm()

        # self.hp_manager = HyperparameterManager()
        # self.hp_manager.add_model_config(self.model)
        # self.hp_manager.log_to_wandb()

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
        x = self.initial_norm(x)
        noise = torch.randn_like(x)
        t_broadcasted = broadcast_from_below(t, x)
        alpha, sigma = self.config.alpha_fn(t_broadcasted), self.config.sigma_fn(t_broadcasted)
        x_noised = alpha * x + sigma * noise
        flow_field = self.model(x_noised, t, y=y)

        alpha_dot, sigma_dot = self.config.alpha_fn_dot(t_broadcasted), self.config.sigma_fn_dot(t_broadcasted)
        target = (alpha_dot * x + sigma_dot * noise)

        loss = self.config.loss_metric_module(flow_field, target)
        if mask is not None:
            # Apply the mask if it is provided
            # We assume that the mask is 1 where the data is absent
            mask = mask.expand_as(loss)
            loss = loss * (1 - mask)
        loss = loss.mean()
        return loss

    def sample_timestep(self, nsamples):
        # Sample from uniform 0 and 1
        t = torch.rand(nsamples)
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
            y: None | Float[Tensor, "*yshape"] = None  # noqa: F821, typing
    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821, typing
        if guidance == 1.0 or y is None:  # Implictly no guidance
            flow_field = self.model(x_noised, t, y=y)
        else:
            flow_field = guidance * self.model(x_noised, t, y=y) + (1 - guidance) * self.model(x_noised, t, y=None)
        return flow_field

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
            time_schedule = torch.linspace(1, 0, nsteps).to(x)
            x = self.integrate_flow_field(x, time_schedule, y, guidance)
        return x  # noqa: F821, F722

    def integrate_flow_field(
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
            x = self.initial_norm.unnorm(x)
            return x
        else:
            history = list(map(lambda tx: (tx[0], self.initial_norm.unnorm(tx[1])), history))
            return history

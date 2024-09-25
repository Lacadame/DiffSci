import torch
import lightning
from torch import Tensor
from jaxtyping import Float

from diffsci.torchutils import broadcast_from_below
from diffsci.utils import get_minibatch_sizes
from . import schedulers
from . import integrators


class DDPMModuleConfig(object):
    def __init__(self,
                 scheduler: schedulers.DDPMScheduler,
                 integrator: integrators.Integrator,
                 loss_metric: str = "huber"):
        self.scheduler = scheduler
        self.integrator = integrator
        self.loss_metric = loss_metric

    @classmethod
    def from_classical_ddpm(self,
                            integrator_type: int = 1,
                            scheduler: str = 'classical'):
        if scheduler == 'classical':
            scheduler = schedulers.ClassicalDDPMScheduler()
        elif scheduler == 'exp':
            scheduler = schedulers.ExpDDPMScheduler()
        elif scheduler == 'cosine':
            scheduler = schedulers.CosineDDPMScheduler()
        if integrator_type == 1:
            integrator = integrators.ClassicalDDPMIntegratorType1(scheduler)
        elif integrator_type == 2:
            integrator = integrators.ClassicalDDPMIntegratorType2(scheduler)
        else:
            raise NotImplementedError
        loss_metric = "huber"
        return DDPMModuleConfig(scheduler=scheduler,
                                integrator=integrator,
                                loss_metric=loss_metric)

    @classmethod
    def from_ddpm(self,
                  scheduler: str = 'classical'):
        if scheduler == 'classical':
            scheduler = schedulers.ClassicalDDPMScheduler()
        elif scheduler == 'exp':
            scheduler = schedulers.ExpDDPMScheduler()
        elif scheduler == 'cosine':
            scheduler = schedulers.CosineDDPMScheduler()
        integrator = integrators.DDPMIntegrator(scheduler)
        loss_metric = "huber"
        return DDPMModuleConfig(scheduler=scheduler,
                                integrator=integrator,
                                loss_metric=loss_metric)

    @classmethod
    def from_ddim(self,
                  scheduler: str = 'classical'):
        if scheduler == 'classical':
            scheduler = schedulers.ClassicalDDPMScheduler()
        elif scheduler == 'exp':
            scheduler = schedulers.ExpDDPMScheduler()
        elif scheduler == 'cosine':
            scheduler = schedulers.CosineDDPMScheduler()
        integrator = integrators.DDIMIntegrator(scheduler)
        loss_metric = "huber"
        return DDPMModuleConfig(scheduler=scheduler,
                                integrator=integrator,
                                loss_metric=loss_metric)

    def change_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.integrator.scheduler = scheduler


class DDPMModule(lightning.LightningModule):
    """
    A diffusion model using the framework found in
    "Elucidating the Design Space of Diffusion-Based
     Generative Models", by Karras et al, 2022.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 config: DDPMModuleConfig,
                 conditional: bool = False):
        super().__init__()
        self.model = model
        self.config = config
        self.conditional = conditional
        self.set_optimizer_and_scheduler()
        self.set_loss_metric()

    def set_optimizer_and_scheduler(
            self,
            optimizer: torch.optim.Optimizer | None = None,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            scheduler_interval: str = "epoch"):
        """
        Parameters
        ----------
        optimizer : None | torch.optim.Optimizer
            if None, use the default optimizer AdamW,
            with learning rate 1e-3, betas=(0.9, 0.999),
            and weight decay 1e-4
        scheduler : None | torch.optim.lr_scheduler._LRScheduler
            if None, use the default scheduler LambdaLR with lambda = 1
        scheduler_interval : str
            "epoch" or "step", whether the scheduler should be called at the
            end of each epoch or each step.
        """
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0 + 0*step
            )  # Neutral learning rate scheduler

        self.lr_scheduler_interval = scheduler_interval

    def set_loss_metric(self):
        """
        Set the loss function to be used.
        """
        if self.config.loss_metric == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif self.config.loss_metric == "huber":
            self.loss_metric = torch.nn.HuberLoss(reduction="none")
        else:
            raise ValueError(f"loss_type {self.loss_metric} not recognized")

    def loss_fn(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821
                t: Float[Tensor, "batch"],  # noqa: F821
                y: None | Float[Tensor, "batch *yshape"] = None  # noqa: F821
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
        if self.conditional:
            assert (y is not None)
        elif not self.conditional:
            assert (y is None)
        noise = torch.randn_like(x)  # [nbatch, *shapex]
        calpha = self.config.scheduler.calpha(t)  # [nbatch]
        calpha_b = broadcast_from_below(calpha, x)  # [nbatch, *1]
        x_noised = torch.sqrt(calpha_b)*x + torch.sqrt(1 - calpha_b)*noise
        if self.conditional:
            noise_prediction = self.model(x_noised, t, y)  # [nbatch, *shapex]
        else:
            noise_prediction = self.model(x_noised, t)
        weight = 1.0  # TODO: Put an actual weighting function here
        loss = (weight*self.loss_metric(noise_prediction, noise)).mean()  # []
        return loss

    def sample_time_for_training(self,
                                 x: Float[Tensor, "batch *shape"]  # noqa: F821
                                 ) -> Float[Tensor, "batch"]:  # noqa: F821
        T = self.config.scheduler.T
        t = torch.randint(1, T+1, size=(x.shape[0],)).to(x)  # [nbatch]
        return t

    def sample(
            self,
            nsamples: int,
            shape: list[int],
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            nsteps: None | int = None,
            record_history: bool = False,
            maximum_batch_size: None | int = None,
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        if maximum_batch_size is not None:
            batch_sizes = get_minibatch_sizes(nsamples, maximum_batch_size)
            result = []
            for batch_size in batch_sizes:
                result.append(self.sample(batch_size,
                                          shape,
                                          y,
                                          nsteps,
                                          record_history,
                                          maximum_batch_size=None))
            result = torch.cat(result, dim=0)
            return result
        else:
            batched_shape = [nsamples] + list(shape)
            white_noise = torch.randn(*batched_shape).to(self.device)
            return self.propagate_toward_sample(
                white_noise,
                y,
                nsteps,
                record_history)

    def propagate_toward_sample(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            y: None | Float[Tensor, "*yshape"] = None,  # noqa: F821
            nsteps: None | int = None,
            record_history: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape
        if y is not None:
            y = y.unsqueeze(0)  # Broadcasting will take care of the rest

        def rhs(x, t):
            with torch.inference_mode():
                if self.conditional:
                    return self.model(x, t, y)
                else:
                    return self.model(x, t)

        result = self.config.integrator.propagate_backward(
                    x,
                    rhs,
                    nsteps=nsteps,
                    record_history=record_history)
        return result

    def propagate_toward_noise(
            self,
            x: Float[Tensor, "nsamples *shape"],  # noqa: F821
            nsteps: int = 100,
            record_history: bool = False
            ) -> Float[Tensor, "..."]:  # TODO: Put the actual shape

        result = self.config.integrator.propagate_forward(
            x,
            nsteps,
            record_history=record_history
        )

        return result

    def training_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.sample_time_for_training(x)  # [nbatch]
        loss = self.loss_fn(x, t, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.sample_time_for_training(x)  # [nbatch]
        loss = self.loss_fn(x, t, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.lr_scheduler is not None:
            lr_scheduler_config = {"scheduler": self.lr_scheduler,
                                   "interval": self.lr_scheduler_interval}
            return [self.optimizer], [lr_scheduler_config]
        else:  # Just fo backward compatibility for some examples
            return self.optimizer

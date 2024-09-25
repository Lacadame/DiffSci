import torch
import lightning

from porenet.torchutils import broadcast_from_below
from . import DDPMSampler


class DDPMModule(lightning.LightningModule):
    def __init__(self, model, scheduler,
                 conditional=True,
                 loss_type="mse",
                 loss_scale_factor=1.0,
                 loss_scaling="constant"):
        """
        Train a model according to the DDPM framework in
        "Denoising Diffusion Probabilistic Models" by Ho et al.

        Parameters
        ----------
        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape [B, [shape]], the original noise
                t : torch.Tensor of shape [B]
                and, if conditional=True
                y : torch.Tensor of shape [B, [yshape]], the conditional data
                and as output
                torch.Tensor of shape [B, [shape]]
        scheduler : DDPMScheduler
        conditional : bool
            whether we are dealing with a conditional or unconditional model
        loss_type : str
            what kind of loss are we using. Options: ["mse", "huber"].
        loss_scale_factor : float
        """

        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.conditional = conditional
        self.loss_type = loss_type
        self.loss_scale_factor = loss_scale_factor
        self.loss_scaling = loss_scaling
        self.set_optimizer_and_scheduler()
        self.set_sampler()
        self.set_loss(loss_type)

    def set_optimizer_and_scheduler(self,
                                    optimizer=None,
                                    lr_scheduler="default",
                                    lr_scheduler_interval="epoch"):
        """
        Parameters
        ----------
        optimizer : None | torch.optim.Optimizer
            if None, use the default optimizer AdamW,
            with learning rate 1e-3, betas=(0.9, 0.999),
            and weight decay 1e-4
        lr_scheduler : "default" | None | torch.optim.lr_scheduler._LRScheduler
            if "default", use the default scheduler
            CosineAnnealingWarmRestarts, with T_0=10.
            If None, don't use any scheduler
        lr_scheduler_interval : str
            "epoch" or "step", whether the scheduler should be called at the
            end of each epoch or each step.
        loss_scaling : str
            One of ["constant", "default"]
        """
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.parameters(),
                                               lr=1e-3,
                                               betas=(0.9, 0.999),
                                               weight_decay=1e-4)
        if lr_scheduler is None:
            self.lr_scheduler = None
        elif isinstance(lr_scheduler, str) and lr_scheduler == "default":
            self.lr_scheduler = (torch.optim.lr_scheduler.
                                 CosineAnnealingWarmRestarts(self.optimizer,
                                                             T_0=10))
        else:
            self.lr_scheduler = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval

    def set_sampler(self, sampler_cls=DDPMSampler, shape=None,
                    *args, **kwargs):
        self.sampler = DDPMSampler(self.model, self.scheduler, shape=shape,
                                   *args, **kwargs)

    def set_loss(self, loss_type):
        """
        Set the loss function to be used.

        Parameters
        ----------
        loss_type : str
            what kind of loss are we using. Options: ["mse", "huber"].
        """
        self.loss_type = loss_type
        if loss_type == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif loss_type == "huber":
            self.loss_metric = torch.nn.HuberLoss(reduction="none")
        else:
            raise ValueError(f"loss_type {loss_type} not recognized")

    def loss_fn(self, x, t, y=None):

        """
        Parameters
        ---------
        x : torch.Tensor of shape [B, *[shapex]], the original noise
        t : torch.Tensor of shape [B]
        y : None or torch.Tensor of shape [B, *[yshape]], the conditional data,
        depending on whether we are dealing with a conditional or unconditional
        model
        """

        if self.conditional:
            assert (y is not None)
        elif not self.conditional:
            assert (y is None)
        calpha = self.scheduler.calpha(t)  # [nbatch]
        beta = self.scheduler.beta(t)  # [nbatch]
        if self.loss_scaling == "default":
            alpha = 1 - beta  # [nbatch]
            lambd = beta**2/(2*beta*alpha*(1-calpha))  # [nbatch]
            lambd = broadcast_from_below(lambd, x)
        elif self.loss_scaling == "constant":
            lambd = 1.0
        else:  # default to constant
            lambd = 1.0
        # [nbatch, *[shape]]
        calpha = broadcast_from_below(calpha, x)
        # [nbatch, *[shape]]
        noise = torch.randn_like(x)  # [nbatch, shape]

        # [nbatch, shape]
        x_noised = torch.sqrt(calpha)*x + torch.sqrt(1-calpha)*noise

        # [nbatch, shape]
        if self.conditional:
            score = self.model(x_noised, t, y)
        elif not self.conditional:
            score = self.model(x_noised, t)
        loss = (lambd*self.loss_metric(score, noise)).mean()
        return self.loss_scale_factor*loss

    def training_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.scheduler.sample(x.shape[0])
        loss = self.loss_fn(x, t, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.scheduler.sample(x.shape[0])
        loss = self.loss_fn(x, t, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("valid_loss", loss)  # For compatibility
        return loss

    def test_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.scheduler.sample(x.shape[0])
        loss = self.loss_fn(x, t, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        else:
            lr_scheduler_config = {"scheduler": self.lr_scheduler,
                                   "interval": self.lr_scheduler_interval}
            return [self.optimizer], [lr_scheduler_config]

    def sample(self, y=None, nsamples=1, device="cpu"):
        return self.sampler.sample(y=y, nsamples=nsamples, device=device)

    def sample_backward(self, x, y=None):
        return self.sampler.backward(x, y=y)

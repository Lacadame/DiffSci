import torch
import lightning

from diffsci.torchutils import broadcast_from_below

from diffsci.models import EulerMaruyamaSampler


class SDEModule(lightning.LightningModule):
    def __init__(self, model, scheduler,
                 conditional=True,
                 loss_type="mse",
                 loss_scale_factor=1.0):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.conditional = conditional
        self.loss_type = loss_type
        self.loss_scale_factor = loss_scale_factor
        self.set_optimizer_and_scheduler()
        self.set_sampler()
        self.set_loss(loss_type)

    def set_optimizer_and_scheduler(self,
                                    optimizer=None,
                                    scheduler=None,
                                    scheduler_interval="epoch"):
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
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                           lr=1e-3,
                                           betas=(0.9, 0.999),
                                           weight_decay=1e-4)
        self.lr_scheduler = (torch.optim.lr_scheduler.
                             CosineAnnealingWarmRestarts(self.optimizer,
                                                         T_0=10))
        self.lr_scheduler_interval = scheduler_interval

    def set_sampler(self, sampler_cls=EulerMaruyamaSampler,
                    shape=None,
                    *args, **kwargs):
        self.sampler = EulerMaruyamaSampler(self.model, self.scheduler,
                                            shape=shape,
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
        x : torch.Tensor of shape [B, [shape]], the original noise
        t : torch.Tensor of shape [B]
        y : None or torch.Tensor of shape [B, [yshape]] the conditional data,
        depending on whether we are dealing with a conditional or
        unconditional model
        """

        if self.conditional:
            assert (y is not None)
        elif not self.conditional:
            assert (y is None)
        mean = self.scheduler.mean(t, x)  # [nbatch, [shape]]]
        std = self.scheduler.std(t)  # [nbatch]
        std = broadcast_from_below(std, mean)  # [nbatch, [shape]]
        noise = torch.randn_like(x)  # [nbatch, [shape]]
        x_noised = mean + std*noise  # [nbatch, [shape]]
        if self.conditional:
            score = self.model(x_noised, t, y)  # [nbatch, [shape]]
        elif not self.conditional:
            score = self.model(x_noised, t)  # [nbatch, [shape]]
        loss = (1/std*self.loss_metric(std*score, -noise)).mean()
        return self.loss_scale_factor*loss

    def training_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.scheduler.sample_time(x.shape[0])
        loss = self.loss_fn(x, t, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.scheduler.sample_time(x.shape[0])
        loss = self.loss_fn(x, t, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.conditional:
            x, y = batch
        else:
            x = batch
            y = None
        t = self.scheduler.sample_time(x.shape[0])
        loss = self.loss_fn(x, t, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr_scheduler_config = {"scheduler": self.lr_scheduler,
                               "interval": self.lr_scheduler_interval}
        return [self.optimizer], [lr_scheduler_config]

    def sample(self, y=None, nsamples=1, device=None, nsteps=500):
        return self.sampler.sample(y=y, nsamples=nsamples,
                                   device=device,
                                   nsteps=nsteps)
    
    def forward(self, x, y=None, nsteps=500):
        return self.sampler.forward(x, y=y, nsteps=nsteps)

import math

import torch
import lightning
from torch import Tensor
from jaxtyping import Float


class VAEModuleConfig(object):
    """Configuration class for VAE module.

    This class holds parameters that control the VAE training behavior,
    including loss weights, variance initialization, and reduction methods.

    Attributes:
        kl_weight: Weight for the KL divergence term in the VAE loss.
        nll_weight: Weight for the negative log-likelihood term in the VAE loss.
        logvar_init: Initial value for the log variance parameter.
        trainable_logvar: Whether the log variance is a trainable parameter.
        reduce_mean: If True, reduce losses by mean; otherwise, sum and divide by batch size.
    """
    def __init__(self,
                 kl_weight: float = 1e-3,
                 nll_weight: float = 1.0,
                 logvar_init: float = 0.0,
                 trainable_logvar: bool = True,
                 reduce_mean: bool = False):
        self.kl_weight = kl_weight
        self.nll_weight = nll_weight
        self.logvar_init = logvar_init
        self.trainable_logvar = trainable_logvar
        self.reduce_mean = reduce_mean


class VAELoss(torch.nn.Module):
    def __init__(self, config: VAEModuleConfig):
        super().__init__()
        self.config = config
        if config.trainable_logvar:
            self.logvar = torch.nn.Parameter(torch.ones(size=(1,)) * config.logvar_init)
        else:
            self.register_buffer("logvar", torch.ones(size=(1,)) * config.logvar_init)

    def forward(self,
                x: Float[Tensor, "batch channels *shape"],  # noqa: F821, F722
                x_recon: Float[Tensor, "batch channels *shape"],  # noqa: F821, F722
                zdistrib: "DiagonalGaussianDistribution"):
        reduce_mean = self.config.reduce_mean
        nll_loss = ((x.contiguous() - x_recon.contiguous())**2)/torch.exp(self.logvar) + self.logvar
        kl_loss = zdistrib.kl(reduce_mean=reduce_mean)
        nsamples = x.shape[0]
        if reduce_mean:
            nll_loss = torch.mean(nll_loss)
        else:
            nll_loss = torch.sum(nll_loss) / nsamples  # Only the batch dimension is mean-reduced
        kl_loss = torch.sum(kl_loss) / nsamples
        loss = nll_loss + self.config.kl_weight * kl_loss
        logs = {
            "nll_loss": nll_loss.item(),
            "kl_loss": kl_loss.item(),
            "logvar": self.logvar.item(),
        }
        return loss, logs


class VAEModule(lightning.LightningModule):
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 config: VAEModuleConfig,
                 conditional: bool = False,
                 verbose: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.conditional = conditional
        self.loss_module = VAELoss(config)
        self.set_optimizer_and_scheduler()
        self.verbose = verbose

    def encode(self, x: Float[Tensor, "batch channels *shape"],  # noqa: F821, F722
               y: Float[Tensor, "batch *yshape"] | None = None,  # noqa: F821, F722
               sample: bool = True):
        z = self.encoder(x, y) if self.conditional else self.encoder(x)
        zdistrib = DiagonalGaussianDistribution(z)
        if sample:
            zsample = zdistrib.sample()
        else:
            zsample = zdistrib.mode()
        return {'zdistrib': zdistrib, 'zsample': zsample}

    def decode(self, z: Float[Tensor, "batch zdim *shape"],  # noqa: F821, F722
               y: Float[Tensor, "batch *yshape"] | None = None):  # noqa: F821, F722
        x_recon = self.decoder(z, y) if self.conditional else self.decoder(z)
        return x_recon

    def loss_fn(self, batch):
        x = batch['x']
        y = batch.get('y', None)
        encoder_outputs = self.encode(x, y)
        x_recon = self.decode(encoder_outputs['zsample'], y)
        loss, logs = self.loss_module(x, x_recon, encoder_outputs['zdistrib'])
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.loss_fn(batch)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        if self.verbose:
            # append train_ to the keys of logs
            logs = {f'train_{k}': v for k, v in logs.items()}
            self.log_dict(logs, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.loss_fn(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        if self.verbose:
            # append val_ to the keys of logs
            logs = {f'val_{k}': v for k, v in logs.items()}
            self.log_dict(logs, prog_bar=True, sync_dist=True)
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

    def configure_optimizers(self):
        if self.lr_scheduler is not None:
            lr_scheduler_config = {"scheduler": self.lr_scheduler,
                                   "interval": self.lr_scheduler_interval}
            return [self.optimizer], [lr_scheduler_config]
        else:  # Just fo backward compatibility for some examples
            return self.optimizer


class DiagonalGaussianDistribution(torch.nn.Module):
    def __init__(self,
                 mean_and_logvar: Float[Tensor, "batch (2*zdim) *shape"],  # noqa: F821
                 low_clamp: float = -30.0,
                 high_clamp: float = 20.0):
        super().__init__()
        mean, logvar = torch.chunk(mean_and_logvar, 2, dim=1)
        self.mean = mean
        self.logvar = torch.clamp(logvar, low_clamp, high_clamp)

    def sample(self):
        x = self.mean + self.std * (
            torch.randn(self.mean.shape).to(self.mean)
        )
        return x

    def kl(self, other: "DiagonalGaussianDistribution | None" = None, reduce_mean: bool = False):
        dims = list(range(1, len(self.mean.shape)))
        reduce_operator = torch.mean if reduce_mean else torch.sum
        if other is None:
            result = 0.5 * reduce_operator((torch.pow(self.mean, 2)
                                            + self.var - 1.0 - self.logvar),
                                           dim=dims)
        else:  # Other is the unit Gaussian
            result = 0.5 * reduce_operator(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var - 1.0 - self.logvar + other.logvar,
                dim=dims)
        return result

    def nll(self, sample: Float[Tensor, "batch zdim *shape"], reduce_mean: bool = False):  # noqa: F821, F722
        logtwopi = math.log(2.0 * math.pi)
        # Sum over all dimensions except batch dimension
        reduce_operator = torch.mean if reduce_mean else torch.sum
        dims = list(range(1, len(sample.shape)))
        result = 0.5 * reduce_operator(
            logtwopi + self.logvar + torch.pow(sample-self.mean, 2) / self.var,
            dim=dims)
        return result

    def mode(self):
        return self.mean

    @property
    def std(self):
        return torch.exp(0.5 * self.logvar)

    @property
    def var(self):
        return torch.exp(self.logvar)

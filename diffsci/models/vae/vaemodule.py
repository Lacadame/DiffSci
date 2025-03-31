import math

import torch
import lightning
from torch import Tensor
from jaxtyping import Float


class VAEModuleConfig(torch.nn.Module):
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
                 reduce_mean: bool = False,
                 teacher_encdec: torch.nn.Module | None = None,
                 distillation_alpha: float = 0.5,
                 latent_matching_type: str = "kl"):
        super().__init__()
        self.kl_weight = kl_weight
        self.nll_weight = nll_weight  # Unused for now
        self.logvar_init = logvar_init
        self.trainable_logvar = trainable_logvar
        self.reduce_mean = reduce_mean
        self.teacher_encdec = teacher_encdec
        self.distillation_alpha = distillation_alpha
        self.latent_matching_type = latent_matching_type

        assert self.latent_matching_type in ["kl", "mse", "modhell", "wasserstein"], \
            "latent_matching_type must be either 'kl', 'mse', 'modhell', or 'wasserstein'"
        if self.has_distillation:
            assert hasattr(self.teacher_encdec, "encoder") and hasattr(self.teacher_encdec, "decoder"), \
                "teacher_encdec must have encoder and decoder attributes"

    @property
    def has_distillation(self):
        return self.teacher_encdec is not None


class VAELoss(torch.nn.Module):
    def __init__(self, config: VAEModuleConfig):
        super().__init__()
        self.config = config
        if config.trainable_logvar:
            self.logvar = torch.nn.Parameter(torch.ones(size=(1,)) * config.logvar_init)
        else:
            self.register_buffer("logvar", torch.ones(size=(1,)) * config.logvar_init)
        self.freeze_teacher()

    def freeze_teacher(self):
        if self.config.teacher_encdec is not None:
            for param in self.config.teacher_encdec.parameters():
                param.requires_grad = False

    def forward(self,
                x: Float[Tensor, "batch channels *shape"],  # noqa: F821, F722
                x_recon: Float[Tensor, "batch channels *shape"],  # noqa: F821, F722
                zdistrib: "DiagonalGaussianDistribution"):
        reduce_mean = self.config.reduce_mean
        nsamples = x.shape[0]
        if self.config.has_distillation and self.config.distillation_alpha == 1.0:
            # No need to compute this loss, everything will be distillation
            nll_loss = torch.tensor(0.0).to(x)
            kl_loss = torch.tensor(0.0).to(x)
            main_loss = torch.tensor(0.0).to(x)
            loss = torch.tensor(0.0).to(x)
        else:
            nll_loss = ((x.contiguous() - x_recon.contiguous())**2)/torch.exp(self.logvar) + self.logvar  # [b, c, ...]
            kl_loss = zdistrib.kl(reduce_mean=reduce_mean)  # [b, ...]
            if reduce_mean:
                nll_loss = torch.mean(nll_loss)  # []
            else:
                nll_loss = torch.sum(nll_loss) / nsamples  # Only the batch dimension is mean-reduced,  []
            kl_loss = torch.sum(kl_loss) / nsamples  # []
            main_loss = nll_loss + self.config.kl_weight * kl_loss  # []
            loss = main_loss.clone()
        if self.config.has_distillation:
            teacher_z = self.config.teacher_encdec.encoder(x)  # [b, 2*zdim, ...]
            teacher_zdistrib = DiagonalGaussianDistribution(teacher_z)
            teacher_zsample = teacher_zdistrib.sample()
            teacher_x_recon = self.config.teacher_encdec.decoder(teacher_zsample)  # [b, c, ...]
            latent_space_matching_loss = self.calculate_latent_space_matching_loss(
                zdistrib, teacher_zdistrib, reduce_mean, nsamples)
            output_matching_loss = torch.nn.functional.mse_loss(x_recon, teacher_x_recon, reduction='none')
            if reduce_mean: 
                output_matching_loss = torch.mean(output_matching_loss)  # []
            else:
                output_matching_loss = torch.sum(output_matching_loss) / nsamples  # []
            loss = ((1 - self.config.distillation_alpha) * loss +
                    self.config.distillation_alpha * (latent_space_matching_loss + output_matching_loss))
        logs = {
            "nll_loss": nll_loss.item(),
            "kl_loss": kl_loss.item(),
            "main_loss": main_loss.item(),
            "logvar": self.logvar.item(),
        }
        if self.config.has_distillation:
            logs["latent_space_matching_loss"] = latent_space_matching_loss.item()
            logs["output_matching_loss"] = output_matching_loss.item()
        return loss, logs

    def calculate_latent_space_matching_loss(self, zdistrib, teacher_zdistrib,
                                             reduce_mean, nsamples):
        if self.config.latent_matching_type == "kl":
            latent_space_matching_loss = zdistrib.kl(teacher_zdistrib, reduce_mean=reduce_mean)
            latent_space_matching_loss = torch.sum(latent_space_matching_loss) / nsamples  # []
        elif self.config.latent_matching_type == "modhell":
            latent_space_matching_loss = zdistrib.modified_hellinger(teacher_zdistrib, reduce_mean=reduce_mean)
            latent_space_matching_loss = torch.sum(latent_space_matching_loss) / nsamples  # []
        elif self.config.latent_matching_type in ["mse", "wasserstein"]:
            latent_space_matching_loss = zdistrib.wasserstein(teacher_zdistrib, reduce_mean=reduce_mean)
            latent_space_matching_loss = torch.sum(latent_space_matching_loss) / nsamples  # []
        else:
            raise ValueError(f"Latent matching type {self.config.latent_matching_type} not supported")
        return latent_space_matching_loss


class VAEModule(lightning.LightningModule):
    def __init__(self,
                 encdec: torch.nn.Module,
                 config: VAEModuleConfig,
                 conditional: bool = False,
                 verbose: bool = False):
        super().__init__()
        self.encdec = encdec
        # Assert whether encdec has "encoder" and "decoder" attributes
        assert hasattr(self.encdec, "encoder") and hasattr(self.encdec, "decoder"), \
            "encdec must have encoder and decoder attributes"
        self.config = config
        self.conditional = conditional
        self.loss_module = VAELoss(config)
        self.set_optimizer_and_scheduler()
        self.verbose = verbose

    @property
    def encoder(self):
        return self.encdec.encoder

    @property
    def decoder(self):
        return self.encdec.decoder

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

    def decode(self, zsample: Float[Tensor, "batch zdim *shape"],  # noqa: F821, F722
               y: Float[Tensor, "batch *yshape"] | None = None):  # noqa: F821, F722
        x_recon = self.decoder(zsample, y) if self.conditional else self.decoder(zsample)
        return x_recon

    def loss_fn(self, batch):
        x, y = self.select_batch(batch)
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

    def select_batch(self, batch):
        if isinstance(batch, dict):
            x = batch['x']
            y = batch.get('y', None)
        else:
            if self.conditional:
                x, y = batch
            else:
                x = batch
                y = None
            return x, y


class DiagonalGaussianDistribution(torch.nn.Module):
    def __init__(self,
                 mean_and_logvar: Float[Tensor, "batch (2*zdim) *shape"],  # noqa: F821
                 low_clamp: float = -30.0,
                 high_clamp: float = 20.0):
        super().__init__()
        mean, logvar = torch.chunk(mean_and_logvar, 2, dim=1)
        self.mean = mean
        self.logvar = torch.clamp(logvar, low_clamp, high_clamp)

    @property
    def mean_and_logvar(self):
        return torch.cat((self.mean, self.logvar), dim=1)

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

    def modified_hellinger(self, other: "DiagonalGaussianDistribution | None" = None, reduce_mean: bool = False):
        # It is another distance operator, but the formula here is (in latex):
        # 1/4 \sum_{i=1}^d (2 \log \frac{\sigma_{i, 1}^2 + \sigma_{i, 2}^2}{2 \sigma_{i, 1} \sigma_{i, 2}} \\
        #                   + \frac{(\mu_{i, 1} - \mu_{i, 2})^2}{\sigma_{i, 1}^2 + \sigma_{i, 2}^2})
        dims = list(range(1, len(self.mean.shape)))
        reduce_operator = torch.mean if reduce_mean else torch.sum

        if other is None:
            # Use mu_{i, 2}=0 and \sigma_{i, 2}=1
            other_mean = torch.zeros_like(self.mean)
            other_var = torch.ones_like(self.var)
        else:
            other_mean = other.mean
            other_var = other.var

        sum_var = self.var + other_var
        log_term = 2 * torch.log(sum_var / (2 * self.std * torch.sqrt(other_var)))
        mean_term = torch.pow(self.mean - other_mean, 2) / sum_var

        result = 0.25 * reduce_operator(log_term + mean_term, dim=dims)
        return result

    def wasserstein(self, other: "DiagonalGaussianDistribution | None" = None, reduce_mean: bool = False):
        dims = list(range(1, len(self.mean.shape)))
        reduce_operator = torch.mean if reduce_mean else torch.sum
        if other is None:
            other_mean = torch.zeros_like(self.mean)
            other_var = torch.ones_like(self.var)
        else:
            other_mean = other.mean
            other_std = other.std

        mean_term = torch.pow(self.mean - other_mean, 2)
        std_term = torch.pow(self.std - other_std, 2)
        result = reduce_operator(mean_term + std_term, dim=dims)
        return result

    def mode(self):
        return self.mean

    @property
    def std(self):
        return torch.exp(0.5 * self.logvar)

    @property
    def var(self):
        return torch.exp(self.logvar)

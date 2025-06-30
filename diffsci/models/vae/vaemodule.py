import math
from typing import Literal

import torch
import lightning
from torch import Tensor
from jaxtyping import Float


LatentMatchingType = Literal["kl", "mse", "modhell", "wasserstein"]
TeachingMode = Literal["both", "encoder", "decoder"]


class VAEModuleConfig:
    """Configuration class for VAE module.

    This class holds parameters that control the VAE training behavior,
    including loss weights, variance initialization, and reduction methods.

    Attributes:
        kl_weight: Weight for the KL divergence term in the VAE loss.
        nll_weight: Weight for the negative log-likelihood term in the VAE loss.
        logvar_init: Initial value for the log variance parameter.
        trainable_logvar: Whether the log variance is a trainable parameter.
        reduce_mean: If True, reduce losses by mean; otherwise, sum and divide by batch size.
        adversarial_weight: Weight for the adversarial loss term.
    """
    def __init__(self,
                 kl_weight: float = 1e-3,
                 nll_weight: float = 1.0,
                 logvar_init: float = 0.0,
                 trainable_logvar: bool = True,
                 reduce_mean: bool = False,
                 teacher_encdec: torch.nn.Module | None = None,
                 teaching_mode: TeachingMode = "both",
                 distillation_alpha: float = 0.5,
                 latent_matching_type: LatentMatchingType = "kl",
                 adversarial_weight: float = 0.01):
        self.kl_weight = kl_weight
        self.nll_weight = nll_weight  # Unused for now
        self.logvar_init = logvar_init
        self.trainable_logvar = trainable_logvar
        self.reduce_mean = reduce_mean
        self.teacher_encdec = teacher_encdec
        self.teaching_mode = teaching_mode
        self.distillation_alpha = distillation_alpha
        self.latent_matching_type = latent_matching_type
        self.adversarial_weight = adversarial_weight

        assert self.latent_matching_type in ["kl", "mse", "modhell", "wasserstein"], \
            "latent_matching_type must be either 'kl', 'mse', 'modhell', or 'wasserstein'"
        assert self.teaching_mode in ["both", "encoder", "decoder"], \
            "teaching_mode must be either 'both', 'encoder', or 'decoder'"
        if self.has_distillation:
            assert hasattr(self.teacher_encdec, "encoder") and hasattr(self.teacher_encdec, "decoder"), \
                "teacher_encdec must have encoder and decoder attributes"
            assert self.distillation_alpha > 0.0 and self.distillation_alpha <= 1.0, \
                "distillation_alpha must be in interval (0.0, 1.0]"

    @property
    def has_distillation(self):
        return self.teacher_encdec is not None

    @property
    def distillation_training_only(self):
        return self.has_distillation and self.distillation_alpha == 1.0


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
                vae_module: 'VAEModule',
                y: Float[Tensor, "batch *yshape"] | None = None,  # noqa: F821, F722
                return_intermediates: bool = False):  # New parameter

        if self.config.teacher_encdec is not None:
            self.config.teacher_encdec.to(x)  # Hack because teacher_encdec is not seen as a module

        if self.config.distillation_training_only:  # We will not compute default encoder/decoder loss
            loss, logs = self.distillation_loss(x, vae_module, y, None, None)
            if return_intermediates:
                # For distillation-only, we still need to compute these
                encoder_outputs = vae_module.encode(x, y)
                x_recon = vae_module.decode(encoder_outputs['zsample'], y)
                return loss, logs, encoder_outputs, x_recon
            return loss, logs

        encoder_outputs = vae_module.encode(x, y)
        x_recon = vae_module.decode(encoder_outputs['zsample'], y)
        zdistrib = encoder_outputs['zdistrib']

        reduce_mean = self.config.reduce_mean
        nsamples = x.shape[0]

        nll_loss = ((x.contiguous() - x_recon.contiguous()) ** 2) / torch.exp(self.logvar) + self.logvar  # [b, c, ...]
        kl_loss = zdistrib.kl(reduce_mean=reduce_mean)  # [b, ...]
        if reduce_mean:
            nll_loss = torch.mean(nll_loss)  # []
        else:
            nll_loss = torch.sum(nll_loss) / nsamples  # Only the batch dimension is mean-reduced,  []
        kl_loss = torch.sum(kl_loss) / nsamples  # []

        main_loss = nll_loss + self.config.kl_weight * kl_loss  # []
        loss = main_loss.clone()

        logs = {
            "nll_loss": nll_loss.item(),
            "kl_loss": kl_loss.item(),
            "main_loss": main_loss.item(),
            "logvar": self.logvar.item(),
        }

        if self.config.has_distillation:
            distillation_loss, distillation_logs = self.distillation_loss(
                x, vae_module, y, zdistrib, x_recon)
            loss = (1 - self.config.distillation_alpha) * loss + \
                self.config.distillation_alpha * distillation_loss
            logs.update(distillation_logs)

        if return_intermediates:
            return loss, logs, encoder_outputs, x_recon
        return loss, logs

    def distillation_loss(self, x, vae_module, y, zdistrib, x_recon):
        nsamples = x.shape[0]  # Duplicated but it is fine, extremely cheap operation
        reduce_mean = self.config.reduce_mean

        if zdistrib is None:
            if self.config.teaching_mode == "decoder":
                zdistrib = DiagonalGaussianDistribution(
                    self.config.teacher_encdec.encoder(x))
                zsample = zdistrib.sample()
            else:
                zdistrib = vae_module.encode(x, y)['zdistrib']
                zsample = zdistrib.sample()

        if x_recon is None:
            if self.config.teaching_mode == "encoder":
                x_recon = 0.0  # It will be ignored
            else:
                x_recon = vae_module.decode(zsample, y)

        if self.config.teaching_mode == "decoder":
            latent_space_matching_loss = torch.tensor(0.0).to(x)
            teacher_zsample = zsample
        else:
            teacher_z = self.config.teacher_encdec.encoder(x)  # [b, 2*zdim, ...]
            teacher_zdistrib = DiagonalGaussianDistribution(teacher_z)
            teacher_zsample = teacher_zdistrib.sample()
            latent_space_matching_loss = self.calculate_latent_space_matching_loss(
                zdistrib, teacher_zdistrib, reduce_mean, nsamples)

        if self.config.teaching_mode == "encoder":
            output_matching_loss = torch.tensor(0.0).to(x)
        else:
            teacher_x_recon = self.config.teacher_encdec.decoder(teacher_zsample)  # [b, c, ...]
            output_matching_loss = torch.nn.functional.mse_loss(x_recon, teacher_x_recon, reduction='none')
            if reduce_mean:
                output_matching_loss = torch.mean(output_matching_loss)  # []
            else:
                output_matching_loss = torch.sum(output_matching_loss) / nsamples  # []

        loss = latent_space_matching_loss + output_matching_loss
        logs = {
            "latent_space_matching_loss": latent_space_matching_loss.item(),
            "output_matching_loss": output_matching_loss.item(),
        }
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

    @property
    def distillation_training_only(self):
        return self.config.has_distillation and self.config.distillation_alpha == 1.0


class VAEModule(lightning.LightningModule):
    def __init__(self,
                 encdec: torch.nn.Module,
                 config: VAEModuleConfig,
                 discriminator: torch.nn.Module | None = None,
                 conditional: bool = False,
                 verbose: bool = False):
        super().__init__()
        self.encdec = encdec
        assert hasattr(self.encdec, "encoder") and hasattr(self.encdec, "decoder"), \
            "encdec must have encoder and decoder attributes"

        self.config = config
        self.conditional = conditional
        self.loss_module = VAELoss(config)

        # Adversarial components (optional)
        self.discriminator = discriminator

        # Training mode tracking
        self.is_adversarial = discriminator is not None
        if self.is_adversarial:
            self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
            self.automatic_optimization = False  # Manual optimization for GAN training

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

        if self.is_adversarial:
            # Get VAE loss + intermediates for adversarial training
            vae_loss, vae_logs, encoder_outputs, x_recon = self.loss_module(
                x, self, y, return_intermediates=True)
            return vae_loss, vae_logs, encoder_outputs, x_recon
        else:
            # Standard VAE loss
            loss, logs = self.loss_module(x, self, y)
            return loss, logs

    def generator_loss_fn(self, batch):
        """Combined VAE + adversarial loss for generator"""
        x, y = self.select_batch(batch)
        vae_loss, vae_logs, encoder_outputs, x_recon = self.loss_module(
            x, self, y, return_intermediates=True)

        # Adversarial loss for generator
        if self.conditional and y is not None:
            fake_pred = self.discriminator(x_recon, y)
        else:
            fake_pred = self.discriminator(x_recon)

        gen_adversarial_loss = self.adversarial_loss(
            fake_pred, torch.ones_like(fake_pred))

        total_loss = vae_loss + self.config.adversarial_weight * gen_adversarial_loss

        logs = vae_logs.copy()
        logs.update({
            'gen_adversarial_loss': gen_adversarial_loss.item(),
            'vae_loss': vae_loss.item(),
            'total_gen_loss': total_loss.item()
        })

        return total_loss, logs

    def discriminator_loss_fn(self, batch):
        """Discriminator loss"""
        x, y = self.select_batch(batch)

        # Generate fake samples (no gradients to generator)
        with torch.no_grad():
            encoder_outputs = self.encode(x, y)
            x_fake = self.decode(encoder_outputs['zsample'], y)

        # Discriminator predictions
        if self.conditional and y is not None:
            real_pred = self.discriminator(x, y)
            fake_pred = self.discriminator(x_fake, y)
        else:
            real_pred = self.discriminator(x)
            fake_pred = self.discriminator(x_fake)

        # Discriminator loss
        real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        discriminator_loss = (real_loss + fake_loss) / 2

        # Accuracy monitoring
        real_acc = (real_pred > 0).float().mean()
        fake_acc = (fake_pred < 0).float().mean()
        d_acc = (real_acc + fake_acc) / 2

        logs = {
            'discriminator_loss': discriminator_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'd_accuracy': d_acc.item(),
            'real_accuracy': real_acc.item(),
            'fake_accuracy': fake_acc.item()
        }

        return discriminator_loss, logs

    def training_step(self, batch, batch_idx):
        if not self.is_adversarial:
            # Standard VAE training
            loss, logs = self.loss_fn(batch)
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
            if self.verbose:
                logs = {f'train_{k}': v for k, v in logs.items()}
                self.log_dict(logs, prog_bar=True, sync_dist=True)
            return loss
        else:
            # Adversarial training with manual optimization
            gen_opt, disc_opt = self.optimizers()

            # Train Generator
            gen_loss, gen_logs = self.generator_loss_fn(batch)
            gen_opt.zero_grad()
            self.manual_backward(gen_loss)
            gen_opt.step()

            # Train Discriminator (with optional balancing)
            disc_loss, disc_logs = self.discriminator_loss_fn(batch)
            d_acc = disc_logs['d_accuracy']

            if d_acc < 0.85:  # Only train discriminator if not too strong
                disc_opt.zero_grad()
                self.manual_backward(disc_loss)
                disc_opt.step()

            # Logging
            self.log('train_gen_loss', gen_loss, prog_bar=True, sync_dist=True)
            self.log('train_disc_loss', disc_loss, prog_bar=True, sync_dist=True)
            self.log('d_accuracy', d_acc, prog_bar=True, sync_dist=True)

            if self.verbose:
                all_logs = {f'train_{k}': v for k, v in {**gen_logs, **disc_logs}.items()}
                self.log_dict(all_logs, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        if not self.is_adversarial:
            # Standard VAE validation
            loss, logs = self.loss_fn(batch)
            self.log('val_loss', loss, prog_bar=True, sync_dist=True)
            if self.verbose:
                logs = {f'val_{k}': v for k, v in logs.items()}
                self.log_dict(logs, prog_bar=True, sync_dist=True)
            return loss
        else:
            # VAE-GAN validation
            gen_loss, gen_logs = self.generator_loss_fn(batch)
            disc_loss, disc_logs = self.discriminator_loss_fn(batch)

            self.log('val_gen_loss', gen_loss, prog_bar=True, sync_dist=True)
            self.log('val_disc_loss', disc_loss, prog_bar=True, sync_dist=True)

            if self.verbose:
                all_logs = {f'val_{k}': v for k, v in {**gen_logs, **disc_logs}.items()}
                self.log_dict(all_logs, sync_dist=True)

    def configure_optimizers(self):
        if not self.is_adversarial:
            if hasattr(self, 'optimizer'):
                if self.lr_scheduler is not None:
                    return [self.optimizer], [{
                        "scheduler": self.lr_scheduler,
                        "interval": self.lr_scheduler_interval
                    }]
                return self.optimizer
            else:
                return torch.optim.AdamW(
                    self.parameters(),
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    weight_decay=1e-4
                )
        else:
            optimizers = [self.gen_optimizer, self.disc_optimizer]
            schedulers = []
            if self.gen_scheduler is not None:
                schedulers.append({
                    "scheduler": self.gen_scheduler,
                    "interval": "step"
                })
            if self.disc_scheduler is not None:
                schedulers.append({
                    "scheduler": self.disc_scheduler,
                    "interval": "step"
                })
            return optimizers, schedulers if schedulers else None

    def set_optimizer_and_scheduler(self,
                                    optimizer=None,
                                    scheduler=None,
                                    scheduler_interval="step",
                                    adversarial_optimizers=None,
                                    adversarial_schedulers=None):
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
        adversarial_optimizers : None | tuple[torch.optim.Optimizer, torch.optim.Optimizer]
            if None and is_adversarial, use default AdamW optimizers for generator and
            discriminator with lr=1e-4 and 2e-4 respectively
        adversarial_scheduler : None | tuple[torch.optim.lr_scheduler._LRScheduler,
            torch.optim.lr_scheduler._LRScheduler]
            if None, no scheduler is used for adversarial training
        """
        if not self.is_adversarial:
            if optimizer is not None:
                self.optimizer = optimizer
            else:
                self.optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    weight_decay=1e-4
                )
            if scheduler is not None:
                self.lr_scheduler = scheduler
            else:
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda step: 1.0 + 0 * step
                )  # Neutral scheduler
            self.lr_scheduler_interval = scheduler_interval
        else:
            if adversarial_optimizers is not None:
                self.gen_optimizer, self.disc_optimizer = adversarial_optimizers
            else:
                gen_params = list(self.encdec.parameters()) + list(self.loss_module.parameters())
                self.gen_optimizer = torch.optim.AdamW(
                    gen_params,
                    lr=1e-4,  # Default generator learning rate
                    betas=(0.5, 0.999)
                )
                self.disc_optimizer = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=2e-4,  # Default discriminator learning rate
                    betas=(0.5, 0.999)
                )

            if adversarial_schedulers is not None:
                self.gen_scheduler, self.disc_scheduler = adversarial_schedulers
            else:
                self.gen_scheduler = None
                self.disc_scheduler = None

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
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
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
            other_std = torch.ones_like(self.std)
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

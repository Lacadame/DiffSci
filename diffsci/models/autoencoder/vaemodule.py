import torch
import torch.nn as nn
import pytorch_lightning as pl

class VAEWithDiscriminator(pl.LightningModule):
    def __init__(self,
                 vae_model,
                 discriminator_model, 
                 in_channels=1,
                 input_shape=None,
                 latent_dim=20,
                 encoder_hidden_dims=None,
                 decoder_hidden_dims=None,
                 disc_hidden_dims=None,
                 input_dim=2,
                 vae_steps=1,
                 disc_steps=1,
                 lr=1e-4):
        
        super(VAEWithDiscriminator, self).__init__()
        self.save_hyperparameters()

        # Initialize VAE with encoder and decoder hidden dimensions
        self.vae = vae_model(
            in_channels=in_channels, 
            input_shape=input_shape, 
            encoder_hidden_dims=encoder_hidden_dims, 
            decoder_hidden_dims=decoder_hidden_dims,
            latent_dim=latent_dim, 
            input_dim=input_dim
        )
        
        # Initialize Discriminator
        self.discriminator = discriminator_model(
            in_channels=in_channels, 
            input_shape=input_shape, 
            hidden_dims=disc_hidden_dims, 
            input_dim=input_dim
        )
        
        self.automatic_optimization = False  # We manually optimize in training_step
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss for Discriminator
        self.rec_loss_fn = nn.MSELoss()  # Reconstruction loss for VAE

    def forward(self, x):
        reconstructed, mu, logvar = self.vae(x)
        return reconstructed, mu, logvar

    def training_step(self, batch, batch_idx):
        x, _ = batch
        optimizer_vae, optimizer_discriminator = self.optimizers()

        # === Train Discriminator ===
        self.vae.eval()
        self.discriminator.train()
        with torch.no_grad():
            reconstructed, _, _ = self.vae(x)
        
        for _ in range(self.hparams.disc_steps):
            real_output = self.discriminator(x)
            fake_output = self.discriminator(reconstructed.detach())

            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)

            d_loss_real = self.criterion(real_output, real_labels)
            d_loss_fake = self.criterion(fake_output, fake_labels)
            disc_loss = (d_loss_real + d_loss_fake) / 2  # Average over real and fake losses

            optimizer_discriminator.zero_grad()
            self.manual_backward(disc_loss)
            optimizer_discriminator.step()

        # === Train VAE ===
        self.vae.train()
        # Freeze Discriminator parameters during VAE training
        for param in self.discriminator.parameters():
            param.requires_grad = False
        
        for _ in range(self.hparams.vae_steps):
            reconstructed, mu, logvar = self.vae(x)

            # Reconstruction loss
            rec_loss = self.rec_loss_fn(reconstructed, x)

            # Regularization loss (Kullback-Leibler divergence)
            reg_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Adversarial loss
            fake_output_for_vae = self.discriminator(reconstructed)
            comp_labels = torch.ones_like(fake_output_for_vae)
            comp_loss = self.criterion(fake_output_for_vae, comp_labels)

            # Total VAE loss
            vae_loss = rec_loss + 1e-4 * reg_loss + 1e-3*comp_loss

            optimizer_vae.zero_grad()
            self.manual_backward(vae_loss)
            optimizer_vae.step()

        # Unfreeze Discriminator parameters
        for param in self.discriminator.parameters():
            param.requires_grad = True

        # === Logging ===
        # Log the last computed losses
        self.log('train_vae_loss', vae_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_disc_loss', disc_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_rec_loss', rec_loss.item(), on_step=False, on_epoch=True)
        self.log('train_reg_loss', reg_loss.item(), on_step=False, on_epoch=True)
        self.log('train_comp_loss', comp_loss.item(), on_step=False, on_epoch=True)
        self.log('train_d_loss_real', d_loss_real.item(), on_step=False, on_epoch=True)
        self.log('train_d_loss_fake', d_loss_fake.item(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=self.hparams.lr)
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)

        scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=5, gamma=0.5)
        scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=5, gamma=0.5)

        return [optimizer_vae, optimizer_discriminator], [scheduler_vae, scheduler_discriminator]
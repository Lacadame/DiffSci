import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 input_shape=None,  # Must be provided, e.g., (C, H, W) for 2D or (C, D, H, W) for 3D
                 encoder_hidden_dims=None, 
                 decoder_hidden_dims=None, 
                 latent_dim=20,
                 input_dim=2):  # 2 for 2D data, 3 for 3D data
        
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.input_shape = input_shape

        # Choose the appropriate convolutional layers based on input dimension
        if input_dim == 2:
            Conv = nn.Conv2d
            ConvTranspose = nn.ConvTranspose2d
            BatchNorm = nn.BatchNorm2d
        elif input_dim == 3:
            Conv = nn.Conv3d
            ConvTranspose = nn.ConvTranspose3d
            BatchNorm = nn.BatchNorm3d
        else:
            raise ValueError("input_dim must be 2 or 3.")

        # Default encoder hidden dimensions
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [32, 64, 128]

        # Build Encoder
        modules = []
        channels = in_channels
        for h_dim in encoder_hidden_dims:
            modules.append(
                nn.Sequential(
                    Conv(channels, out_channels=h_dim,
                         kernel_size=3, stride=2, padding=1),
                    nn.ReLU())
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Compute the shape after convolution layers to flatten
        if self.input_shape is None:
            raise ValueError("input_shape must be specified.")
        # Use dummy input with batch size 1
        dummy_input = torch.zeros(1, *self.input_shape)
        conv_out = self.encoder(dummy_input)
        self.conv_output_shape = conv_out.shape[1:]  # Exclude batch size
        self.flatten_dim = conv_out.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Default decoder hidden dimensions
        if decoder_hidden_dims is None:
            decoder_hidden_dims = encoder_hidden_dims[::-1]

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        modules = []
        channels = decoder_hidden_dims[0]
        num_decoder_layers = len(decoder_hidden_dims)
        for i in range(num_decoder_layers - 1):
            # Adjust output_padding based on layer index
            if i == 0:
                output_padding = 0
            else:
                output_padding = 1
            modules.append(
                nn.Sequential(
                    ConvTranspose(channels, decoder_hidden_dims[i + 1],
                                  kernel_size=3, stride=2, padding=1, output_padding=output_padding),
                    nn.ReLU())
            )
            channels = decoder_hidden_dims[i + 1]

        # Final layer to get back to original image size (28x28)
        modules.append(
            nn.Sequential(
                ConvTranspose(channels, out_channels=in_channels,
                              kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.conv_output_shape)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Discriminator(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 input_shape=None,  # Must be provided
                 hidden_dims=None, 
                 input_dim=2):  # 2 for 2D data, 3 for 3D data
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.input_shape = input_shape

        # Choose the appropriate convolutional layers
        if input_dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            Dropout = nn.Dropout2d
        elif input_dim == 3:
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            Dropout = nn.Dropout3d
        else:
            raise ValueError("input_dim must be 2 or 3.")

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        # Build Discriminator
        modules = []
        channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    Conv(channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    Dropout(0.25),
                    BatchNorm(h_dim)
                )
            )
            channels = h_dim

        self.model = nn.Sequential(*modules)

        # Compute the shape after convolution layers to flatten
        if self.input_shape is None:
            raise ValueError("input_shape must be specified.")
        dummy_input = torch.zeros(1, *self.input_shape)
        conv_out = self.model(dummy_input)
        self.flatten_dim = conv_out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        validity = self.classifier(x)
        return validity
import math

import torch
import einops

"""
Deprecated and marked for removal
"""


class GaussianFourierProjection(torch.nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        """
        Parameters
        ----------
        embed_dim : int
            The dimension of the embedding
        scale : float
            The scale of the gaussian distribution
        """
        super().__init__()
        self.register_buffer('W',
                             torch.randn(embed_dim//2)*scale)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (..., 1)

        Returns
        -------
        x_proj : torch.Tensor of shape (..., embed_dim)
        """
        x = x[..., None]  # (..., 1)
        x_proj = 2*math.pi*x*self.W  # (..., embed_dim//2)
        x_proj = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # (..., embed_dim)
        return x_proj


class ResnetTimeBlock(torch.nn.Module):
    def __init__(self, embed_channels, ouput_channels):
        """
        Parameters
        ----------
        embed_channels : int
            The number of channels in the embedding
        ouput_channels : int
            The number of channels in the output
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_channels, 4*embed_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(4*embed_channels, 4*embed_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(4*embed_channels, ouput_channels)
        )

    def forward(self, te):
        """
        Parameters
        ----------
        te : torch.Tensor of shape (nbatch, embed_channels)

        Returns
        -------
        torch.Tensor of shape (nbatch, output_channels, 1, 1, 1)
        """
        # te : (nbatch, embed_channels)
        # returns : (nbatch, output_channels, 1, 1, 1)
        return self.net(te).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class ResnetBlockB(torch.nn.Module):
    def __init__(self, input_channels,
                 time_embed_dim,
                 output_channels=None,
                 kernel_size=3,
                 dropout=0.0):
        """
        Parameters
        ----------
        input_channels : int
            The number of input channels
        time_embed_dim : int
            The dimension of the time embedding
        output_channels : int | None
            The number of output channels. If None,
            then output_channels = input_channels
        kernel_size : int
            Size of convolutional kernel.
        dropout : float
            The dropout value
        """
        super().__init__()
        kernel_size = kernel_size
        if output_channels is None:
            output_channels = input_channels
            self.has_residual_connection = True
        else:
            self.has_residual_connection = False
        self.act = torch.nn.SiLU()
        self.gnorm1 = torch.nn.GroupNorm(input_channels, input_channels)
        self.conv1 = torch.nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.gnorm2 = torch.nn.GroupNorm(output_channels, output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv2 = torch.nn.Conv3d(
            output_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.timeblock = ResnetTimeBlock(time_embed_dim, output_channels)

    def forward(self, x, te):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, D, H, W)
        te : torch.Tensor of shape (B, time_embed_dim)

        Returns
        -------
        torch.Tensor of shape (B, output_channels, D, H, W)
        """
        # x : (B, C_in, D, H, W)
        # te : (B, C_embed)
        y = self.conv1(self.act(self.gnorm1(x)))  # (B, C_out, D, H, W)
        yt = self.timeblock(te)
        y = y + yt  # (B, C_out, D, H, W)
        y = self.conv2(
            self.dropout(self.act(self.gnorm2(y)))
        )  # (B, C_out, D, H, W)
        if self.has_residual_connection:
            y = y + x
        return y


class DownSampler(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Parameters
        ----------
        input_channels : int
            The number of input channels
        output_channels : int
            The number of output channels
        """
        super().__init__()
        self.downsampler = torch.nn.MaxPool3d(2)
        self.conv = torch.nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding='same')

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, D, H, W)

        Returns
        -------
        torch.Tensor of shape (B, output_channels, D//2, H//2, W//2)
        """
        # x : (B, C_in, D, H, W)
        # returns : (B, C_out, D//2, H//2, W//2)
        return self.conv(self.downsampler(x))


class UpSampler(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Parameters
        ----------
        input_channels : int
            The number of input channels
        output_channels : int
            The number of output channels
        """
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=2)
        self.conv = torch.nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding='same')

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, D, H, W)

        Returns
        -------
        torch.Tensor of shape (B, output_channels, D*2, H*2, W*2)
        """
        # x : (B, C_in, D, H, W)
        # returns : (B, C_out, D*2, H*2, W*2)
        return self.conv(self.upsampler(x))


class ThreeDimensionalAttention(torch.nn.Module):
    def __init__(self, num_channels):
        """
        Parameters
        ----------
        num_channels : int
            The number of channels in the input
        """
        super().__init__()
        self.mhattn = torch.nn.MultiheadAttention(num_channels, num_heads=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, D, H, W)

        Returns
        -------
        torch.Tensor of shape (B, C, D, H, W)

        OBS: VARIABLES NAMES IN THIS FUNCTION ARE NOT GOOD!!! CHANGE IT IN THE
        FUTURE!!!
        """
        d, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
        x_r = einops.rearrange(x, 'b c d h w -> b (d h w) c')
        x_r, _ = self.mhattn(x_r, x_r, x_r)
        x_r = einops.rearrange(x_r, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        return x_r


class PUNet3DB(torch.nn.Module):
    def __init__(self, model_channels,
                 channels=1,
                 conditional_channels=0,
                 kernel_size=3,
                 dropout=0.0):
        """
        Parameters
        ----------
        model_channels : int
            The number of channels in the model
        channels : int
            The number of input channes
        conditional_channels : int
            The number of conditional channels in the model.
            If we are doing unconditioned, then this equals to zero.
        kernel_size : int
            The size of the convolutional kernel.
        dropout : float
            The dropout value.
        """
        super().__init__()
        in_channels = channels+conditional_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.model_channels = model_channels
        self.convin = torch.nn.Conv3d(in_channels=in_channels,
                                      out_channels=model_channels,
                                      kernel_size=kernel_size,
                                      padding='same')
        self.time_embed = GaussianFourierProjection(model_channels)
        self.resnet1a = self.resnet_block_fn(1)
        self.resnet1b = self.resnet_block_fn(1)
        self.downsampler12 = DownSampler(model_channels, 2*model_channels)
        self.resnet2a = self.resnet_block_fn(2)
        self.resnet2b = self.resnet_block_fn(2)
        self.downsampler23 = DownSampler(2*model_channels, 4*model_channels)
        self.resnet3a = self.resnet_block_fn(4)
        self.resnet3b = self.resnet_block_fn(4)
        self.attn = ThreeDimensionalAttention(4*model_channels)
        self.resnetattna = self.resnet_block_fn(4)
        self.resnetattnb = self.resnet_block_fn(4)
        self.resnet3c = self.resnet_block_fn(4)
        self.resnet3d = self.resnet_block_fn(4)
        self.upsampler34 = UpSampler(4*model_channels, 2*model_channels)
        self.resnet4a = self.resnet_block_fn(2)
        self.resnet4b = self.resnet_block_fn(2)
        self.upsampler45 = UpSampler(2*model_channels, model_channels)
        self.resnet5a = self.resnet_block_fn(1)
        self.resnet5b = self.resnet_block_fn(1)
        self.convout = torch.nn.Conv3d(model_channels,
                                       channels,
                                       kernel_size,
                                       padding='same')

    def forward(self, x, t):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, channels, D, H, W)
        t : torch.Tensor of shape (B,)

        Returns
        -------
        torch.Tensor of shape (B, channels, D, H, W)
        """
        # t : (nbatch,)
        # x : (nbatch, 1, d, h, w)

        # Downsampled part
        te = self.time_embed(t)  # (nbatch, m)
        y = self.convin(x)  # (nbatch, m, d, h, w)
        y1 = self.resnet1b(
            self.resnet1a(y, te), te
        )  # (nbatch, m, d, h, w)

        # Downsampled part
        y = self.downsampler12(y1)  # (nbatch, 2*m, d//2, h//2, w//2)

        y2 = self.resnet2b(
            self.resnet2a(y, te), te
        )  # (nbatch, 2*m, d//2, h//2, w//2)

        y = self.downsampler23(y2)  # (nbatch, 4*m, d//4, h//4, w//4)

        # The lower part
        y = self.resnet3b(
            self.resnet3a(y, te), te
        )  # (nbatch, 4*m, d//4, h//4, w//4)

        yattn = self.resnetattnb(
                    self.attn(self.resnetattna(y, te)), te
        )  # (nbatch, 4*m, d//4, h//4, w//4)
        y = y + yattn

        y = self.resnet3d(
            self.resnet3c(y, te), te
        )  # (nbatch, 4*m, d//4, h//4, w//4)

        # Upsampling back
        y = self.upsampler34(y) + y2  # (nbatch, 2*m, d//2, h//2, w//2)

        y = self.resnet4b(
            self.resnet4a(y, te), te
        )  # (nbatch, 2*m, d//2, h//2, w//2)

        # Upsampling back
        y = self.upsampler45(y) + y1  # (nbatch, m, d, h, w)
        y = self.resnet5b(self.resnet5a(y, te), te)  # (nbatch, m, d, h, w)
        y = self.convout(y)  # (nbatch, 1, d, h, w)

        return y

    def resnet_block_fn(self,
                        input_multiplier):
        return ResnetBlockB(input_multiplier*self.model_channels,
                            self.model_channels,
                            kernel_size=self.kernel_size,
                            dropout=self.dropout)


class PUNet3DBUncond(PUNet3DB):
    def __init__(self, model_channels,
                 channels=1,
                 kernel_size=3,
                 dropout=0.0):
        """
        Parameters
        ----------
        model_channels : int
            The number of channels in the model
        channels : int
            The number of input channes
        kernel_size : int
            The size of the convolutional kernel.
        dropout : float
            The dropout value.
        """
        super().__init__(model_channels,
                         channels,
                         conditional_channels=0,
                         kernel_size=kernel_size,
                         dropout=dropout)


class PUNet3DBCond(PUNet3DB):
    def __init__(self, model_channels,
                 channels=1,
                 conditional_channels=1,
                 kernel_size=3,
                 dropout=0.0):
        """
        Parameters
        ----------
        model_channels : int
            The number of channels in the model
        channels : int
            The number of input (and output) channels in the model
        conditional_channels : int
            The number of the conditional channels in the model
        kernel_size : int
            The size of the convolutional kernel.
        dropout : float
            The dropout value.
        """
        super().__init__(model_channels,
                         channels,
                         conditional_channels,
                         kernel_size=kernel_size,
                         dropout=dropout)

    def forward(self, x, t, y):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, channels, D, H, W)
        t : torch.Tensor of shape (B,)
        y : torch.Tensor of shape (B, conditional_channels, D, H, W)

        Returns
        -------
        torch.Tensor of shape (B, channels, D, H, W)
        """
        # t : (nbatch,)
        # x : (nbatch, 1, d, h, w)
        # y : (nbatch, 1, d, h, w)
        x = torch.concatenate([x, y], axis=1)  # (nbatch, 2, d, h, w)
        return super().forward(x, t)

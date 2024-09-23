import math

import torch
import einops

from .utils import GaussianFourierProjection
from .utils import DateGaussianFourierProjection


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
        torch.Tensor of shape (nbatch, output_channels, 1, 1)
        """

        # te : (nbatch, embed_channels)
        # returns : (nbatch, output_channels, 1, 1)
        return self.net(te).unsqueeze(-1).unsqueeze(-1)


class ResnetBlockB(torch.nn.Module):
    def __init__(self, input_channels,
                 time_embed_dim,
                 date_embed_dim=None,
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
        if date_embed_dim is None:
            date_embed_dim = time_embed_dim
        if output_channels is None:
            output_channels = input_channels
            self.has_residual_connection = True
        else:
            self.has_residual_connection = False
        self.act = torch.nn.SiLU()
        self.gnorm1 = torch.nn.GroupNorm(input_channels, input_channels)
        self.conv1 = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.gnorm2 = torch.nn.GroupNorm(output_channels, output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv2 = torch.nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.timeblock = ResnetTimeBlock(time_embed_dim, output_channels)
        self.dateblock = ResnetTimeBlock(date_embed_dim, output_channels)

    def forward(self, x, te, de=None):

        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, H, W)
        te : torch.Tensor of shape (B, time_embed_dim)
        de : torch.Tensor of shape (B, date_embed_dim)

        Returns
        -------
        torch.Tensor of shape (B, output_channels, H, W)
        """

        # x : (B, C_in, H, W)
        # te : (B, C_embed)
        y = self.conv1(self.act(self.gnorm1(x)))  # (B, C_out, H, W)
        yt = self.timeblock(te)
        if de is None:
            y = y + yt
        else:
            yd = self.dateblock(de)
            y = y + yt+yd  # (B, C_out, H, W)
        y = self.conv2(
            self.dropout(self.act(self.gnorm2(y)))
        )  # (B, C_out, H, W)
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
        self.downsampler = torch.nn.MaxPool2d(2)
        self.conv = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding='same')

    def forward(self, x):

        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, H, W)

        Returns
        -------
        torch.Tensor of shape (B, output_channels, H//2, W//2)
        """

        # x : (B, C_in, H, W)
        # returns : (B, C_out, H//2, W//2)
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
        self.conv = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding='same')

    def forward(self, x):

        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, H, W)

        Returns
        -------
        torch.Tensor of shape (B, output_channels, H*2, W*2)
        """

        # x : (B, C_in, H, W)
        # returns : (B, C_out, H*2, W*2)
        return self.conv(self.upsampler(x))


class TwoDimensionalAttention(torch.nn.Module):

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
        x : torch.Tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor of shape (B, C, H, W)

        OBS: VARIABLES NAMES IN THIS FUNCTION ARE NOT GOOD!!! CHANGE IT IN THE
        FUTURE!!!
        """

        w, h = x.shape[-2], x.shape[-1]
        x_r = einops.rearrange(x, 'b c w h -> b (w h) c')
        x_r, _ = self.mhattn(x_r, x_r, x_r)
        x_r = einops.rearrange(x_r, 'b (w h) c -> b c w h', w=w, h=h)
        return x_r


class OUNet(torch.nn.Module):
    def __init__(self, model_channels,
                 channels=1,
                 conditional_channels=0,
                 kernel_size=3,
                 dropout=0.0
                 ):

        """
        Parameters
        ----------
        model_channels : int
            The number of channels in the model
        channels : int
            The number of input channels
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
        self.convin = torch.nn.Conv2d(in_channels=in_channels,
                                      out_channels=model_channels,
                                      kernel_size=kernel_size,
                                      padding='same')
        self.time_embed = GaussianFourierProjection(model_channels)
        self.date_emb = DateGaussianFourierProjection(model_channels)
        self.resnet1a = self.resnet_block_fn(1)
        self.resnet1b = self.resnet_block_fn(1)
        self.downsampler12 = DownSampler(model_channels, 2*model_channels)
        self.resnet2a = self.resnet_block_fn(2)
        self.resnet2b = self.resnet_block_fn(2)
        self.downsampler23 = DownSampler(2*model_channels, 4*model_channels)
        self.resnet3a = self.resnet_block_fn(4)
        self.resnet3b = self.resnet_block_fn(4)
        self.attn = TwoDimensionalAttention(4*model_channels)
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
        self.convout = torch.nn.Conv2d(model_channels,
                                       channels,
                                       kernel_size,
                                       padding='same')

    def forward(self, x, t, dates=None):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, channels, H, W)
        t : torch.Tensor of shape (B,)

        Returns
        -------
        torch.Tensor of shape (B, channels, H, W)
        """

        # t : (nbatch,)
        # x : (nbatch, 1, w, h)
        # Downsampled part

        te = self.time_embed(t)  # (nbatch, m)
        if dates is not None:
            de = self.date_emb(dates)  # (nbatch, m)
        else:
            de = None
        y = self.convin(x)  # (nbatch, m, w, h)
        y1 = self.resnet1b(
            self.resnet1a(y, te, de), te, de
        )  # (nbatch, m, w, h)

        # Downsampled part
        y = self.downsampler12(y1)  # (nbatch, 2*m, w//2, h//2)

        y2 = self.resnet2b(
            self.resnet2a(y, te, de), te, de
        )  # (nbatch, 2*m, w//2, h//2)

        y = self.downsampler23(y2)  # (nbatch, 4*m, w//4, h//4)

        # The lower part
        y = self.resnet3b(
            self.resnet3a(y, te, de), te, de
        )  # (nbatch, 4*m, w//4, h//4)

        # y = y + self.attn(y)  # (nbatch, 4*m, w//4, h//4)
        yattn = self.resnetattnb(
                    self.attn(self.resnetattna(y, te,)), te
        )  # (nbatch, 4*m, w//4, h//4)
        y = y + yattn

        y = self.resnet3d(
            self.resnet3c(y, te, de), te, de
        )  # (nbatch, 4*m, w//4, h//4)

        # Upsampling back
        y = self.upsampler34(y) + y2  # (nbatch, 2*m, w//2, h//2)

        y = self.resnet4b(
            self.resnet4a(y, te, de), te, de
        )  # (nbatch, 2*m, w//2, h//2)

        # Upsampling back
        y = self.upsampler45(y) + y1  # (nbatch, m, w, h)
        y = self.resnet5b(self.resnet5a(y, te, de), te, de)  # (nbatch, m, w,h)
        y = self.convout(y)  # (nbatch, 1, w, h)

        return y

    def resnet_block_fn(self,
                        input_multiplier):
        return ResnetBlockB(input_multiplier*self.model_channels,
                            self.model_channels,
                            kernel_size=self.kernel_size,
                            dropout=self.dropout)


class OUNetUncond(OUNet):

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


class OUNetCond(OUNet):

    def __init__(self, model_channels,
                 channels=1,
                 conditional_channels=1,
                 kernel_size=3,
                 dropout=0.0,
                 ):

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
                         dropout=dropout,
                         )

    def forward(self, x, t, y):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, channels, H, W)
        t : torch.Tensor of shape (B,)
        y : torch.Tensor of shape (B, conditional_channels, H, W)
        dates : torch.Tensor of shape (B, day, month)

        Returns
        -------
        torch.Tensor of shape (B, channels, H, W)
        """
        if type(y) == type({}):

            dates = y['dates']
            try:
                latlon = y['latlon']
            except Exception:
                pass
            y = y['y']

        # t : (nbatch,)
        # x : (nbatch, 1, w, h)
        # y : (nbatch, 1, w, h)

        x = torch.concatenate([x, y], axis=1)
        # (nbatch, 2, w, h)
        # print(f'PunetCOND input shape{x.shape}')
        return super().forward(x, t, dates)

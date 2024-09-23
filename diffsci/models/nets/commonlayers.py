import math

import torch

# The following import is done for not breaking any code importing
# attention layers from commonlayers.py
from .attention import (NDimensionalAttention,  # noqa: F401
                        TwoDimensionalAttention,  # noqa: F401
                        ThreeDimensionalAttention)  # noqa: F401
from . import normedlayers


class SwiGLU(torch.nn.Module):
    def __init__(self, in_dims: int,
                 out_dims: int):
        super().__init__()
        self.act = torch.nn.SiLU()
        self.linear1 = torch.nn.Linear(in_dims, out_dims)
        self.linear2 = torch.nn.Linear(in_dims, out_dims)

    def forward(self, x):
        return self.linear1(x) * self.act(self.linear2(x))


class DownSampler(torch.nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 dimension=2,
                 scale_factor=2,
                 kernel_size=3,
                 bias=True,
                 convolution_type="default"):

        """
        Parameters
        ----------
        input_channels : int
            The number of input channels
        output_channels : int
            The number of output channels
        dimension : int
            The dimension of the input
        """

        super().__init__()
        self.dimension = dimension
        self.convolution_type = convolution_type
        self.bias = bias
        conv_fn = self.get_convolution_function()

        self.conv = conv_fn(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding='same',
            bias=bias)

        if dimension == 2:
            self.downsampler = torch.nn.MaxPool2d(scale_factor)
        elif dimension == 3:
            self.downsampler = torch.nn.MaxPool3d(scale_factor)
        else:
            raise ValueError(f"Invalid dimension {dimension}")

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

    def get_convolution_function(self):
        if self.convolution_type == "default":
            return torch.nn.Conv2d if self.dimension == 2 else torch.nn.Conv3d
        elif self.convolution_type == "circular":
            return CircularConv2d if self.dimension == 2 else CircularConv3d
        elif self.convolution_type == "mp":
            return (normedlayers.MagnitudePreservingConv2d
                    if self.dimension == 2
                    else normedlayers.MagnitudePreservingConv3d)
        else:
            raise ValueError(
                f"Invalid convolution type: {self.convolution_type}")


class UpSampler(torch.nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 dimension=2,
                 scale_factor=2,
                 kernel_size=3,
                 bias=True,
                 convolution_type="default"):

        """
        Parameters
        ----------
        input_channels : int
            The number of input channels
        output_channels : int
            The number of output channels
        """

        super().__init__()
        self.dimension = dimension
        self.convolution_type = convolution_type
        self.bias = bias
        conv_fn = self.get_convolution_function()

        self.conv = conv_fn(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding='same',
            bias=bias)
        self.upsampler = torch.nn.Upsample(scale_factor=scale_factor)

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

    def get_convolution_function(self):
        if self.convolution_type == "default":
            return torch.nn.Conv2d if self.dimension == 2 else torch.nn.Conv3d
        elif self.convolution_type == "circular":
            return CircularConv2d if self.dimension == 2 else CircularConv3d
        elif self.convolution_type == "mp":
            return (normedlayers.MagnitudePreservingConv2d
                    if self.dimension == 2
                    else normedlayers.MagnitudePreservingConv3d)
        else:
            raise ValueError(
                f"Invalid convolution type: {self.convolution_type}")


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
        x : torch.Tensor of shape (...)

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


class ConvolutionalFourierProjection(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 scale=30.0,
                 bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.scale = scale
        wshape = [input_dim, embed_dim//2]
        self.register_buffer('W',
                             torch.randn(wshape)*scale)
        if bias:
            self.register_buffer('bias',
                                 torch.randn(embed_dim//2)*scale)

    def forward(self, x):
        # x : (B, C, H, W)
        # returns : (B, embed_dim, H, W)
        xc = torch.einsum('bc...,cd->bd...', x, 2*math.pi*self.W)
        # xc : (B, embed_dim//2, H, W)
        if hasattr(self, 'bias'):
            bias_shape = self.embed_dim + [1]*(x.dim()-2)
            xc = xc + self.bias.view(*bias_shape)
        xc = torch.cat([torch.sin(xc), torch.cos(xc)], dim=1)
        return xc


class GaussianFourierProjectionVector(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, scale=30.0):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the input
        embed_dim : int
            The dimension of the embedding
        scale : float
            The scale of the gaussian distribution
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        W = torch.randn((input_dim, embed_dim//2))*scale
        self.register_buffer('W',
                             W)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (..., input_dim)

        Returns
        -------
        x_proj : torch.Tensor of shape (..., embed_dim)
        """
        x_proj = 2*math.pi*x@self.W  # (..., embed_dim//2)
        x_proj = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # (..., embed_dim)
        return x_proj


class GroupRMSNorm(torch.nn.Module):
    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-5,
                 affine=True):
        """
        Group RMS normalization layer. This layer divides the channels into
        groups and normalizes along the (C//G, *) dimensions.

        Parameters
        ----------
        num_groups : int
            The number of groups to divide the channels
        num_channels : int
            The number of channels expected in the input
        eps : float
            The epsilon value to avoid division by zero
        affine : bool
            Whether to apply an affine transformation to the input
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, *)

        Returns
        -------
        x : torch.Tensor of shape (B, C, *)
        """
        B, C = x.shape[:2]
        G = self.num_groups
        x = x.view(B, G, C//G, *x.shape[2:])
        normalize_dims = tuple(range(2, x.dim()))

        x = x / (torch.sqrt(x.pow(2).mean(dim=normalize_dims, keepdim=True)
                            + self.eps))
        x = x.view(B, C, *x.shape[3:])
        if self.affine:
            w = self.weight.view(1, C, *([1]*(x.dim()-2)))
            b = self.bias.view(1, C, *([1]*(x.dim()-2)))
            x = x*w + b
        return x


class GroupPixNorm(torch.nn.Module):
    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-5,
                 affine=True):
        """
        Group Pix normalization layer. This layer divides the channels into
        groups and normalizes along the [C//G] dimensions.

        Parameters
        ----------
        num_groups : int
            The number of groups to divide the channels
        num_channels : int
            The number of channels expected in the input
        eps : float
            The epsilon value to avoid division by zero
        affine : bool
            Whether to apply an affine transformation to the input
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, *)

        Returns
        -------
        x : torch.Tensor of shape (B, C, *)
        """
        B, C = x.shape[:2]
        G = self.num_groups
        x = x.view(B, G, C//G, *x.shape[2:])
        normalize_dims = [2]
        x = x / (torch.sqrt(x.pow(2).mean(dim=normalize_dims, keepdim=True)
                            + self.eps))
        x = x.view(B, C, *x.shape[3:])
        if self.affine:
            w = self.weight.view(1, C, *([1]*(x.dim()-2)))
            b = self.bias.view(1, C, *([1]*(x.dim()-2)))
            x = x*w + b
        return x


class GroupLNorm(torch.nn.Module):
    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-5,
                 affine=True):
        """
        Group Layer normalization layer. This layer divides the channels into
        groups and normalizes along the (C//G, *) dimensions.

        Parameters
        ----------
        num_groups : int
            The number of groups to divide the channels
        num_channels : int
            The number of channels expected in the input
        eps : float
            The epsilon value to avoid division by zero
        affine : bool
            Whether to apply an affine transformation to the input
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, *)

        Returns
        -------
        x : torch.Tensor of shape (B, C, *)
        """
        B, C = x.shape[:2]
        G = self.num_groups
        x = x.view(B, G, C//G, *x.shape[2:])
        normalize_dims = tuple(range(2, x.dim()))
        x = x - x.mean(dim=normalize_dims, keepdim=True)
        x = x / (torch.sqrt(x.pow(2).mean(dim=normalize_dims, keepdim=True)
                            + self.eps))
        x = x.view(B, C, *x.shape[3:])
        if self.affine:
            w = self.weight.view(1, C, *([1]*(x.dim()-2)))
            b = self.bias.view(1, C, *([1]*(x.dim()-2)))
            x = x*w + b
        return x


class ResnetTimeBlock(torch.nn.Module):
    def __init__(self,
                 embed_channels,
                 ouput_channels,
                 dimension=2,
                 magnitude_preserving=False):
        """
        Parameters
        ----------
        embed_channels : int
            The number of channels in the embedding
        ouput_channels : int
            The number of channels in the output
        """
        super().__init__()
        self.dimension = dimension
        if not magnitude_preserving:
            linear_fn = torch.nn.Linear
        else:
            linear_fn = normedlayers.MagnitudePreservingLinear
        self.net = torch.nn.Sequential(
            linear_fn(embed_channels, 4*embed_channels),
            torch.nn.SiLU(),
            linear_fn(4*embed_channels, 4*embed_channels),
            torch.nn.SiLU(),
            linear_fn(4*embed_channels, ouput_channels)
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
        yt = self.net(te)
        newdim = yt.shape + (1,)*self.dimension
        yt = yt.view(*newdim)
        return yt


class ResnetBlock(torch.nn.Module):
    def __init__(self, input_channels,
                 time_embed_dim,
                 output_channels=None,
                 dimension=2,
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
        convfunc = torch.nn.Conv2d if dimension == 2 else torch.nn.Conv3d
        self.conv1 = convfunc(
            input_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.gnorm2 = torch.nn.GroupNorm(output_channels, output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv2 = convfunc(
            output_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.timeblock = ResnetTimeBlock(time_embed_dim,
                                         output_channels,
                                         dimension=dimension)

    def forward(self, x, te):

        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, H, W)
        te : torch.Tensor of shape (B, time_embed_dim)

        Returns
        -------
        torch.Tensor of shape (B, output_channels, H, W)
        """

        # x : (B, C_in, H, W)
        # te : (B, C_embed)
        y = self.conv1(self.act(self.gnorm1(x)))  # (B, C_out, H, W)
        yt = self.timeblock(te)
        y = y + yt  # (B, C_out, H, W)
        y = self.conv2(
            self.dropout(self.act(self.gnorm2(x)))
        )  # (B, C_out, H, W)
        if self.has_residual_connection:
            y = y + x
        return y


class ResnetBlockB(torch.nn.Module):
    def __init__(self, input_channels,
                 time_embed_dim,
                 output_channels=None,
                 dimension=2,
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
        convfunc = torch.nn.Conv2d if dimension == 2 else torch.nn.Conv3d
        self.conv1 = convfunc(
            input_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.gnorm2 = torch.nn.GroupNorm(output_channels, output_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.conv2 = convfunc(
            output_channels,
            output_channels,
            kernel_size,
            padding="same"
        )

        self.timeblock = ResnetTimeBlock(time_embed_dim,
                                         output_channels,
                                         dimension=dimension)

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


class ResnetBlockC(torch.nn.Module):
    def __init__(self, input_channels,
                 time_embed_dim,
                 output_channels=None,
                 dimension=2,
                 kernel_size=3,
                 dropout=0.0,
                 first_norm="GroupLN",
                 second_norm="GroupRMS",
                 affine_norm=True,
                 convolution_type="default",
                 bias=True):
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
        first_norm : str
            The normalization layer to use after the first convolution.
            Default: "GroupLN"
        second_norm : str
            The normalization layer to use after the second convolution.
            Default: "GroupRMS"
        affine_norm : bool
            Whether to apply an learnable affine transformation
            to the normalization
        convolution_type : str
            The type of convolution to use. Default: "default"
        """
        super().__init__()
        kernel_size = kernel_size
        if output_channels is None:
            output_channels = input_channels
            self.has_residual_connection = True
        else:
            self.has_residual_connection = False
        self.act = torch.nn.SiLU()
        self.dimension = dimension
        self.convolution_type = convolution_type
        self.first_norm = first_norm
        self.second_norm = second_norm

        gnorm1_fn, gnorm2_fn = self.get_normalization_functions()

        self.gnorm1 = gnorm1_fn(input_channels,
                                input_channels,
                                affine=affine_norm)

        self.gnorm2 = gnorm2_fn(output_channels,
                                output_channels,
                                affine=affine_norm)

        convfunc = self.get_convolution_function()
        self.conv1 = convfunc(
            input_channels,
            output_channels,
            kernel_size,
            padding="same",
            bias=bias
        )
        self.conv2 = convfunc(
            output_channels,
            output_channels,
            kernel_size,
            padding="same",
            bias=bias
        )

        self.dropout = torch.nn.Dropout(p=dropout)

        if convolution_type == "mp":
            magnitude_preserving = True
        else:
            magnitude_preserving = False
        self.timeblock = ResnetTimeBlock(
            time_embed_dim,
            output_channels,
            dimension=dimension,
            magnitude_preserving=magnitude_preserving
        )

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
            self.dropout(self.act((self.gnorm2(y))))
        )  # (B, C_out, D, H, W)
        if self.has_residual_connection:
            y = y + x
        return y

    def get_convolution_function(self):
        if self.convolution_type == "default":
            return torch.nn.Conv2d if self.dimension == 2 else torch.nn.Conv3d
        elif self.convolution_type == "circular":
            return CircularConv2d if self.dimension == 2 else CircularConv3d
        elif self.convolution_type == "mp":
            return (normedlayers.MagnitudePreservingConv2d
                    if self.dimension == 2
                    else normedlayers.MagnitudePreservingConv3d)
        else:
            raise ValueError(
                f"Invalid convolution type: {self.convolution_type}")

    def get_normalization_functions(self):
        if self.first_norm == "GroupLN":
            gnorm1 = torch.nn.GroupNorm
        elif self.first_norm == "GroupRMS":
            gnorm1 = GroupRMSNorm
        elif self.first_norm == "GroupPix":
            gnorm1 = GroupPixNorm
        else:
            gnorm1 = torch.nn.Identity
        if self.second_norm == "GroupLN":
            gnorm2 = torch.nn.GroupNorm
        elif self.second_norm == "GroupRMS":
            gnorm2 = GroupRMSNorm
        elif self.second_norm == "GroupPix":
            gnorm2 = GroupPixNorm
        else:
            gnorm2 = torch.nn.Identity
        return gnorm1, gnorm2


class BatchDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size(0), device=x.device) > self.p
            xshape = x.size()
            mask = mask.view(x.size(0), *([1]*(len(xshape)-1)))
            x = x*mask
        return x


class CircularConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 *args, **kwargs):
        super().__init__()
        assert (kernel_size % 2 == 1)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = kernel_size//2
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    *args,
                                    **kwargs)
        self.pad = torch.nn.CircularPad2d(self.padding)

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)


class CircularConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 *args, **kwargs):
        super().__init__()
        assert (kernel_size % 2 == 1)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = kernel_size//2
        self.conv = torch.nn.Conv3d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    *args, **kwargs)
        self.pad = torch.nn.CircularPad3d(self.padding)

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)

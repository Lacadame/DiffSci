from typing import List, Optional, Tuple

import torch
import einops

from . import commonlayers
from . import normedlayers
from .punetg_config import PUNetGConfig


class PUNetGEncoder(torch.nn.Module):
    def __init__(self,
                 config: PUNetGConfig,
                 use_time_embedding: bool = False,
                 output_channels: Optional[int] = None):
        super().__init__()
        self.config = config
        self.use_time_embedding = use_time_embedding
        self.output_channels = output_channels

        if use_time_embedding:
            self.time_projection = commonlayers.GaussianFourierProjection(
                embed_dim=config.model_channels,
                scale=config.time_projection_scale
            )

        self.convin = self.make_convin()
        self.downward_blocks, self.downsamplers = self.make_downward_blocks()
        self.bottom_blocks = self.make_bottom_blocks()

        if self.output_channels is not None:
            self.projection = EncoderFlattener(
                config.extended_channel_expansion[-1] * config.model_channels,
                output_channels
            )

    def make_convin(self):
        conv_cls = self.choose_conv_cls()
        input_channels = (self.config.input_channels +
                          (0 if self.config.bias else 1))

        if self.config.in_embedding:
            return commonlayers.ConvolutionalFourierProjection(
                input_dim=input_channels,
                embed_dim=self.config.model_channels,
                scale=self.config.input_projection_scale,
                bias=self.config.bias
            )
        else:
            return conv_cls(
                in_channels=input_channels,
                out_channels=self.config.model_channels,
                kernel_size=self.config.in_out_kernel_size,
                padding='same',
                bias=self.config.bias
            )

    def make_downward_blocks(self):
        blocks = torch.nn.ModuleList()
        downsamplers = torch.nn.ModuleList()
        number_resnet_per_block = self.config.number_resnet_downward_block

        for i, input_multiplier in enumerate(
                self.config.extended_channel_expansion[:-1]
                ):
            output_multiplier = self.config.extended_channel_expansion[i+1]
            resnet_block = self.resnet_block_fn(input_multiplier,
                                                number_resnet_per_block)
            downsampler = self.downsampler_fn(input_multiplier,
                                              output_multiplier)
            blocks.append(resnet_block)
            downsamplers.append(downsampler)
        return blocks, downsamplers

    def make_bottom_blocks(self):
        input_multiplier = self.config.extended_channel_expansion[-1]
        before_block = self.resnet_block_fn(
            input_multiplier, self.config.number_resnet_before_attn_block)
        attn_resnet_block = self.resnet_block_fn(
            input_multiplier, self.config.number_resnet_attn_block)
        attn_block = self.attn_block_fn(
            input_multiplier, self.config.number_resnet_attn_block)
        after_block = self.resnet_block_fn(
            input_multiplier, self.config.number_resnet_after_attn_block)

        return torch.nn.ModuleList([before_block,
                                    attn_resnet_block,
                                    attn_block,
                                    after_block])

    def choose_conv_cls(self):
        if self.config.dimension == 1:
            raise NotImplementedError("1D convolution not implemented yet")
        elif self.config.dimension == 2:
            if self.config.convolution_type == "default":
                conv_cls = torch.nn.Conv2d
            elif self.config.convolution_type == "circular":
                conv_cls = commonlayers.CircularConv2d
            elif self.config.convolution_type == "mp":
                conv_cls = normedlayers.MagnitudePreservingConv2d
        elif self.config.dimension == 3:
            if self.config.convolution_type == "default":
                conv_cls = torch.nn.Conv3d
            elif self.config.convolution_type == "circular":
                conv_cls = commonlayers.CircularConv3d
            elif self.config.convolution_type == "mp":
                conv_cls = normedlayers.MagnitudePreservingConv3d
        else:
            raise ValueError(f"Invalid dimension {self.config.dimension}")
        return conv_cls

    def resnet_fn(self, input_multiplier: int):
        model_channels = self.config.model_channels
        dimension = self.config.dimension
        kernel_size = self.config.kernel_size
        dropout = self.config.dropout
        resnet_channels = input_multiplier * model_channels
        first_norm = self.config.first_resblock_norm
        second_norm = self.config.second_resblock_norm
        conv_type = self.config.convolution_type
        affine_norm = self.config.affine_norm
        time_embed_dim = (model_channels
                          if self.use_time_embedding
                          else None)
        return commonlayers.ResnetBlockC(
            resnet_channels,
            time_embed_dim,
            dimension=dimension,
            kernel_size=kernel_size,
            dropout=dropout,
            first_norm=first_norm,
            second_norm=second_norm,
            affine_norm=affine_norm,
            convolution_type=conv_type,
            bias=self.config.bias
        )

    def resnet_block_fn(self,
                        input_multiplier: int,
                        number_resnet_per_block: int):
        return torch.nn.ModuleList(
            [self.resnet_fn(input_multiplier)
             for _ in range(number_resnet_per_block)])

    def attn_fn(self, input_multiplier: int):
        input_channels = input_multiplier * self.config.model_channels
        attn_residual = self.config.attn_residual
        attn_type = self.config.attn_type
        magnitude_preserving = self.config.magnitude_preserving
        if self.config.dimension == 1:
            raise NotImplementedError("1D attention not implemented yet")
        elif self.config.dimension == 2:
            attn_cls = commonlayers.TwoDimensionalAttention
        elif self.config.dimension == 3:
            attn_cls = commonlayers.ThreeDimensionalAttention
        else:
            raise ValueError(f"Invalid dimension {self.config.dimension}")
        return attn_cls(input_channels,
                        attn_residual=attn_residual,
                        type=attn_type,
                        magnitude_preserving=magnitude_preserving)

    def attn_block_fn(self,
                      input_multiplier: int,
                      number_resnet_attn_block: int):
        return torch.nn.ModuleList([
            self.attn_fn(input_multiplier)
            for _ in range(number_resnet_attn_block-1)])

    def downsampler_fn(self, input_multiplier: int, output_multiplier: int):
        dimension = self.config.dimension
        kernel_size = self.config.transition_kernel_size
        scale_factor = self.config.transition_scale_factor
        input_channels = input_multiplier * self.config.model_channels
        output_channels = output_multiplier * self.config.model_channels
        return commonlayers.DownSampler(
            input_channels,
            output_channels,
            dimension=dimension,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            bias=self.config.bias,
            convolution_type=self.config.convolution_type
        )

    def resnet_block_forward(self, x, te, resnet_block):
        for resnet in resnet_block:
            x = resnet(x, te)
        return x

    def resnet_attn_block_forward(self, x, te, resnet_block, attn_block):
        for i, resnet in enumerate(resnet_block):
            x = resnet(x, te)
            if i < len(attn_block):
                attn = attn_block[i]
                x = attn(x)
        return x

    def forward(self,
                x: torch.Tensor,
                t: Optional[torch.Tensor] = None,
                return_intermediate_outputs: bool = False
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        if not self.config.bias:
            xe_shape = list(x.shape)
            xe_shape[1] = 1
            xe = torch.ones(xe_shape, device=x.device)
            x = torch.cat([x, xe], dim=1)

        x = self.convin(x)

        if self.use_time_embedding and t is not None:
            te = self.time_projection(t)
        else:
            te = None

        intermediate_outputs = []
        for resnet_block, downsampler in zip(self.downward_blocks,
                                             self.downsamplers):
            x = self.resnet_block_forward(x, te, resnet_block)
            intermediate_outputs.append(x.clone())
            x = downsampler(x)

        (before_block,
         attn_resnet_block,
         attn_block,
         after_block) = self.bottom_blocks
        x = self.resnet_block_forward(x, te, before_block)
        x = self.resnet_attn_block_forward(x,
                                           te,
                                           attn_resnet_block,
                                           attn_block)
        x = self.resnet_block_forward(x, te, after_block)
        if self.output_channels is not None:
            x = self.projection(x)
        if not return_intermediate_outputs:
            return x
        else:
            return x, intermediate_outputs


class PUNetGDecoder(torch.nn.Module):
    def __init__(self, config: PUNetGConfig, use_time_embedding: bool = False):
        super().__init__()
        self.config = config
        self.use_time_embedding = use_time_embedding

        if use_time_embedding:
            self.time_projection = commonlayers.GaussianFourierProjection(
                embed_dim=config.model_channels,
                scale=config.time_projection_scale
            )

        self.upward_blocks, self.upsamplers = self.make_upward_blocks()
        self.convout = self.make_convout()

    def make_upward_blocks(self):
        blocks = torch.nn.ModuleList()
        upsamplers = torch.nn.ModuleList()
        number_resnet_per_block = self.config.number_resnet_upward_block
        reversed_extended_channel_expansion = list(
            reversed(self.config.extended_channel_expansion))

        for i, input_multiplier in enumerate(
                reversed_extended_channel_expansion[:-1]):
            output_multiplier = reversed_extended_channel_expansion[i+1]
            upsampler = self.upsampler_fn(
                input_multiplier, output_multiplier)
            resnet_block = self.resnet_block_fn(
                output_multiplier, number_resnet_per_block)
            blocks.append(resnet_block)
            upsamplers.append(upsampler)
        return blocks, upsamplers

    def make_convout(self):
        conv_cls = self.choose_conv_cls()
        return conv_cls(
            in_channels=self.config.model_channels,
            out_channels=self.config.output_channels,
            kernel_size=self.config.in_out_kernel_size,
            padding='same',
            bias=self.config.bias
        )

    def choose_conv_cls(self):
        if self.config.dimension == 1:
            raise NotImplementedError("1D convolution not implemented yet")
        elif self.config.dimension == 2:
            if self.config.convolution_type == "default":
                conv_cls = torch.nn.Conv2d
            elif self.config.convolution_type == "circular":
                conv_cls = commonlayers.CircularConv2d
            elif self.config.convolution_type == "mp":
                conv_cls = normedlayers.MagnitudePreservingConv2d
        elif self.config.dimension == 3:
            if self.config.convolution_type == "default":
                conv_cls = torch.nn.Conv3d
            elif self.config.convolution_type == "circular":
                conv_cls = commonlayers.CircularConv3d
            elif self.config.convolution_type == "mp":
                conv_cls = normedlayers.MagnitudePreservingConv3d
        else:
            raise ValueError(f"Invalid dimension {self.config.dimension}")
        return conv_cls

    def resnet_fn(self, input_multiplier: int):
        model_channels = self.config.model_channels
        dimension = self.config.dimension
        kernel_size = self.config.kernel_size
        dropout = self.config.dropout
        resnet_channels = input_multiplier * model_channels
        first_norm = self.config.first_resblock_norm
        second_norm = self.config.second_resblock_norm
        conv_type = self.config.convolution_type
        affine_norm = self.config.affine_norm
        time_embed_dim = (model_channels
                          if self.use_time_embedding
                          else None)
        return commonlayers.ResnetBlockC(
            resnet_channels,
            time_embed_dim,
            dimension=dimension,
            kernel_size=kernel_size,
            dropout=dropout,
            first_norm=first_norm,
            second_norm=second_norm,
            affine_norm=affine_norm,
            convolution_type=conv_type,
            bias=self.config.bias
        )

    def resnet_block_fn(self, input_multiplier: int,
                        number_resnet_per_block: int):
        return torch.nn.ModuleList([
            self.resnet_fn(input_multiplier)
            for _ in range(number_resnet_per_block)])

    def upsampler_fn(self, input_multiplier: int, output_multiplier: int):
        dimension = self.config.dimension
        kernel_size = self.config.transition_kernel_size
        scale_factor = self.config.transition_scale_factor
        input_channels = input_multiplier * self.config.model_channels
        output_channels = output_multiplier * self.config.model_channels
        return commonlayers.UpSampler(
            input_channels,
            output_channels,
            dimension=dimension,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            bias=self.config.bias,
            convolution_type=self.config.convolution_type
        )

    def resnet_block_forward(self, x, te, resnet_block):
        for resnet in resnet_block:
            x = resnet(x, te)
        return x

    def forward(self,
                x: torch.Tensor,
                t: Optional[torch.Tensor] = None,
                intermediate_outputs: Optional[List[torch.Tensor]] = None,
                ) -> torch.Tensor:
        if self.use_time_embedding and t is not None:
            te = self.time_projection(t)
        else:
            te = None

        for resnet_block, upsampler in zip(self.upward_blocks,
                                           self.upsamplers):
            x = upsampler(x)
            if intermediate_outputs:
                x = x + intermediate_outputs.pop()
            x = self.resnet_block_forward(x, te, resnet_block)
        x = self.convout(x)
        return x


class EncoderFlattener(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.linear = torch.nn.Linear(input_channels, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : shape [batch, input_channels, *spatial]
        # out : shape [batch, output_channels]
        x = einops.reduce(x, 'b c ... -> b c', 'mean')
        return self.linear(x)

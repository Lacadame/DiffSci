from typing import Any

import torch

from . import commonlayers
from . import normedlayers
from .punetg_config import PUNetGConfig


class PUNetG(torch.nn.Module):
    def __init__(self,
                 config: PUNetGConfig,
                 conditional_embedding: torch.nn.Module | None = None):
        super().__init__()
        self.config = config
        self.time_projection = commonlayers.GaussianFourierProjection(
            embed_dim=config.model_channels,
            scale=config.time_projection_scale
        )
        self.conditional_embedding = conditional_embedding
        self.convin, self.convout = self.make_convin_and_convout()
        self.downward_blocks, self.downsamplers = self.make_downward_blocks()
        self.upward_blocks, self.upsamplers = self.make_upward_blocks()
        self.before_block, self.after_block = \
            self.make_non_attn_bottom_blocks()
        self.attn_resnet_block, self.attn_block = \
            self.make_attn_bottom_blocks()
        self.cond_dropout = torch.nn.Dropout(config.cond_dropout)

    def export_description(self) -> dict[Any]:
        has_conditional_embedding = self.conditional_embedding is not None
        if getattr(self.conditional_embedding,
                   "export_description", None):
            cemb_args = self.conditional_embedding.export_description()
        else:
            cemb_args = None
        args = dict(
            config=self.config.export_description(),
            conditional_embedding_args=cemb_args,
            has_conditional_embedding=has_conditional_embedding
        )
        return args

    def make_downward_blocks(self):
        blocks = torch.nn.ModuleList()
        downsamplers = torch.nn.ModuleList()
        number_resnet_per_block = self.config.number_resnet_downward_block

        for i, input_multiplier in enumerate(
                self.config.extended_channel_expansion[:-1]):
            output_multiplier = self.config.extended_channel_expansion[i+1]
            resnet_block = self.resnet_block_fn(input_multiplier,
                                                number_resnet_per_block)
            downsampler = self.downsampler_fn(input_multiplier,
                                              output_multiplier)
            blocks.append(resnet_block)
            downsamplers.append(downsampler)
        return blocks, downsamplers

    def make_upward_blocks(self):
        blocks = torch.nn.ModuleList()
        upsamplers = torch.nn.ModuleList()
        number_resnet_per_block = self.config.number_resnet_upward_block
        reversed_extended_channel_expansion = list(reversed(
            self.config.extended_channel_expansion
        ))
        for i, input_multiplier in enumerate(
                reversed_extended_channel_expansion[:-1]):
            output_multiplier = reversed_extended_channel_expansion[i+1]
            upsampler = self.upsampler_fn(input_multiplier,
                                          output_multiplier)
            resnet_block = self.resnet_block_fn(output_multiplier,
                                                number_resnet_per_block)
            blocks.append(resnet_block)
            upsamplers.append(upsampler)
        return blocks, upsamplers

    def make_non_attn_bottom_blocks(self):
        number_resnet_before_attn_block = \
            self.config.number_resnet_before_attn_block
        number_resnet_after_attn_block = \
            self.config.number_resnet_after_attn_block
        input_multiplier = self.config.extended_channel_expansion[-1]
        before_block = self.resnet_block_fn(input_multiplier,
                                            number_resnet_before_attn_block)
        after_block = self.resnet_block_fn(input_multiplier,
                                           number_resnet_after_attn_block)
        return before_block, after_block

    def make_attn_bottom_blocks(self):
        number_resnet_attn_block = self.config.number_resnet_attn_block
        input_multiplier = self.config.extended_channel_expansion[-1]
        resnet_attn_block = self.resnet_block_fn(input_multiplier,
                                                 number_resnet_attn_block)
        attn_block = self.attn_block_fn(input_multiplier,
                                        number_resnet_attn_block)
        return resnet_attn_block, attn_block

    def make_convin_and_convout(self):
        conv_cls = self.choose_conv_cls()
        if not self.config.bias:
            input_channels = self.config.input_channels + 1
        else:
            input_channels = self.config.input_channels
        if self.config.in_embedding:
            # Uses fixed input embedding, ignore input_channels parameters
            self.convin = commonlayers.ConvolutionalFourierProjection(
                input_dim=input_channels,
                embed_dim=self.config.model_channels,
                scale=self.config.input_projection_scale,
                bias=self.config.bias
            )
        else:
            self.convin = conv_cls(
                in_channels=input_channels,
                out_channels=self.config.model_channels,
                kernel_size=self.config.in_out_kernel_size,
                padding='same',
                bias=self.config.bias)
        self.convout = conv_cls(
            in_channels=self.config.model_channels,
            out_channels=self.config.output_channels,
            kernel_size=self.config.in_out_kernel_size,
            padding='same',
            bias=self.config.bias)
        return self.convin, self.convout

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

    def resnet_fn(self,
                  input_multiplier: int):
        model_channels = self.config.model_channels
        dimension = self.config.dimension
        kernel_size = self.config.kernel_size
        dropout = self.config.dropout
        resnet_channels = input_multiplier * model_channels
        first_norm = self.config.first_resblock_norm
        second_norm = self.config.second_resblock_norm
        conv_type = self.config.convolution_type
        affine_norm = self.config.affine_norm
        return commonlayers.ResnetBlockC(
                           resnet_channels,
                           model_channels,
                           dimension=dimension,
                           kernel_size=kernel_size,
                           dropout=dropout,
                           first_norm=first_norm,
                           second_norm=second_norm,
                           affine_norm=affine_norm,
                           convolution_type=conv_type,
                           bias=self.config.bias)

    def resnet_block_fn(self,
                        input_multiplier: int,
                        number_resnet_per_block: int):
        resnet_block = torch.nn.ModuleList(
            [self.resnet_fn(input_multiplier)
             for _ in range(number_resnet_per_block)]
        )
        return resnet_block

    def attn_fn(self,
                input_multiplier: int):
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
        attn_block = torch.nn.ModuleList(
            [self.attn_fn(input_multiplier)
             for _ in range(number_resnet_attn_block-1)]
        )
        return attn_block

    def downsampler_fn(self,
                       input_multiplier: int,
                       output_multiplier: int):
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

    def upsampler_fn(self,
                     input_multiplier: int,
                     output_multiplier: int):
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

    def resnet_block_forward(self,
                             x,
                             te,
                             resnet_block):
        for resnet in resnet_block:
            x = resnet(x, te)
        return x

    def resnet_attn_block_forward(self,
                                  x,
                                  te,
                                  resnet_block,
                                  attn_block):
        for i, resnet in enumerate(resnet_block):
            x = resnet(x, te)
            if i < len(attn_block):
                attn = attn_block[i]
                x = attn(x)
        return x

    def encode(self,
               x,
               te):
        intermediate_outputs = []
        for resnet_block, downsampler in zip(self.downward_blocks,
                                             self.downsamplers):
            x = self.resnet_block_forward(x, te, resnet_block)
            intermediate_outputs.append(x.clone())
            x = downsampler(x)
        return x, intermediate_outputs

    def decode(self,
               x,
               te,
               intermediate_outputs):
        for resnet_block, upsampler in zip(self.upward_blocks,
                                           self.upsamplers):
            x = upsampler(x)
            x = x + intermediate_outputs.pop()
            x = self.resnet_block_forward(x, te, resnet_block)
        return x

    def bottom_forward(self,
                       x,
                       te):
        x = self.resnet_block_forward(x, te, self.before_block)
        xa = self.resnet_attn_block_forward(x, te,
                                            self.attn_resnet_block,
                                            self.attn_block)
        x = x + xa
        x = self.resnet_block_forward(x, te, self.after_block)
        return x

    def forward(self, x, t, y=None):
        if not self.config.bias:
            xe_shape = list(x.shape)
            xe_shape[1] = 1
            xe = torch.ones(xe_shape).to(x)
            x = torch.cat([x, xe], dim=1)
        x = self.convin(x)
        te = self.time_projection(t)
        if y is not None:
            if self.conditional_embedding is None:
                ye = y
            else:
                ye = self.conditional_embedding(y)
            te = te + self.cond_dropout(ye)
        x, intermediate_outputs = self.encode(x, te)
        x = self.bottom_forward(x, te)
        x = self.decode(x, te, intermediate_outputs)
        x = self.convout(x)
        return x


class PUNetGCond(PUNetG):
    def __init__(self,
                 config: PUNetGConfig,
                 conditional_embedding: torch.nn.Module | None = None,
                 channel_conditional_items: list[str] | None = False):
        super().__init__(config, conditional_embedding)
        self.channel_conditional_items = channel_conditional_items

    def export_description(self) -> dict[Any]:
        args = super().export_description()
        args["channel_conditional_items"] = self.channel_conditional_items
        return args

    def forward(self, x, t, y=None):
        y_channels = []
        for item in self.channel_conditional_items:
            y_channels.append(y[item])
        # Filter the y dict to exclude the channel conditional items
        y = {k: v for k, v in y.items()
             if k not in self.channel_conditional_items}
        if len(y) == 0:  # If y is empty, set it to None
            y = None
        y_cat = torch.cat(y_channels, dim=1)

        # If y_cat has only one batch dimension and x has more than one,
        # we need to expand y_cat to match the batch dimension of x
        if y_cat.shape[0] == 1 and x.shape[0] > 1:
            y_cat = torch.cat([y_cat]*x.shape[0], dim=0)
        x = torch.cat([x, y_cat], dim=1)
        return super().forward(x, t, y)
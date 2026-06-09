from typing import Any

import torch

from . import commonlayers


class ADMConfig(object):
    def __init__(self,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 dimension: list[int] = 2,
                 model_channels: int = 64,
                 time_embed_dim: int = 64,
                 output_embed_dim: int = 256,
                 channel_expansion: list[int] = [2, 4],
                 number_resnet_downward_block: int = 2,
                 number_resnet_upward_block: int = 2,
                 number_resnet_attn_block: int = 2,
                 number_resnet_before_attn_block: int = 2,
                 number_resnet_after_attn_block: int = 2,
                 kernel_size: int = 3,
                 time_projection_scale: float = 30.0,
                 transition_scale_factor: int = 2,
                 transition_kernel_size: int = 3,
                 dropout: float = 0.0,
                 cond_dropout: float = 0.0,
                 first_resblock_norm: str = 'GroupLN',
                 second_resblock_norm: str = 'GroupRMS',
                 affine_norm: bool = True,
                 convolution_type: str = "default",
                 num_groups: int = 1,
                 skip_integration_type: str = 'concat',
                 attn_residual: bool = True,
                 decoder_type: int = 1):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model_channels = model_channels
        self.time_embed_dim = time_embed_dim
        self.output_embed_dim = output_embed_dim
        self.channel_expansion = channel_expansion
        self.dimension = dimension
        self.number_resnet_downward_block = number_resnet_downward_block
        self.number_resnet_upward_block = number_resnet_upward_block
        self.number_resnet_attn_block = number_resnet_attn_block
        self.number_resnet_before_attn_block = number_resnet_before_attn_block
        self.number_resnet_after_attn_block = number_resnet_after_attn_block
        self.number_resnet_attn_block = number_resnet_attn_block
        self.kernel_size = kernel_size
        self.time_projection_scale = time_projection_scale
        self.transition_scale_factor = transition_scale_factor
        self.transition_kernel_size = transition_kernel_size
        self.dropout = dropout
        self.cond_dropout = cond_dropout
        self.first_resblock_norm = first_resblock_norm
        self.second_resblock_norm = second_resblock_norm
        self.affine_norm = affine_norm
        self.convolution_type = convolution_type
        self.num_groups = num_groups
        self.skip_integration_type = skip_integration_type
        self.attn_residual = attn_residual
        self.decoder_type = decoder_type

    @property
    def middle_channel(self):
        return self.model_channels * self.channel_expansion[-1]

    @property
    def extended_channel_expansion(self):
        return [1] + self.channel_expansion

    @property
    def middle_block_attn_config(self):
        part1 = [False]*self.number_resnet_before_attn_block
        part2 = [True]*(self.number_resnet_attn_block - 1) + [False]
        part3 = [False]*self.number_resnet_after_attn_block
        return part1 + part2 + part3

    @property
    def num_blocks_middle_block(self):
        return (self.number_resnet_before_attn_block +
                self.number_resnet_attn_block +
                self.number_resnet_after_attn_block)

    def export_description(self) -> dict[Any]:
        args = dict(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            model_channels=self.model_channels,
            time_embed_dim=self.time_embed_dim,
            output_embed_dim=self.output_embed_dim,
            channel_expansion=self.channel_expansion,
            dimension=self.dimension,
            number_resnet_downward_block=self.number_resnet_downward_block,
            number_resnet_upward_block=self.number_resnet_upward_block,
            number_resnet_attn_block=self.number_resnet_attn_block,
            number_resnet_before_attn_block=(
                self.number_resnet_before_attn_block
            ),
            number_resnet_after_attn_block=self.number_resnet_after_attn_block,
            kernel_size=self.kernel_size,
            time_projection_scale=self.time_projection_scale,
            transition_scale_factor=self.transition_scale_factor,
            transition_kernel_size=self.transition_kernel_size,
            dropout=self.dropout,
            cond_dropout=self.cond_dropout,
            first_resblock_norm=self.first_resblock_norm,
            second_resblock_norm=self.second_resblock_norm,
            affine_norm=self.affine_norm,
            convolution_type=self.convolution_type,
            num_groups=self.num_groups,
            skip_integration_type=self.skip_integration_type,
            attn_residual=self.attn_residual,
            decoder_type=self.decoder_type
        )
        return args


class ADM(torch.nn.Module):
    def __init__(self,
                 config: ADMConfig,
                 conditional_embedding: torch.nn.Module | None = None):
        super().__init__()
        self.config = config
        self.conditional_embedding = conditional_embedding

        self.time_embedding = ADMTimeEmbedding(config.time_embed_dim,
                                               config.output_embed_dim,
                                               config.time_projection_scale)

        self.encoder = ADMEncoder(
            config.model_channels,
            config.output_embed_dim,
            config.extended_channel_expansion,
            config.number_resnet_downward_block,
            config.convolution_type,
            has_residual=True,
            has_attn=False,
            first_norm=config.first_resblock_norm,
            second_norm=config.second_resblock_norm,
            dimension=config.dimension,
            num_groups=config.num_groups,
            pdrop=config.dropout,
            downsample_type='avg',
            downsample_factor=config.transition_scale_factor,
            attn_type='default',
            attn_heads=1,
            attn_residual=config.attn_residual
        )

        self.middle_block = ADMMiddleBlock(
            config.middle_channel,
            config.output_embed_dim,
            config.num_blocks_middle_block,
            conv_type=config.convolution_type,
            has_residual=True,
            has_attn=config.middle_block_attn_config,
            first_norm=config.first_resblock_norm,
            second_norm=config.second_resblock_norm,
            dimension=config.dimension,
            num_groups=config.num_groups,
            pdrop=config.dropout,
            attn_type='default',
            attn_heads=1,
            attn_residual=config.attn_residual
        )

        self.decoder = ADMDecoder(
            config.model_channels,
            config.output_embed_dim,
            config.extended_channel_expansion[::-1],
            config.number_resnet_upward_block,
            config.convolution_type,
            has_residual=True,
            has_attn=False,
            first_norm=config.first_resblock_norm,
            second_norm=config.second_resblock_norm,
            dimension=config.dimension,
            num_groups=config.num_groups,
            pdrop=config.dropout,
            upsample_factor=config.transition_scale_factor,
            attn_type='default',
            attn_heads=1,
            attn_residual=config.attn_residual,
            skip_integration_type=config.skip_integration_type,
            decoder_type=config.decoder_type
        )

        self.input_layer = torch.nn.Conv2d(config.input_channels,
                                           config.model_channels,
                                           kernel_size=config.kernel_size,
                                           padding='same')
        self.output_layer = torch.nn.Conv2d(config.model_channels,
                                            config.output_channels,
                                            kernel_size=config.kernel_size,
                                            padding='same')
        self.cond_dropout = torch.nn.Dropout(config.cond_dropout)

    def forward(self, x, t, y=None):
        if y is not None:
            ye = self.conditional_embedding(y)
            if self.cond_dropout is not None:
                ye = self.cond_dropout(ye)
        else:
            if self.conditional_embedding is not None:
                ye = torch.zeros(x.shape[0],
                                 self.config.output_embed_dim).to(x)
            else:
                ye = None
        te = self.time_embedding(t, ye)
        x = self.input_layer(x)
        x, intermediate_outputs = self.encoder(x, te)
        x = self.middle_block(x, te)
        x = self.decoder(x, te, intermediate_outputs)
        x = self.output_layer(x)
        return x


class ADMBaseBlock(torch.nn.Module):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 channels_embed: int,
                 channels_skip: int | None = None,
                 conv_type: str = 'default',
                 image_sample: str | None = None,
                 has_residual: bool = False,
                 has_attn: bool = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 affine_norm: bool = True,
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 image_sample_type: str | None = None,
                 image_sample_factor: int = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True,
                 skip_integration_type: str = 'concat'):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.channels_embed = channels_embed
        self.channels_skip = channels_skip
        self.conv_type = conv_type
        self.image_sample = image_sample
        self.has_residual = has_residual
        self.has_attn = has_attn
        self.first_norm = first_norm
        self.second_norm = second_norm
        self.affine_norm = affine_norm
        self.dimension = dimension
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.image_sample_type = image_sample_type
        self.image_sample_factor = image_sample_factor
        self.attn_type = attn_type
        self.attn_heads = attn_heads
        self.attn_residual = attn_residual
        self.skip_integration_type = skip_integration_type

        self.channels_in_modified = self.get_channels_in_modified()

        self.norm1, self.norm2 = self.make_norm_layers(
            self.channels_in_modified,
            channels_out)
        self.conv1 = self.conv_fn(self.channels_in_modified,
                                  channels_out,
                                  kernel_size=3,
                                  padding='same')
        self.conv2 = self.conv_fn(channels_out,
                                  channels_out,
                                  kernel_size=3,
                                  padding='same')
        self.act = torch.nn.SiLU()
        self.embed_linear = torch.nn.Linear(channels_embed, 2*channels_out)

        self.dropout = torch.nn.Dropout(pdrop)

        if self.has_residual:
            self.convresidual = self.conv_fn(self.channels_in_modified,
                                             channels_out,
                                             kernel_size=1)

        if self.has_attn:
            self.attn = self.make_attn_layer()

        if self.image_sample:
            self.image_sample_layer = self.make_image_sample()

    def forward(self, x, te, skip=None):
        # x : (B, Cin, H, W) or (B, Cin, D, H, W)
        # te : (B, Cembed)
        # skip : (B, Cskip, H, W) or (B, Cskip, D, H, W) or None
        # In the following comments, we assume that x is (B, Cin, H, W)
        if self.channels_skip:
            if self.skip_integration_type == 'concat':
                x = torch.cat([x, skip], dim=1)  # (B, Cin+Cskip, H, W)
            elif self.skip_integration_type == 'add':
                x = x + skip  # (B, Cin, H, W)
            else:
                raise ValueError(f"Invalid skip integration type "
                                 f"{self.skip_integration_type}")
        x1 = self.first_block(x)  # (B, Cout, H, W)
        te1, te2 = self.embed_block(te)  # (B, Cout, 1, 1), (B, Cout, 1, 1)
        x1t = x1*te1 + te2  # (B, Cout, H, W)
        x2 = self.second_block(x1t)  # (B, Cout, H, W)
        if self.has_residual:
            xr = self.residual_block(x)  # (B, Cout, H, W)
            x2 = x2 + xr
        if self.has_attn:
            x2 = self.attn(x2)
        return x2

    def first_block(self, x):
        y = self.norm1(x)
        y = self.act(y)
        if self.image_sample:
            y = self.image_sample_layer(y)
        y = self.conv1(y)
        y = self.norm2(y)
        return y

    def second_block(self, x):
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

    def embed_block(self, te):
        te = self.embed_linear(te)  # (B, 2*Cout)
        te1, te2 = torch.chunk(te, 2, dim=-1)  # (B, Cout), (B, Cout)
        old_shape = te1.shape
        if self.dimension == 2:
            new_shape = (old_shape[0], old_shape[1], 1, 1)
        elif self.dimension == 3:
            new_shape = (old_shape[0], old_shape[1], 1, 1, 1)
        else:
            raise ValueError(f"Invalid dimension {self.dimension}")
        te1 = te1.view(new_shape)
        te2 = te2.view(new_shape)
        return te1, te2

    def residual_block(self, x):
        if self.image_sample:
            x = self.image_sample_layer(x)
        x = self.convresidual(x)
        return x

    def make_image_sample(self):
        if self.image_sample == 'upsample':
            image_sample = self.make_upsample()
        elif self.image_sample == 'downsample':
            image_sample = self.make_downsample()
        else:
            raise ValueError(f"Invalid image sample type"
                             f"{self.image_sample_type}")
        return image_sample

    def make_downsample(self):
        if self.dimension == 1:
            raise NotImplementedError("1D downsampling not implemented yet")
        elif self.dimension == 2:
            if self.image_sample_type == "avg":
                downsample_cls = torch.nn.AvgPool2d
            elif self.image_sample_type == "max":
                downsample_cls = torch.nn.MaxPool2d
        elif self.dimension == 3:
            if self.image_sample_type == "avg":
                downsample_cls = torch.nn.AvgPool3d
            elif self.image_sample_type == "max":
                downsample_cls = torch.nn.MaxPool3d
        else:
            raise ValueError(f"Invalid dimension {self.dimension}")
        return downsample_cls(kernel_size=self.image_sample_factor)

    def make_upsample(self):
        upsample = torch.nn.Upsample(
            scale_factor=self.image_sample_factor,
            mode=self.image_sample_type
        )
        return upsample

    def make_norm_layers(self, channels_in, channels_out):
        if self.first_norm == 'GroupLN':
            norm1 = torch.nn.GroupNorm(self.num_groups,
                                       channels_in,
                                       affine=self.affine_norm)
        elif self.first_norm == 'GroupRMS':
            norm1 = commonlayers.GroupRMSNorm(self.num_groups,
                                              channels_in,
                                              affine=self.affine_norm)
        else:
            raise ValueError(f"Invalid norm {self.first_norm}")
        if self.second_norm == 'GroupLN':
            norm2 = torch.nn.GroupNorm(self.num_groups,
                                       channels_out,
                                       affine=self.affine_norm)
        elif self.second_norm == 'GroupRMS':
            norm2 = commonlayers.GroupRMSNorm(self.num_groups,
                                              channels_out,
                                              affine=self.affine_norm)
        else:
            raise ValueError(f"Invalid norm {self.second_norm}")
        return norm1, norm2

    def make_attn_layer(self):
        if self.dimension == 2:
            attn_cls = commonlayers.TwoDimensionalAttention
        elif self.dimension == 3:
            attn_cls = commonlayers.ThreeDimensionalAttention
        else:
            raise ValueError(f"Invalid dimension {self.dimension}")
        return attn_cls(num_channels=self.channels_out,
                        num_heads=self.attn_heads,
                        type=self.attn_type,
                        attn_residual=self.attn_residual)

    def get_channels_in_modified(self):
        if self.channels_skip and self.skip_integration_type == 'concat':
            channels_in_modified = self.channels_in + self.channels_skip
        else:
            channels_in_modified = self.channels_in
        return channels_in_modified

    @property
    def conv_fn(self):
        if self.dimension == 1:
            raise NotImplementedError("1D convolution not implemented yet")
        elif self.dimension == 2:
            if self.conv_type == "default":
                conv_cls = torch.nn.Conv2d
            elif self.conv_type == "circular":
                conv_cls = commonlayers.CircularConv2d
        elif self.dimension == 3:
            if self.conv_type == "default":
                conv_cls = torch.nn.Conv3d
            elif self.conv_type == "circular":
                conv_cls = commonlayers.CircularConv3d
        else:
            raise ValueError(f"Invalid dimension {self.dimension}")
        return conv_cls


class ADMEncoderBlock(ADMBaseBlock):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 channels_embed: int,
                 conv_type: str = 'default',
                 has_downsample: bool = False,
                 has_residual: bool = False,
                 has_attn: bool = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 downsample_type: str = 'avg',
                 downsample_factor: int = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: str = True):
        image_sample = 'downsample' if has_downsample else None
        channels_skip = None
        super().__init__(channels_in=channels_in,
                         channels_out=channels_out,
                         channels_embed=channels_embed,
                         channels_skip=channels_skip,
                         conv_type=conv_type,
                         image_sample=image_sample,
                         has_residual=has_residual,
                         has_attn=has_attn,
                         first_norm=first_norm,
                         second_norm=second_norm,
                         dimension=dimension,
                         num_groups=num_groups,
                         pdrop=pdrop,
                         image_sample_type=downsample_type,
                         image_sample_factor=downsample_factor,
                         attn_type=attn_type,
                         attn_heads=attn_heads,
                         attn_residual=attn_residual,
                         skip_integration_type='concat')


class ADMDecoderBlock(ADMBaseBlock):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 channels_embed: int,
                 channels_skip: int | None = None,
                 conv_type: str = 'default',
                 has_upsample: bool = False,
                 has_residual: bool = False,
                 has_attn: bool = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 upsample_type: str = 'nearest',
                 upsample_factor: int = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True,
                 skip_integration_type: str = 'concat'):
        image_sample = 'upsample' if has_upsample else None
        super().__init__(channels_in=channels_in,
                         channels_out=channels_out,
                         channels_embed=channels_embed,
                         channels_skip=channels_skip,
                         conv_type=conv_type,
                         image_sample=image_sample,
                         has_residual=has_residual,
                         has_attn=has_attn,
                         first_norm=first_norm,
                         second_norm=second_norm,
                         dimension=dimension,
                         num_groups=num_groups,
                         pdrop=pdrop,
                         image_sample_type=upsample_type,
                         image_sample_factor=upsample_factor,
                         attn_type=attn_type,
                         attn_heads=attn_heads,
                         attn_residual=attn_residual,
                         skip_integration_type=skip_integration_type)


class ADMEncoderLayer(torch.nn.Module):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 channels_embed: int,
                 nblocks: int,
                 conv_type: str = 'default',
                 has_residual: bool = True,
                 has_attn: bool = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 downsample_type: str = 'avg',
                 downsample_factor: int = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.channels_embed = channels_embed
        self.nblocks = nblocks
        self.conv_type = conv_type
        self.has_attn = has_attn
        self.first_norm = first_norm
        self.second_norm = second_norm
        self.dimension = dimension
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.downsample_type = downsample_type
        self.downsample_factor = downsample_factor
        self.attn_type = attn_type
        self.attn_heads = attn_heads
        self.attn_residual = attn_residual

        self.input_blocks = torch.nn.ModuleList([])
        for i in range(nblocks):
            if i != (nblocks - 1):
                channels_out_i = channels_in
                has_downsample_i = False
            else:
                channels_out_i = channels_out
                has_downsample_i = True
            block = ADMEncoderBlock(channels_in=channels_in,
                                    channels_out=channels_out_i,
                                    channels_embed=channels_embed,
                                    conv_type=conv_type,
                                    has_downsample=has_downsample_i,
                                    has_residual=has_residual,
                                    has_attn=has_attn,
                                    first_norm=first_norm,
                                    second_norm=second_norm,
                                    dimension=dimension,
                                    num_groups=num_groups,
                                    pdrop=pdrop,
                                    downsample_type=downsample_type,
                                    downsample_factor=downsample_factor,
                                    attn_type=attn_type,
                                    attn_heads=attn_heads,
                                    attn_residual=attn_residual)
            self.input_blocks.append(block)

    def forward(self, x, te):
        for block in self.input_blocks:
            x = block(x, te)
        xskip = x.clone()
        return x, xskip


class ADMEncoder(torch.nn.Module):
    def __init__(self,
                 model_channels: int,
                 channels_embed: int,
                 channels_mult: list[int] = [1, 2, 4],
                 nblocks_per_layer: int | list[int] = 2,
                 conv_type: str = 'default',
                 has_residual: bool = True,
                 has_attn: bool | list[bool] = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 downsample_type: str = 'avg',
                 downsample_factor: int | list[int] = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True):
        super().__init__()
        self.model_channels = model_channels
        self.channels_mult = channels_mult
        self.channels_embed = channels_embed
        self.nblocks_per_layer = nblocks_per_layer
        self.conv_type = conv_type
        self.has_residual = has_residual
        self.has_attn = has_attn
        self.first_norm = first_norm
        self.second_norm = second_norm
        self.dimension = dimension
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.downsample_type = downsample_type
        self.downsample_factor = downsample_factor
        self.attn_type = attn_type
        self.attn_heads = attn_heads
        self.attn_residual = attn_residual

        if not isinstance(nblocks_per_layer, list):
            nblocks_per_layer = [nblocks_per_layer] * self.nlayers
        if not isinstance(downsample_factor, list):
            downsample_factor = [downsample_factor] * self.nlayers
        if not isinstance(has_attn, list):
            has_attn = [has_attn] * self.nlayers
        assert len(nblocks_per_layer) == self.nlayers
        assert len(downsample_factor) == self.nlayers

        self.layers = torch.nn.ModuleList([])
        for i in range(self.nlayers):
            layer = ADMEncoderLayer(channels_in=self.channels_in[i],
                                    channels_out=self.channels_outs[i],
                                    channels_embed=channels_embed,
                                    nblocks=nblocks_per_layer[i],
                                    conv_type=conv_type,
                                    has_residual=has_residual,
                                    has_attn=has_attn[i],
                                    first_norm=first_norm,
                                    second_norm=second_norm,
                                    dimension=dimension,
                                    num_groups=num_groups,
                                    pdrop=pdrop,
                                    downsample_type=downsample_type,
                                    downsample_factor=downsample_factor[i],
                                    attn_type=attn_type,
                                    attn_heads=attn_heads,
                                    attn_residual=attn_residual)
            self.layers.append(layer)

    def forward(self, x, te):
        intermediate_outputs = [x.clone()]
        for i, layer in enumerate(self.layers):
            x, xskip = layer(x, te)
            intermediate_outputs.append(xskip)
        return x, intermediate_outputs

    @property
    def channels_in(self):
        return [self.model_channels*i for i in self.channels_mult[:-1]]

    @property
    def channels_outs(self):
        return [self.model_channels*i for i in self.channels_mult[1:]]

    @property
    def nlayers(self):
        return len(self.channels_mult) - 1


class ADMDecoderLayer1(torch.nn.Module):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 channels_embed: int,
                 channels_skip: int,
                 nblocks: int = 2,
                 conv_type: str = 'default',
                 has_residual: bool = True,
                 has_attn: bool = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 upsample_factor: int = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True,
                 skip_integration_type: str = 'concat'):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.channels_embed = channels_embed
        self.channels_skip = channels_skip
        self.nblocks = nblocks
        self.conv_type = conv_type
        self.has_residual = has_residual
        self.has_attn = has_attn
        self.first_norm = first_norm
        self.second_norm = second_norm
        self.dimension = dimension
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.upsample_factor = upsample_factor
        self.attn_type = attn_type
        self.attn_heads = attn_heads
        self.attn_residual = attn_residual
        self.skip_integration_type = skip_integration_type

        self.input_blocks = torch.nn.ModuleList([])
        for i in range(nblocks):
            if skip_integration_type == 'concat':
                channels_in_i = channels_in + channels_skip
            else:
                channels_in_i = channels_in
            if i != (nblocks - 1):
                if skip_integration_type == 'concat':
                    channels_out_i = channels_in + channels_skip
                else:
                    channels_out_i = channels_in
                has_upsample_i = False
            else:
                channels_out_i = channels_out
                has_upsample_i = True
            block = ADMDecoderBlock(channels_in=channels_in_i,
                                    channels_out=channels_out_i,
                                    channels_embed=channels_embed,
                                    channels_skip=None,
                                    conv_type=conv_type,
                                    has_upsample=has_upsample_i,
                                    has_residual=has_residual,
                                    has_attn=has_attn,
                                    first_norm=first_norm,
                                    second_norm=second_norm,
                                    dimension=dimension,
                                    num_groups=num_groups,
                                    pdrop=pdrop,
                                    upsample_factor=upsample_factor,
                                    attn_type=attn_type,
                                    attn_heads=attn_heads,
                                    attn_residual=attn_residual)
            self.input_blocks.append(block)

    def forward(self, x, te, skip):
        if self.skip_integration_type == 'concat':
            xh = torch.cat([x, skip], dim=1)
        elif self.skip_integration_type == 'add':
            xh = x + skip
        else:
            raise ValueError(f"Invalid skip integration type "
                             f"{self.skip_integration_type}")
        for block in self.input_blocks:
            xh = block(xh, te)
        return xh


class ADMDecoderLayer2(torch.nn.Module):
    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 channels_embed: int,
                 channels_skip: int,
                 nblocks: int = 2,
                 conv_type: str = 'default',
                 has_residual: bool = True,
                 has_attn: bool = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 upsample_factor: int = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True,
                 skip_integration_type: str = 'concat'):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.channels_embed = channels_embed
        self.channels_skip = channels_skip
        self.nblocks = nblocks
        self.conv_type = conv_type
        self.has_residual = has_residual
        self.has_attn = has_attn
        self.first_norm = first_norm
        self.second_norm = second_norm
        self.dimension = dimension
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.upsample_factor = upsample_factor
        self.attn_type = attn_type
        self.attn_heads = attn_heads
        self.attn_residual = attn_residual
        self.skip_integration_type = skip_integration_type

        self.input_blocks = torch.nn.ModuleList([])
        for i in range(nblocks):
            if i != (nblocks - 1):
                channels_out_i = channels_in
                has_upsample_i = False
            else:
                channels_out_i = channels_out
                has_upsample_i = True
            block = ADMDecoderBlock(
                channels_in=channels_in,
                channels_out=channels_out_i,
                channels_embed=channels_embed,
                channels_skip=channels_skip,
                conv_type=conv_type,
                has_upsample=has_upsample_i,
                has_residual=has_residual,
                has_attn=has_attn,
                first_norm=first_norm,
                second_norm=second_norm,
                dimension=dimension,
                num_groups=num_groups,
                pdrop=pdrop,
                upsample_factor=upsample_factor,
                attn_type=attn_type,
                attn_residual=attn_residual,
                attn_heads=attn_heads,
                skip_integration_type=skip_integration_type
            )
            self.input_blocks.append(block)

    def forward(self, x, te, skip):
        for block in self.input_blocks:
            x = block(x, te, skip)
        return x


class ADMDecoder(torch.nn.Module):
    def __init__(self,
                 model_channels: int,
                 channels_embed: int,
                 channels_mult: list[int] = [4, 2, 1],
                 nblocks_per_layer: int | list[int] = 2,
                 conv_type: str = 'default',
                 has_residual: bool = True,
                 has_attn: bool | list[bool] = False,
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 upsample_factor: int | list[int] = 2,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True,
                 skip_integration_type: str = 'concat',
                 decoder_type: int = 1):
        super().__init__()
        self.model_channels = model_channels
        self.channels_mult = channels_mult
        self.channels_embed = channels_embed
        self.nblocks_per_layer = nblocks_per_layer
        self.conv_type = conv_type
        self.has_residual = has_residual
        self.has_attn = has_attn
        self.first_norm = first_norm
        self.second_norm = second_norm
        self.dimension = dimension
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.upsample_factor = upsample_factor
        self.attn_type = attn_type
        self.attn_heads = attn_heads
        self.attn_residual = attn_residual
        self.decoder_type = decoder_type
        self.skip_integration_type = skip_integration_type

        if not isinstance(nblocks_per_layer, list):
            nblocks_per_layer = [nblocks_per_layer] * self.nlayers
        if not isinstance(upsample_factor, list):
            upsample_factor = [upsample_factor] * self.nlayers
        if not isinstance(has_attn, list):
            has_attn = [has_attn] * self.nlayers
        assert len(nblocks_per_layer) == self.nlayers
        assert len(upsample_factor) == self.nlayers
        assert len(has_attn) == self.nlayers

        self.layers = torch.nn.ModuleList([])
        for i in range(self.nlayers):
            layer = self.decoder_fn(
                channels_in=self.channels_ins[i],
                channels_out=self.channels_outs[i],
                channels_embed=channels_embed,
                channels_skip=self.channels_ins[i],
                nblocks=nblocks_per_layer[i],
                conv_type=conv_type,
                has_residual=has_residual,
                has_attn=has_attn[i],
                first_norm=first_norm,
                second_norm=second_norm,
                dimension=dimension,
                num_groups=num_groups,
                pdrop=pdrop,
                upsample_factor=upsample_factor[i],
                attn_type=attn_type,
                attn_heads=attn_heads,
                attn_residual=attn_residual,
                skip_integration_type=skip_integration_type
            )
            self.layers.append(layer)

    def forward(self, x, te, intermediate_outputs, pop=True):
        for i, layer in enumerate(self.layers):
            if pop:
                h = intermediate_outputs.pop()
            else:
                h = intermediate_outputs[-(i+1)]
            x = layer(x, te, h)
        return x

    @property
    def decoder_fn(self):
        if self.decoder_type == 1:
            return ADMDecoderLayer1
        elif self.decoder_type == 2:
            return ADMDecoderLayer2
        else:
            raise ValueError(f"Invalid decoder type {self.decoder_type}")

    @property
    def channels_ins(self):
        return [self.model_channels*i for i in self.channels_mult[:-1]]

    @property
    def channels_outs(self):
        return [self.model_channels*i for i in self.channels_mult[1:]]

    @property
    def nlayers(self):
        return len(self.channels_mult) - 1


class ADMMiddleBlock(torch.nn.Module):
    def __init__(self,
                 channels: int,
                 channels_embed: int,
                 nblocks: int = 2,
                 conv_type: str = 'default',
                 has_residual: bool = True,
                 has_attn: bool | list[bool] | str = 'default',
                 first_norm: str = 'GroupLN',
                 second_norm: str = 'GroupRMS',
                 dimension: int = 2,
                 num_groups: int = 1,
                 pdrop: float = 0.0,
                 attn_type: str = 'default',
                 attn_heads: int = 1,
                 attn_residual: bool = True):
        super().__init__()
        self.channels = channels
        self.channels_embed = channels_embed
        self.nblocks = nblocks
        self.conv_type = conv_type
        self.has_residual = has_residual
        self.has_attn = has_attn
        self.first_norm = first_norm
        self.second_norm = second_norm
        self.dimension = dimension
        self.num_groups = num_groups
        self.pdrop = pdrop
        self.attn_type = attn_type
        self.attn_heads = attn_heads
        self.attn_residual = attn_residual

        if isinstance(has_attn, str):
            if has_attn == 'default':
                has_attn = [True] * (nblocks - 1) + [False]
            else:
                raise ValueError(f"Invalid has_attn {has_attn}")
        if not isinstance(has_attn, list):
            has_attn = [has_attn] * nblocks
        assert len(has_attn) == nblocks

        self.middle_blocks = torch.nn.ModuleList([])
        for i in range(nblocks):
            block = ADMEncoderBlock(channels_in=channels,
                                    channels_out=channels,
                                    channels_embed=channels_embed,
                                    conv_type=conv_type,
                                    has_downsample=False,
                                    has_residual=has_residual,
                                    has_attn=has_attn[i],
                                    first_norm=first_norm,
                                    second_norm=second_norm,
                                    dimension=dimension,
                                    num_groups=num_groups,
                                    pdrop=pdrop,
                                    attn_type=attn_type,
                                    attn_heads=attn_heads,
                                    attn_residual=attn_residual)
            self.middle_blocks.append(block)

    def forward(self, x, te):
        for block in self.middle_blocks:
            x = block(x, te)
        return x


class ADMTimeEmbedding(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 output_dim: int,
                 projection_scale: float = 30.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.projection_scale = projection_scale

        self.projection = commonlayers.GaussianFourierProjection(
            embed_dim=embed_dim,
            scale=projection_scale
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, output_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(output_dim, output_dim)
        )

        self.act_final = torch.nn.SiLU()

    def forward(self, t, ye=None):
        te = self.projection(t)
        te = self.mlp(te)
        if ye is not None:
            te = te + ye
        te = self.act_final(te)
        return te

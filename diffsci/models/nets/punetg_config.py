from typing import Any


class PUNetGConfig(object):
    def __init__(self,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 dimension: list[int] = 2,
                 model_channels: int = 64,
                 channel_expansion: list[int] = [2, 4],
                 number_resnet_downward_block: int = 2,
                 number_resnet_upward_block: int = 2,
                 number_resnet_attn_block: int = 2,
                 number_resnet_before_attn_block: int = 2,
                 number_resnet_after_attn_block: int = 2,
                 kernel_size: int = 3,
                 in_out_kernel_size: int = 3,
                 in_embedding: bool = False,
                 time_projection_scale: float = 30.0,
                 input_projection_scale: float = 1.0,
                 transition_scale_factor: int = 2,
                 transition_kernel_size: int = 3,
                 dropout: float = 0.0,
                 cond_dropout: float = 0.0,
                 first_resblock_norm: str = 'GroupLN',
                 second_resblock_norm: str = 'GroupRMS',
                 affine_norm: bool = True,
                 convolution_type: str = "default",
                 num_groups: int = 1,
                 attn_residual: bool = False,
                 attn_type: str = "default",
                 bias: bool = True
                 ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model_channels = model_channels
        self.channel_expansion = channel_expansion
        self.dimension = dimension
        self.number_resnet_downward_block = number_resnet_downward_block
        self.number_resnet_upward_block = number_resnet_upward_block
        self.number_resnet_attn_block = number_resnet_attn_block
        self.number_resnet_before_attn_block = number_resnet_before_attn_block
        self.number_resnet_after_attn_block = number_resnet_after_attn_block
        self.number_resnet_attn_block = number_resnet_attn_block
        self.kernel_size = kernel_size
        self.in_out_kernel_size = in_out_kernel_size
        self.in_embedding = in_embedding
        self.time_projection_scale = time_projection_scale
        self.input_projection_scale = input_projection_scale
        self.transition_scale_factor = transition_scale_factor
        self.transition_kernel_size = transition_kernel_size
        self.dropout = dropout
        self.cond_dropout = cond_dropout
        self.first_resblock_norm = first_resblock_norm
        self.second_resblock_norm = second_resblock_norm
        self.affine_norm = affine_norm
        self.convolution_type = convolution_type
        self.num_groups = num_groups
        self.attn_residual = attn_residual
        self.attn_type = attn_type
        self.bias = bias

    @property
    def extended_channel_expansion(self):
        return [1] + self.channel_expansion

    def export_description(self) -> dict[Any]:
        args = dict(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            model_channels=self.model_channels,
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
            in_out_kernel_size=self.in_out_kernel_size,
            in_embedding=self.in_embedding,
            time_projection_scale=self.time_projection_scale,
            input_projection_scale=self.input_projection_scale,
            transition_scale_factor=self.transition_scale_factor,
            transition_kernel_size=self.transition_kernel_size,
            dropout=self.dropout,
            cond_dropout=self.cond_dropout,
            first_resblock_norm=self.first_resblock_norm,
            second_resblock_norm=self.second_resblock_norm,
            affine_norm=self.affine_norm,
            convolution_type=self.convolution_type,
            num_groups=self.num_groups,
            attn_residual=self.attn_residual,
            attn_type=self.attn_type,
            bias=self.bias
        )
        return args

    @property
    def magnitude_preserving(self):
        return self.convolution_type == "mp"

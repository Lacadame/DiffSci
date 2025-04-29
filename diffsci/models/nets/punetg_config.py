from typing import Any


class PUNetGConfig(object):

    """
    PUNetGConfig is a configuration class for defining the hyperparameters and settings
    used in the PUNetG model. This configuration encapsulates all aspects of the model architecture,
    including input/output channels, model dimensions, normalization schemes, and dropout rates.

    Parameters
    ----------
    input_channels : int, optional
        Number of input channels for the model (e.g., 1 for grayscale images). Default is 1.

    output_channels : int, optional
        Number of output channels for the model. Default is 1.

    dimension : list of int, optional
        Spatial dimensions of the input (e.g., 2 for 2D data). Default is [2].

    model_channels : int, optional
        Number of channels in the base layer of the model. Default is 64.

    channel_expansion : list of int, optional
        List specifying the expansion factors for channels at each UNet level. Default is [2, 4] (double, quadruple).

    number_resnet_downward_block : int, optional
        Number of ResNet blocks in each downward (encoder) block. Default is 2.

    number_resnet_upward_block : int, optional
        Number of ResNet blocks in each upward (decoder) block. Default is 2.

    number_resnet_attn_block : int, optional
        Number of ResNet blocks with attention at the bottleneck. Default is 2.

    number_resnet_before_attn_block : int, optional
        Number of ResNet blocks before the attention block at the bottleneck. Default is 2.

    number_resnet_after_attn_block : int, optional
        Number of ResNet blocks after the attention block at the bottleneck. Default is 2.

    kernel_size : int, optional
        Size of the convolutional kernels used in the ResNet blocks. Default is 3.

    in_out_kernel_size : int, optional
        Kernel size for the initial input and final output convolution layers. Default is 3.

    in_embedding : bool, optional
        Whether to use an embedding layer for the input. Default is False.

    time_projection_scale : float, optional
        Scaling factor for the time projection embeddings. Default is 30.0.

    input_projection_scale : float, optional
        Scaling factor for input projection. Default is 1.0.

    transition_scale_factor : int, optional
        Factor for scaling the channels during transitions between blocks. Default is 2.

    transition_kernel_size : int, optional
        Kernel size for transition convolutions. Default is 3.

    dropout : float, optional
        Dropout rate applied within the model. Default is 0.0.

    cond_dropout : float, optional
        Dropout rate applied to the conditional embeddings. Default is 0.0.

    first_resblock_norm : str, optional
        Normalization type for the first ResNet block (e.g., 'GroupLN'). Default is 'GroupLN'.

    second_resblock_norm : str, optional
        Normalization type for the second ResNet block (e.g., 'GroupRMS'). Default is 'GroupRMS'.

    affine_norm : bool, optional
        Whether to use affine normalization in the model. Default is True.

    convolution_type : str, optional
        Type of convolution to use (e.g., 'default'). Default is 'default'.

    num_groups : int, optional
        Number of groups for GroupNorm, if used. Default is 1.

    attn_residual : bool, optional
        Whether to include residual connections in the attention blocks. Default is False.

    attn_type : str, optional
        Type of attention mechanism to use (e.g., 'default'). Default is 'default'.

    bias : bool, optional
        Whether to use bias in convolutional layers. Default is True.

    Attributes
    ----------
    input_channels : int
        Number of input channels for the model.

    output_channels : int
        Number of output channels for the model.

    model_channels : int
        Number of base channels in the model.

    channel_expansion : list of int
        Expansion factors for channels at each level.

    dimension : list of int
        Spatial dimensions of the input data.

    kernel_size : int
        Size of convolutional kernels.

    in_out_kernel_size : int
        Kernel size for input/output convolutions.

    affine_norm : bool
        Indicates if affine normalization is used.

    convolution_type : str
        Type of convolution used in the model.

    dropout : float
        Dropout rate within the model.

    cond_dropout : float
        Dropout rate for conditional embeddings.

    """
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

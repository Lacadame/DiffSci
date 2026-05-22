from typing import Any

import torch

from . import commonlayers
from . import normedlayers
from .punetg_forecast_config import PUNetGForecastConfig


class PUNetGForecast(torch.nn.Module):

    """
    PUNetGForecast is a UNet-style model for deterministic ocean forecasting tasks.

    This model uses convolutional layers, downward and upward UNet blocks for feature extraction
    and reconstruction. Unlike the diffusion-based PUNetG, this version does not use time embeddings
    and is designed for direct prediction tasks in ocean forecasting.
    
    It includes attention-based blocks at the bottleneck for enhanced modeling capacity and can
    optionally use conditional embeddings for guided predictions.

    Parameters
    ----------
    config : PUNetGForecastConfig
        Configuration object that contains hyperparameters and settings for the model,
        including the number of model channels, dropout rates, etc.

    conditional_embedding : torch.nn.Module, optional
        A module for generating conditional embeddings (e.g., environmental or temporal embeddings).
        If provided, it is used to guide the prediction. Default is None.
        When using the conditional data must be associated with the key 'y'. Other keys can be used
        for other types of embedding depending on the application.

    Attributes
    ----------
    config : PUNetGForecastConfig
        Stores the configuration object used to initialize the model.

    conditional_embedding : torch.nn.Module or None
        Optional module for generating conditional embeddings for guided prediction.

    convin : torch.nn.Module
        Initial convolutional layer for processing the input data.

    convout : torch.nn.Module
        Final convolutional layer for producing the output.

    downward_blocks : torch.nn.ModuleList
        List of downward UNet blocks for feature extraction.

    downsamplers : torch.nn.ModuleList
        List of downsampling layers that reduce the spatial dimensions.

    upward_blocks : torch.nn.ModuleList
        List of upward UNet blocks for feature reconstruction.

    upsamplers : torch.nn.ModuleList
        List of upsampling layers that increase the spatial dimensions.

    before_block : torch.nn.Module
        Non-attention block at the bottom of the UNet structure, applied before the attention blocks.

    after_block : torch.nn.Module
        Non-attention block at the bottom of the UNet structure, applied after the attention blocks.

    attn_resnet_block : torch.nn.Module
        Residual block with attention mechanism at the bottleneck.

    attn_block : torch.nn.Module
        Attention block at the bottleneck for enhanced feature extraction.

    cond_dropout : torch.nn.Dropout
        Dropout layer applied to conditional embeddings to prevent overfitting.

    extra_residual : Extra residual extraction on residual layer

    """
    def __init__(self,
                 config: PUNetGForecastConfig,
                 conditional_embedding: torch.nn.Module | None = None,
                 extra_residual: torch.nn.Module | None = None):

        super().__init__()
        self.config = config
        self.extra_residual = extra_residual
        self.conditional_embedding = conditional_embedding
        self.convin, self.convout = self.make_convin_and_convout()
        self.downward_blocks, self.downsamplers = self.make_downward_blocks()
        self.upward_blocks, self.upsamplers = self.make_upward_blocks()
        self.before_block, self.after_block = \
            self.make_non_attn_bottom_blocks()
        self.attn_resnet_block, self.attn_block = \
            self.make_attn_bottom_blocks()
        self.cond_dropout = torch.nn.Dropout(config.cond_dropout)
        if config.cond_drop is not None and config.cond_drop > 0:
            self.cond_drop = commonlayers.ConditionDrop(
                p=config.cond_drop, hidden_dim=config.model_channels, null_is_learnable=config.cond_drop_learnable)
        else:
            self.cond_drop = None

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
            has_conditional_embedding=has_conditional_embedding,
        )
        return args

    def make_downward_blocks(self):
        """
        Create the downward blocks of the Unet architecture based on the number_resnet_downward_block parameter
        """
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
        """
        Create the upward blocks of the Unet architecture based on the number_resnet_upward_block parameter
        """
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
        attn_resnet_block = self.resnet_block_fn(input_multiplier,
                                                 number_resnet_attn_block)
        attn_block = self.attn_block_fn(input_multiplier,
                                        number_resnet_attn_block)
        return attn_resnet_block, attn_block

    def make_convin_and_convout(self):
        input_channels = self.config.input_channels
        conv_cls = self.choose_conv_cls()
        if self.config.in_embedding:
            self.convin = conv_cls(
                in_channels=input_channels+1,
                out_channels=self.config.model_channels,
                kernel_size=self.config.in_out_kernel_size,
                padding='same',
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
        extra_residual = self.extra_residual
        return commonlayers.ResnetBlockC(
                           input_channels=resnet_channels,
                           time_embed_dim=None,
                           dimension=dimension,
                           kernel_size=kernel_size,
                           dropout=dropout,
                           first_norm=first_norm,
                           second_norm=second_norm,
                           affine_norm=affine_norm,
                           convolution_type=conv_type,
                           bias=self.config.bias,
                           extra_residual=extra_residual)

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
             for _ in range(number_resnet_attn_block - 1)]
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
                             resnet_block):
        """
        Forward pass through a sequence of ResNet blocks.
        
        Note: Unlike the diffusion version, this version does not use time embeddings,
        so ResNet blocks are called without the 'te' parameter.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        resnet_block : torch.nn.ModuleList
            List of ResNet blocks
            
        Returns
        -------
        torch.Tensor
            Output tensor after passing through all ResNet blocks
        """
        for resnet in resnet_block:
            x = resnet(x)
        return x

    def resnet_attn_block_forward(self,
                                  x,
                                  resnet_block,
                                  attn_block):
        """
        Forward pass through ResNet blocks with attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        resnet_block : torch.nn.ModuleList
            List of ResNet blocks
        attn_block : torch.nn.ModuleList
            List of attention blocks
            
        Returns
        -------
        torch.Tensor
            Output tensor with attention applied
        """
        for i, resnet in enumerate(resnet_block):
            x = resnet(x)
            if i < len(attn_block):
                attn = attn_block[i]
                x = attn(x)
        return x

    def encode(self,
               x):
        """
        Encode input through downsampling path.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        x : torch.Tensor
            Encoded feature map at the bottleneck
        intermediate_outputs : list[torch.Tensor]
            List of intermediate feature maps for skip connections
        """
        intermediate_outputs = []
        for resnet_block, downsampler in zip(self.downward_blocks,
                                             self.downsamplers):
            x = self.resnet_block_forward(x, resnet_block)
            intermediate_outputs.append(x.clone())
            x = downsampler(x)
        return x, intermediate_outputs

    def decode(self,
               x,
               intermediate_outputs):
        """
        Decode from bottleneck through upsampling path.
        
        Parameters
        ----------
        x : torch.Tensor
            Feature map from bottleneck
        intermediate_outputs : list[torch.Tensor]
            List of intermediate feature maps for skip connections
            
        Returns
        -------
        torch.Tensor
            Upsampled and processed feature map
        """
        for resnet_block, upsampler in zip(self.upward_blocks,
                                           self.upsamplers):
            x = upsampler(x)
            x = x + intermediate_outputs.pop()
            x = self.resnet_block_forward(x, resnet_block)
        return x

    def bottom_forward(self,
                       x):
        """
        Forward pass through the bottleneck of the UNet.
        
        Parameters
        ----------
        x : torch.Tensor
            Input feature map
            
        Returns
        -------
        torch.Tensor
            Output feature map after attention processing
        """
        x = self.resnet_block_forward(x, self.before_block)
        xa = self.resnet_attn_block_forward(x,
                                            self.attn_resnet_block,
                                            self.attn_block)
        x = x + xa
        x = self.resnet_block_forward(x, self.after_block)
        return x

    def forward(self, x, y=None):
        """
        Forward pass of the forecast model.
        
        Unlike the diffusion-based version, this model does not take a time parameter 't'.
        It is designed for direct deterministic forecasting.
        
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W) or (B, C, D, H, W)
            Input ocean data
        y : dict, optional
            Dictionary containing conditional data. Keys can include:
            - 'dates': temporal information
            - 'latlon': geographic information
            - 'bat': bathymetry data
            - 'tau': wind stress data
            Default is None.
            
        Returns
        -------
        torch.Tensor
            Predicted ocean state with same shape as input
        """
        if not self.config.bias:
            xe_shape = list(x.shape)
            xe_shape[1] = 1
            xe = torch.ones(xe_shape).to(x)
            x = torch.cat([x, xe], dim=1)
        
        x = self.convin(x)
        
        # Process conditional embeddings if provided
        ce = None
        if y is not None:
            if self.conditional_embedding is None:
                ce = y
            else:
                ce = self.conditional_embedding(y)  # [B, C]
            
            if self.cond_drop is not None:
                ce = self.cond_drop(ce)
            
            ce = self.cond_dropout(ce)  # [B, C]
            
            # Reshape conditional embedding to match spatial dimensions
            # ce shape: [B, C] -> [B, C, 1, 1] (or [B, C, 1, 1, 1] for 3D)
            ce_shape = [ce.shape[0], ce.shape[1]] + [1] * (x.ndim - 2)
            ce = ce.reshape(ce_shape)
            
            # Add conditional embedding to the input
            x = x + ce

        x, intermediate_outputs = self.encode(x)
        x = self.bottom_forward(x)
        x = self.decode(x, intermediate_outputs)
        x = self.convout(x)
        return x

    def set_conditional_embedding(
            self,
            conditional_embedding: torch.nn.Module | None = None):
        self.conditional_embedding = conditional_embedding

    def calculate_receptive_field(self) -> dict:
        """
        Calculate the theoretical receptive field (RF) of the PUNetGForecast model.

        The receptive field is the region of input space that influences a single
        output pixel. For UNet architectures, we must trace through all conv operations
        accounting for the cumulative stride from downsampling.

        RF Calculation Formula:
            rf += (kernel_size - 1) * current_stride

        This formula accounts for the fact that after downsampling, each "pixel" in
        the feature map corresponds to a larger region in the input space.

        Architecture components analyzed:
        - convin: Single conv with in_out_kernel_size
        - ResnetBlockC: TWO convolutions with kernel_size (norm -> act -> conv1 -> conv2)
        - DownSampler: MaxPool(scale_factor) followed by conv(transition_kernel_size)
        - UpSampler: Upsample (nearest neighbor interpolation) + conv(transition_kernel_size)
        - Attention blocks: GLOBAL attention (flatten all spatial dims) -> infinite RF
        - convout: Single conv with in_out_kernel_size

        Returns
        -------
        dict
            Contains:
            - 'rf': The receptive field size (int or float('inf'))
            - 'has_attention': Whether the model uses attention (makes RF infinite)
            - 'num_attention_layers': Number of attention layers
            - 'trace': Detailed trace of RF computation
            - 'feasible_chunking': Whether the model can be chunked/tiled
            - 'downsampling_factor': Total downsampling factor from input to bottleneck
            - 'config_summary': Summary of relevant config parameters
        """
        config = self.config

        trace = []
        num_down_levels = len(self.downward_blocks)
        trace.append(f"Total downsampling levels: {num_down_levels}")

        # Helper function to add RF contribution from a convolution
        def add_conv_rf(current_rf, kernel_size, current_stride, layer_name):
            rf_contrib = (kernel_size - 1) * current_stride
            new_rf = current_rf + rf_contrib
            trace.append(
                f"  {layer_name}: kernel_size={kernel_size}, stride={current_stride}, "
                f"rf_contrib={rf_contrib}, total_rf={new_rf}"
            )
            return new_rf

        # Helper function to add RF contribution from a ResNet block (TWO convolutions)
        def add_resblock_rf(current_rf, kernel_size, current_stride, block_name):
            # ResNet block has two convolutions
            rf = current_rf
            for i in range(2):
                rf_contrib = (kernel_size - 1) * current_stride
                rf = rf + rf_contrib
            total_contrib = (kernel_size - 1) * current_stride * 2
            trace.append(
                f"  {block_name}: 2x conv(kernel_size={kernel_size}, stride={current_stride}), "
                f"rf_contrib={total_contrib}, total_rf={rf}"
            )
            return rf

        # =====================================================================
        # convin: Initial convolution
        # =====================================================================
        trace.append(f"\n--- INPUT ---")
        rf = 1
        stride = 1
        rf = add_conv_rf(rf, config.in_out_kernel_size, stride, "convin")

        # =====================================================================
        # Downward path: For each level, apply ResnetBlock then DownSampler
        # =====================================================================
        trace.append(f"\n--- DOWNWARD PATH ({num_down_levels} levels) ---")

        for level_idx in range(num_down_levels):
            trace.append(f"\nLevel {level_idx}:")

            # ResnetBlockC blocks at this level
            for block_idx in range(config.number_resnet_downward_block):
                rf = add_resblock_rf(rf, config.kernel_size, stride,
                                     f"  down[{level_idx}].resnet[{block_idx}]")

            # DownSampler: MaxPool + Conv
            # MaxPool does not change RF (it just decimates)
            trace.append(f"  down[{level_idx}].maxpool: no RF change, "
                        f"stride *= {config.transition_scale_factor}")
            stride = stride * config.transition_scale_factor

            # Conv after MaxPool
            rf = add_conv_rf(rf, config.transition_kernel_size, stride,
                            f"  down[{level_idx}].downsample_conv")

        # =====================================================================
        # Bottom: before_block + attn_resnet_block (no attention here) + after_block
        # =====================================================================
        trace.append(f"\n--- BOTTOM (at stride {stride}) ---")

        # before_block: number_resnet_before_attn_block ResnetBlockC
        for block_idx in range(config.number_resnet_before_attn_block):
            rf = add_resblock_rf(rf, config.kernel_size, stride,
                                 f"  before_block[{block_idx}]")

        # attn_resnet_block: number_resnet_attn_block ResnetBlockC (but 0 attention layers)
        # Note: Even with 0 attention layers, the resnet blocks still exist
        for block_idx in range(config.number_resnet_attn_block):
            rf = add_resblock_rf(rf, config.kernel_size, stride,
                                 f"  attn_resnet_block[{block_idx}]")

        # after_block: number_resnet_after_attn_block ResnetBlockC
        for block_idx in range(config.number_resnet_after_attn_block):
            rf = add_resblock_rf(rf, config.kernel_size, stride,
                                 f"  after_block[{block_idx}]")

        # =====================================================================
        # Upward path: For each level, apply UpSampler then ResnetBlockC blocks
        # =====================================================================
        trace.append(f"\n--- UPWARD PATH ({num_down_levels} levels) ---")

        for level_idx in range(num_down_levels - 1, -1, -1):
            trace.append(f"\nLevel {level_idx}:")

            # UpSampler: Upsample (nearest neighbor) then conv(transition_kernel_size)
            # Upsample (nearest neighbor interpolation) does NOT change RF in input space
            # It just spreads values to more pixels. The stride decreases.
            stride = stride // config.transition_scale_factor
            trace.append(f"  up[{level_idx}].upsample: no RF change, stride = {stride}")

            # Conv after upsampling
            rf = add_conv_rf(rf, config.transition_kernel_size, stride,
                             f"  up[{level_idx}].upsample_conv")

            # Skip connection (element-wise add) does not change RF
            trace.append(f"  up[{level_idx}].skip_add: no RF change")

            # ResnetBlockC blocks at this level
            for block_idx in range(config.number_resnet_upward_block):
                rf = add_resblock_rf(rf, config.kernel_size, stride,
                                     f"  up[{level_idx}].resnet[{block_idx}]")

        # =====================================================================
        # convout: Final convolution
        # =====================================================================
        trace.append(f"\n--- OUTPUT ---")
        rf = add_conv_rf(rf, config.in_out_kernel_size, stride, "convout")

        trace.append(f"\nFINAL RF = {rf} pixels")

        return {
            'rf': rf,
            'has_attention': False,
            'num_attention_layers': 0,
            'trace': trace,
            'feasible_chunking': True,
            'downsampling_factor': config.transition_scale_factor ** num_down_levels,
            'config_summary': {
                'number_resnet_attn_block': config.number_resnet_attn_block,
                'number_resnet_downward_block': config.number_resnet_downward_block,
                'number_resnet_upward_block': config.number_resnet_upward_block,
                'number_resnet_before_attn_block': config.number_resnet_before_attn_block,
                'number_resnet_after_attn_block': config.number_resnet_after_attn_block,
                'kernel_size': config.kernel_size,
                'in_out_kernel_size': config.in_out_kernel_size,
                'transition_kernel_size': config.transition_kernel_size,
                'transition_scale_factor': config.transition_scale_factor,
                'channel_expansion': config.channel_expansion,
            }
        }


class PUNetGForecastCond(PUNetGForecast):

    """
    PUNetGForecastCond is a UNet-style model for conditional ocean forecasting tasks.

    This model extends PUNetGForecast by adding support for channel-wise conditional inputs.
    Like the base forecast model, it does not use time embeddings and is designed for direct
    deterministic prediction with additional conditional channels.

    Parameters
    ----------
    config : PUNetGForecastConfig
        Configuration object that contains hyperparameters and settings for the model,
        including the number of model channels, dropout rates, etc.

    conditional_embedding : torch.nn.Module, optional
        A module for generating conditional embeddings (e.g., environmental or temporal embeddings).
        If provided, it is used to guide the prediction. Default is None.

    channel_conditional_items : list[str], optional
        List of dictionary keys in the conditional data that should be concatenated
        to the input channels rather than added as embeddings. Default is None.

    extra_residual : torch.nn.Module, optional
        Additional residual extraction module to be applied on residual blocks.
        Default is None.

    Attributes
    ----------
    channel_conditional_items : list[str]
        The keys referring to the conditional information on the dictionary y
        that should be added as input channels.
    """
    def __init__(self,
                 config: PUNetGForecastConfig,
                 conditional_embedding: torch.nn.Module | None = None,
                 channel_conditional_items: list[str] | None = False,
                 extra_residual: torch.nn.Module | None = None):
        super().__init__(config, conditional_embedding, extra_residual=extra_residual)
        self.channel_conditional_items = channel_conditional_items

    def export_description(self) -> dict[Any]:
        args = super().export_description()
        args["channel_conditional_items"] = self.channel_conditional_items
        return args

    def forward(self, x, y=None):
        """
        Forward pass of the conditional forecast model.
        
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W) or (B, C, D, H, W)
            Input ocean data
        y : dict, optional
            Dictionary containing conditional data. Keys that appear in 
            channel_conditional_items will be concatenated to input channels.
            Other keys are passed to the conditional embedding module.
            Default is None.
            
        Returns
        -------
        torch.Tensor
            Predicted ocean state
        """
        # Extract channel conditional items
        y_channels = []
        for item in self.channel_conditional_items:
            y_channels.append(y[item])
        
        # Filter the y dict to exclude the channel conditional items
        y = {k: v for k, v in y.items()
             if k not in self.channel_conditional_items}
        if len(y) == 0:  # If y is empty, set it to None
            y = None
        
        # Concatenate channel conditional items
        y_cat = torch.cat(y_channels, dim=1)

        # If y_cat has only one batch dimension and x has more than one,
        # we need to expand y_cat to match the batch dimension of x
        if y_cat.shape[0] == 1 and x.shape[0] > 1:
            y_cat = torch.cat([y_cat]*x.shape[0], dim=0)
        
        # Concatenate to input
        x = torch.cat([x, y_cat], dim=1)
        
        # Call parent forward with filtered y
        return super().forward(x, y)
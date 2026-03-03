from typing import Any

import pathlib
import yaml

import torch
import einops

from . import commonlayers
from . import normedlayers


class ResnetSliceBlock(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 dimension=2,
                 magnitude_preserving=False):
        """
        ResNet block for processing temporal slice embeddings.
        
        Takes temporal slices, applies spatial convolutions, and returns
        a spatially-aware embedding that can be added to feature maps.
        """
        super().__init__()
        self.dimension = dimension
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Choose convolution function
        if not magnitude_preserving:
            if dimension == 2:
                conv_fn = torch.nn.Conv2d
            elif dimension == 3:
                conv_fn = torch.nn.Conv3d
            else:
                raise ValueError(f"Invalid dimension {dimension}")
        else:
            if dimension == 2:
                conv_fn = normedlayers.MagnitudePreservingConv2d
            elif dimension == 3:
                conv_fn = normedlayers.MagnitudePreservingConv3d
            else:
                raise ValueError(f"Invalid dimension {dimension}")
        
        # Network architecture similar to ResnetTimeBlock but with convolutions
        intermediate_channels = 4 * input_channels
        
        self.conv1 = conv_fn(
            input_channels,
            intermediate_channels,
            kernel_size=3,
            padding='same'
        )
        self.act1 = torch.nn.SiLU()
        
        self.conv2 = conv_fn(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            padding='same'
        )
        self.act2 = torch.nn.SiLU()
        
        self.conv3 = conv_fn(
            intermediate_channels,
            output_channels,
            kernel_size=3,
            padding='same'
        )
        
        # Group normalization layers
        self.gnorm1 = torch.nn.GroupNorm(
            num_groups=min(32, input_channels), 
            num_channels=input_channels
        )
        self.gnorm2 = torch.nn.GroupNorm(
            num_groups=min(32, intermediate_channels), 
            num_channels=intermediate_channels
        )
        self.gnorm3 = torch.nn.GroupNorm(
            num_groups=min(32, intermediate_channels), 
            num_channels=intermediate_channels
        )

    def forward(self, slice_embeddings, temporal_mask=None, target_spatial_size=None):
        """
        Process temporal slice embeddings with optional masking and spatial resizing.
        
        Parameters
        ----------
        slice_embeddings : torch.Tensor of shape (N, T, C, H, W) or (N, T, C, D, H, W)
            Temporal slice embeddings
        temporal_mask : torch.Tensor of shape (N, T), optional
            Boolean mask indicating valid temporal positions
        target_spatial_size : tuple, optional
            Target spatial size to resize to (H, W) or (D, H, W)
        
        Returns
        -------
        torch.Tensor of shape (N, output_channels, H', W') or (N, output_channels, D', H', W')
            Spatially-aware slice embedding that can be added to feature maps
        """
        N, T, C = slice_embeddings.shape[:3]
        spatial_dims = slice_embeddings.shape[3:]
        
        # Validate input channels
        assert C == self.input_channels, f"Expected {self.input_channels} input channels, got {C}"
        
        # Resize slice embeddings to target spatial size if needed
        if target_spatial_size is not None and target_spatial_size != spatial_dims:
            # Reshape to [N*T, C, ...] for interpolation
            slice_flat = slice_embeddings.view(N * T, C, *spatial_dims)
            
            # Interpolate to target size
            if self.dimension == 2:
                slice_resized = torch.nn.functional.interpolate(
                    slice_flat, size=target_spatial_size, mode='bilinear', align_corners=False
                )
            elif self.dimension == 3:
                slice_resized = torch.nn.functional.interpolate(
                    slice_flat, size=target_spatial_size, mode='trilinear', align_corners=False
                )
            else:
                raise ValueError(f"Unsupported dimension: {self.dimension}")
            
            # Reshape back to [N, T, C, ...]
            slice_embeddings = slice_resized.view(N, T, C, *target_spatial_size)
            spatial_dims = target_spatial_size
        
        # Apply temporal mask if provided
        if temporal_mask is not None:
            mask_expanded = temporal_mask.view(N, T, 1, *([1] * len(spatial_dims)))
            slice_embeddings = slice_embeddings * mask_expanded.float()
        
        # Reshape from [N, T, C, ...] to [N*T, C, ...] for batch processing
        reshaped_input = slice_embeddings.view(N * T, C, *spatial_dims)
        
        # Apply convolutional layers
        y = self.conv1(self.act1(self.gnorm1(reshaped_input)))
        y = self.conv2(self.act2(self.gnorm2(y)))
        y = self.conv3(self.gnorm3(y))
        
        # Reshape back to [N, T, output_channels, ...]
        y = y.view(N, T, self.output_channels, *spatial_dims)
        
        # Handle masked mean computation
        if temporal_mask is not None:
            # Apply mask and compute masked mean
            mask_expanded = temporal_mask.view(N, T, 1, *([1] * len(spatial_dims)))
            y_masked = y * mask_expanded.float()
            
            # Compute mean only over valid positions
            valid_counts = temporal_mask.sum(dim=1, keepdim=True).float()  # [N, 1]
            valid_counts = valid_counts.view(N, 1, *([1] * len(spatial_dims)))
            
            y_sum = y_masked.sum(dim=1)  # [N, output_channels, ...]
            y_mean = y_sum / torch.clamp(valid_counts, min=1.0)  # Avoid division by zero
        else:
            # Take simple mean over temporal dimension
            y_mean = torch.mean(y, dim=1)  # [N, output_channels, ...]
        
        return y_mean


class PUNetVConfig(object):
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
                 bias: bool = True,
                 slice_embed_channels: int | None = None  # New parameter for slice embeddings
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
        self.slice_embed_channels = slice_embed_channels  # New parameter

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
            bias=self.bias,
            slice_embed_channels=self.slice_embed_channels  # Include new parameter
        )
        return args

    @property
    def magnitude_preserving(self):
        return self.convolution_type == "mp"

    @classmethod
    def from_description(cls, description: dict):
        return cls(**description)
    
    @classmethod
    def from_config_file(cls, config_file: pathlib.Path | str):
        with open(config_file, "r") as f:
            description = yaml.safe_load(f)
        return cls.from_description(description)


class PUNetV(torch.nn.Module):
    
    """
    PUNetV is a UNet-style generative model for probabilistic generation task with slice embeddings.

    This model uses a combination of Gaussian Fourier projections for time embeddings,
    convolutional layers, and both downward and upward UNet blocks for feature extraction
    and reconstruction. It is designed to handle conditional embeddings for guided generation
    and includes attention-based blocks at the bottleneck for enhanced modeling capacity.
    
    Now includes support for temporal slice embeddings for video-like data.

    Parameters
    ----------
    config : PUNetVConfig
        Configuration object that contains hyperparameters and settings for the model,
        including the number of model channels, scale of time projection, dropout rates, etc.

    conditional_embedding : torch.nn.Module, optional
        A module for generating conditional embeddings (e.g., text or class embeddings).
        If provided, it is used to guide the generative process. Default is None.
        When using the conditional data must be associated with de key 'y'. Other keys can be used
        for other tipes of embedding depending on the application.

    Attributes
    ----------
    config : PUNetVConfig
        Stores the configuration object used to initialize the model.

    time_projection : GaussianFourierProjection
        A layer for projecting time information into the embedding space using Gaussian Fourier projections.

    conditional_embedding : torch.nn.Module or None
        Optional module for generating conditional embeddings for guided generation.

    convin : torch.nn.Module
        Initial convolutional layer for processing the input data.

    convout : torch.nn.Module
        Final convolutional layer for producing the output.

    slice_projection : torch.nn.Module
        Projection layer for slice embeddings.

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

    extra_residual : Extra residual extration on residual layer 


    """
    def __init__(self,
                 config: PUNetVConfig,
                 conditional_embedding: torch.nn.Module | None = None,
                 extra_residual: torch.nn.Module | None = None):

        super().__init__()
        self.config = config
        self.time_projection = commonlayers.GaussianFourierProjection(
            embed_dim=config.model_channels,
            scale=config.time_projection_scale
        )

        self.extra_residual = extra_residual
        self.conditional_embedding = conditional_embedding
        self.convin, self.convout, self.slice_projection = self.make_convin_and_convout_and_slice_projection()
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
        """
        Create the donward blocks of the Unet architecture based on the number_resnet_downward_block parameter
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
        resnet_attn_block = self.resnet_block_fn(input_multiplier,
                                                 number_resnet_attn_block)
        attn_block = self.attn_block_fn(input_multiplier,
                                        number_resnet_attn_block)
        return resnet_attn_block, attn_block

    def make_convin_and_convout_and_slice_projection(self):
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
        
        # Only create slice projection if slice embeddings are enabled
        if self.config.slice_embed_channels is not None:
            self.slice_projection = conv_cls(
                in_channels=self.config.slice_embed_channels,
                out_channels=self.config.model_channels,
                kernel_size=self.config.in_out_kernel_size,
                padding='same',
                bias=self.config.bias)
        else:
            self.slice_projection = None
    
        return self.convin, self.convout, self.slice_projection

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
        slice_embed_channels = self.config.slice_embed_channels
        
        # Use modified ResnetBlockC that includes slice embeddings
        return ResnetBlockCWithSlices(
                           resnet_channels,
                           model_channels,  # time_embed_dim
                           slice_embed_channels=model_channels if slice_embed_channels is not None else None,
                           output_channels=resnet_channels,
                           dimension=dimension,
                           kernel_size=kernel_size,
                           dropout=dropout,
                           first_norm=first_norm,
                           second_norm=second_norm,
                           affine_norm=affine_norm,
                           convolution_type=conv_type,
                           bias=self.config.bias,
                           extra_residual=extra_residual,
                           downsampler_multiplier=input_multiplier)  # New parameter

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

    def compute_spatial_size_at_level(self, original_size, level):
        """
        Compute the spatial size at a given level of the UNet.
        
        Parameters
        ----------
        original_size : tuple
            Original spatial size (H, W) or (D, H, W)
        level : int
            The level in the UNet (0 = original, 1 = first downsample, etc.)
        
        Returns
        -------
        tuple
            Spatial size at the given level
        """
        scale_factor = self.config.transition_scale_factor ** level
        if isinstance(original_size, int):
            return original_size // scale_factor
        else:
            return tuple(dim // scale_factor for dim in original_size)

    def resnet_block_forward(self,
                             x,
                             te,
                             slice_embeddings,
                             temporal_mask,
                             resnet_block,
                             level=0):
        for resnet in resnet_block:
            x = resnet(x, te, slice_embeddings, temporal_mask, level)
        return x

    def resnet_attn_block_forward(self,
                                  x,
                                  te,
                                  slice_embeddings,
                                  temporal_mask,
                                  resnet_block,
                                  attn_block,
                                  level):
        for i, resnet in enumerate(resnet_block):
            x = resnet(x, te, slice_embeddings, temporal_mask, level)
            if i < len(attn_block):
                attn = attn_block[i]
                x = attn(x)
        return x

    def encode(self,
               x,
               te,
               slice_embeddings,
               temporal_mask):
        intermediate_outputs = []
        current_level = 0
        
        for resnet_block, downsampler in zip(self.downward_blocks,
                                             self.downsamplers):
            x = self.resnet_block_forward(x, te, slice_embeddings, temporal_mask, resnet_block, current_level)
            intermediate_outputs.append(x.clone())
            x = downsampler(x)
            current_level += 1
        return x, intermediate_outputs

    def decode(self,
               x,
               te,
               slice_embeddings,
               temporal_mask,
               intermediate_outputs):
        current_level = len(self.config.extended_channel_expansion) - 1
        
        for resnet_block, upsampler in zip(self.upward_blocks,
                                           self.upsamplers):
            x = upsampler(x)
            x = x + intermediate_outputs.pop()
            current_level -= 1
            x = self.resnet_block_forward(x, te, slice_embeddings, temporal_mask, resnet_block, current_level)
        return x

    def bottom_forward(self,
                       x,
                       te,
                       slice_embeddings,
                       temporal_mask):
        # Bottom level is the deepest level
        bottom_level = len(self.config.extended_channel_expansion) - 1
        
        x = self.resnet_block_forward(x, te, slice_embeddings, temporal_mask, self.before_block, bottom_level)
        xa = self.resnet_attn_block_forward(x, te, slice_embeddings, temporal_mask,
                                            self.attn_resnet_block,
                                            self.attn_block, bottom_level)
        x = x + xa
        x = self.resnet_block_forward(x, te, slice_embeddings, temporal_mask, self.after_block, bottom_level)
        return x

    def apply_slice_projection(self, yb):
        # Only apply if slice projection exists
        if self.slice_projection is None:
            return None
            
        # Reshape [B, T, C, H, W] -> [BT, C, H, W]
        B, T = yb.shape[:2]
        yb_flat = einops.rearrange(yb, 'b t ... -> (b t) ...')
        
        # Apply projection
        yb_proj = self.slice_projection(yb_flat)
        
        # Reshape back to [B, T, C', H, W]
        yb_out = einops.rearrange(yb_proj, '(b t) ... -> b t ...', b=B, t=T)
        
        return yb_out

    def forward(self, x, t, y=None):
        """
        Forward pass with slice embeddings support.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        t : torch.Tensor 
            Time embeddings
        y : dict, optional
            Conditional embeddings. Must contain 'yb' and 'temporal_mask'
        
        Returns
        -------
        torch.Tensor
            Model output
        """
        if not self.config.bias:
            xe_shape = list(x.shape)
            xe_shape[1] = 1
            xe = torch.ones(xe_shape).to(x)
            x = torch.cat([x, xe], dim=1)
        x = self.convin(x)
        te = self.time_projection(t)  # [B, C]
        
        # Process slice embeddings if provided
        yb = None
        temporal_mask = None
        if y is not None:
            if 'yb' in y:
                yb = y.pop('yb')
            if 'temporal_mask' in y:
                temporal_mask = y.pop('temporal_mask')
            if len(y) == 0:
                y = None

        ybe = None
        if yb is not None and self.config.slice_embed_channels is not None:
            ybe = self.apply_slice_projection(yb)  # [B, T, C', H, W]
        
        if y is not None:
            if self.conditional_embedding is None:
                ye = y
            else:
                ye = self.conditional_embedding(y)  # [B, C]
            te = te + self.cond_dropout(ye)  # [B, C]
        
        x, intermediate_outputs = self.encode(x, te, ybe, temporal_mask)
        x = self.bottom_forward(x, te, ybe, temporal_mask)
        x = self.decode(x, te, ybe, temporal_mask, intermediate_outputs)
        x = self.convout(x)
        return x

    def set_conditional_embedding(
            self,
            conditional_embedding: torch.nn.Module | None = None):
        self.conditional_embedding = conditional_embedding


# Create a custom ResnetBlockC that supports slice embeddings
class ResnetBlockCWithSlices(commonlayers.ResnetBlockC):
    def __init__(self, input_channels,
                 time_embed_dim,
                 slice_embed_channels=None,
                 output_channels=None,
                 dimension=2,
                 kernel_size=3,
                 dropout=0.0,
                 first_norm="GroupLN",
                 second_norm="GroupRMS",
                 affine_norm=True,
                 convolution_type="default",
                 bias=True,
                 extra_residual: None | torch.nn.Module = None,
                 downsampler_multiplier=1):
        
        # Initialize parent class
        super().__init__(
            input_channels=input_channels,
            time_embed_dim=time_embed_dim,
            output_channels=output_channels,
            dimension=dimension,
            kernel_size=kernel_size,
            dropout=dropout,
            first_norm=first_norm,
            second_norm=second_norm,
            affine_norm=affine_norm,
            convolution_type=convolution_type,
            bias=bias,
            extra_residual=extra_residual
        )
        
        # Add slice embedding support
        self.has_slice_embed = slice_embed_channels is not None
        self.downsampler_multiplier = downsampler_multiplier
        
        if self.has_slice_embed:
            magnitude_preserving = (convolution_type == "mp")
            
            # Get the actual output channels (may be different from input_channels)
            actual_output_channels = output_channels if output_channels is not None else input_channels
            
            self.slice_embedding = ResnetSliceBlock(
                slice_embed_channels,
                actual_output_channels,
                dimension=dimension,
                magnitude_preserving=magnitude_preserving
            )

    def forward(self, x, te=None, slice_embeddings=None, temporal_mask=None, level=0):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, input_channels, H, W) or (B, input_channels, D, H, W)
            Input feature maps
        te : torch.Tensor of shape (B, time_embed_dim), optional
            Time embeddings
        slice_embeddings : torch.Tensor of shape (B, T, slice_embed_channels, H, W), optional
            Temporal slice embeddings
        temporal_mask : torch.Tensor of shape (B, T), optional
            Temporal mask for variable-length sequences
        level : int
            Current level in the UNet (for spatial scaling)
        
        Returns
        -------
        torch.Tensor
            Output feature maps
        """
        # Validate inputs
        if te is None:
            assert not self.has_time_embed, "Time embedding expected but not provided"
        if slice_embeddings is None:
            assert not self.has_slice_embed, "Slice embeddings expected but not provided"
        
        # First convolution
        y = self.conv1(self.act(self.gnorm1(x)))
        
        # Add time embedding (from parent class)
        if self.has_time_embed and te is not None:
            yt = self.timeblock(te)
            y = y + yt
        
        # Add slice embedding with spatial scaling
        if self.has_slice_embed and slice_embeddings is not None:
            # Compute target spatial size based on current level
            current_spatial_size = x.shape[2:]  # Get spatial dimensions from current feature map
            
            ys = self.slice_embedding(slice_embeddings, temporal_mask, current_spatial_size)
            y = y + ys
        
        # Second convolution
        y = self.conv2(
            self.dropout(self.act(self.gnorm2(y)))
        )
        
        # Residual connections
        if self.has_residual_connection:
            y = y + x
        if self.extra_residual is not None:
            y = y + self.extra_residual(x)
        
        return y


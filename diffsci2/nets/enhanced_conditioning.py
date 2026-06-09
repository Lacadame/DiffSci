"""
Enhanced conditioning modules for stronger spatial conditioning.

These modules are designed for POST-TRAINING from an existing model:
- New layers are initialized to IDENTITY so the model starts where it left off
- No global attention (shift-invariant)
- Per-pixel operations only (computationally efficient)
- Compatible with existing PUNetG weight loading

Key Components:
- FiLMConditioner: Per-pixel affine modulation (gamma * x + beta)
- SpatialConditionEncoder: Multi-scale conditioning preparation
- EnhancedConditioningWrapper: Wraps PUNetG with enhanced conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional

from . import commonlayers


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies per-pixel affine transformation: y = x * (1 + gamma) + beta

    Key design for post-training:
    - gamma initialized to 0, beta initialized to 0
    - At init: y = x * (1 + 0) + 0 = x (identity)
    - Gradients can then learn to modulate features

    Args:
        cond_channels: Number of conditioning channels
        feature_channels: Number of feature channels to modulate
        dimension: 2 for 2D, 3 for 3D
    """

    def __init__(
        self,
        cond_channels: int,
        feature_channels: int,
        dimension: int = 3,
        use_conv: bool = True
    ):
        super().__init__()
        self.dimension = dimension
        self.feature_channels = feature_channels

        if use_conv:
            # 1x1x1 conv to project conditioning to gamma/beta
            # Per-pixel, no spatial mixing = shift-invariant
            Conv = nn.Conv3d if dimension == 3 else nn.Conv2d
            self.gamma_proj = Conv(cond_channels, feature_channels, kernel_size=1, bias=True)
            self.beta_proj = Conv(cond_channels, feature_channels, kernel_size=1, bias=True)
        else:
            # Linear projection (for non-spatial conditioning)
            self.gamma_proj = nn.Linear(cond_channels, feature_channels)
            self.beta_proj = nn.Linear(cond_channels, feature_channels)

        self.use_conv = use_conv
        self._init_identity()

    def _init_identity(self):
        """Initialize to identity: gamma=0, beta=0 -> y = x * 1 + 0 = x"""
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, C, D, H, W] or [B, C, H, W]
            cond: Conditioning tensor [B, C_cond, D', H', W'] (spatial)
                  or [B, C_cond] (global)

        Returns:
            Modulated features: x * (1 + gamma) + beta
        """
        # Handle spatial dimension mismatch
        if cond.ndim > 2 and cond.shape[2:] != x.shape[2:]:
            # Interpolate conditioning to match feature resolution
            cond = F.interpolate(
                cond,
                size=x.shape[2:],
                mode='trilinear' if self.dimension == 3 else 'bilinear',
                align_corners=False
            )

        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)

        # Handle non-spatial conditioning
        if gamma.ndim == 2:
            # Expand to spatial dims
            for _ in range(self.dimension):
                gamma = gamma.unsqueeze(-1)
                beta = beta.unsqueeze(-1)

        return x * (1.0 + gamma) + beta


class SpatialConditionEncoder(nn.Module):
    """
    Multi-scale spatial conditioning encoder.

    Takes a porosity field and produces embeddings at multiple scales
    for injection at different UNet levels.

    All operations are per-pixel (1x1x1 convs) = shift-invariant.

    Args:
        input_channels: Input conditioning channels (1 for porosity)
        embed_dim: Base embedding dimension
        scales: List of spatial scales (e.g., [32, 16, 8, 4] for UNet levels)
        dimension: 2 or 3
    """

    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 64,
        scales: list[int] = [32, 16, 8, 4],
        dimension: int = 3,
        fourier_scale: float = 30.0,
        use_gradient: bool = True
    ):
        super().__init__()
        self.scales = scales
        self.dimension = dimension
        self.embed_dim = embed_dim
        self.use_gradient = use_gradient

        # Gaussian Fourier projection for porosity values
        self.fourier_proj = commonlayers.GaussianFourierProjection(
            embed_dim=embed_dim,
            scale=fourier_scale
        )

        # Per-scale embedding networks (1x1x1 convs = per-pixel, shift-invariant)
        Conv = nn.Conv3d if dimension == 3 else nn.Conv2d

        # Input channels: embed_dim (from Fourier) + optional 3 for gradient
        proj_input_dim = embed_dim + (3 if use_gradient and dimension == 3 else 0)
        if use_gradient and dimension == 2:
            proj_input_dim = embed_dim + 2

        self.scale_projectors = nn.ModuleDict()
        for scale in scales:
            self.scale_projectors[str(scale)] = nn.Sequential(
                Conv(proj_input_dim, embed_dim * 2, kernel_size=1),
                nn.SiLU(),
                Conv(embed_dim * 2, embed_dim * 2, kernel_size=1),
                nn.SiLU(),
                Conv(embed_dim * 2, embed_dim, kernel_size=1),
            )

    def compute_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient magnitude (per-pixel operation)."""
        if self.dimension == 3:
            # [B, 1, D, H, W]
            grad_d = F.pad(field[:, :, 1:] - field[:, :, :-1], (0, 0, 0, 0, 0, 1))
            grad_h = F.pad(field[:, :, :, 1:] - field[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
            grad_w = F.pad(field[:, :, :, :, 1:] - field[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
            return torch.cat([grad_d, grad_h, grad_w], dim=1)  # [B, 3, D, H, W]
        else:
            grad_h = F.pad(field[:, :, 1:] - field[:, :, :-1], (0, 0, 0, 1))
            grad_w = F.pad(field[:, :, :, 1:] - field[:, :, :, :-1], (0, 1, 0, 0))
            return torch.cat([grad_h, grad_w], dim=1)

    def forward(self, porosity: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Args:
            porosity: Porosity field [B, D, H, W] or [B, 1, D, H, W]

        Returns:
            Dict mapping scale -> embedding tensor [B, embed_dim, scale, scale, scale]
        """
        # Ensure [B, 1, D, H, W]
        if porosity.ndim == 4:
            porosity = porosity.unsqueeze(1)

        B = porosity.shape[0]
        embeddings = {}

        for scale in self.scales:
            # Downsample to target scale
            if self.dimension == 3:
                target_size = (scale, scale, scale)
                mode = 'trilinear'
            else:
                target_size = (scale, scale)
                mode = 'bilinear'

            porosity_scaled = F.interpolate(
                porosity, size=target_size, mode=mode, align_corners=False
            )

            # Flatten for Fourier projection
            # [B, 1, D, H, W] -> [B*D*H*W, 1] -> Fourier -> [B*D*H*W, embed_dim]
            flat = porosity_scaled.squeeze(1).flatten()  # [B*D*H*W]
            fourier = self.fourier_proj(flat)  # [B*D*H*W, embed_dim]

            # Reshape back: [B, embed_dim, D, H, W]
            if self.dimension == 3:
                fourier = fourier.view(B, scale, scale, scale, self.embed_dim)
                fourier = fourier.permute(0, 4, 1, 2, 3)
            else:
                fourier = fourier.view(B, scale, scale, self.embed_dim)
                fourier = fourier.permute(0, 3, 1, 2)

            # Optionally add gradient features
            if self.use_gradient:
                grad = self.compute_gradient(porosity_scaled)
                fourier = torch.cat([fourier, grad], dim=1)

            # Project to final embedding
            embeddings[scale] = self.scale_projectors[str(scale)](fourier)

        return embeddings


class ConditionAmplifier(nn.Module):
    """
    Simple conditioning amplifier with learnable per-channel scales.

    Allows the model to learn how much to weight conditioning vs time.
    Initialized to 1.0 (identity).
    """

    def __init__(self, channels: int, init_scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels) * init_scale)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim > 2:
            # Spatial conditioning: expand scale to match
            scale = self.scale
            for _ in range(cond.ndim - 2):
                scale = scale.unsqueeze(-1)
            return cond * scale
        return cond * self.scale


class EnhancedConditioningWrapper(nn.Module):
    """
    Wrapper that adds enhanced conditioning to an existing PUNetG model.

    Designed for post-training:
    - Loads existing PUNetG weights unchanged
    - Adds FiLM layers initialized to identity
    - Adds separate conditioning pathway
    - Multi-scale conditioning injection

    Args:
        base_model: Existing PUNetG model with trained weights
        condition_embed_dim: Dimension for conditioning embeddings
        use_film: Whether to use FiLM modulation
        use_multiscale: Whether to use multi-scale conditioning
        use_gradient: Whether to include porosity gradient features
        condition_amplification: Initial amplification factor for conditioning
    """

    def __init__(
        self,
        base_model: nn.Module,
        condition_embed_dim: int = 64,
        use_film: bool = True,
        use_multiscale: bool = True,
        use_gradient: bool = True,
        condition_amplification: float = 1.0,
        film_injection_points: list[str] = ['encoder', 'decoder'],
        cond_drop_p: float = 0.1,
    ):
        super().__init__()

        # Store base model (with existing weights)
        self.base_model = base_model
        self.config = base_model.config
        self.use_film = use_film
        self.use_multiscale = use_multiscale

        model_channels = self.config.model_channels
        channel_expansion = self.config.extended_channel_expansion
        dimension = self.config.dimension

        # Compute scales from channel expansion
        # Typically: [1, 2, 4] -> scales at 32, 16, 8, 4 for 256^3 input
        n_levels = len(channel_expansion)
        # Assume base latent is 32^3, each level halves
        base_scale = 32
        self.scales = [base_scale // (2**i) for i in range(n_levels + 1)]
        # Filter to valid scales (> 0)
        self.scales = [s for s in self.scales if s > 0]

        # Condition amplifier
        self.condition_amplifier = ConditionAmplifier(
            model_channels, init_scale=condition_amplification
        )

        # Multi-scale conditioning encoder
        if use_multiscale:
            self.condition_encoder = SpatialConditionEncoder(
                input_channels=1,
                embed_dim=condition_embed_dim,
                scales=self.scales,
                dimension=dimension,
                use_gradient=use_gradient
            )
        else:
            self.condition_encoder = None

        # FiLM layers at each scale
        if use_film:
            self.film_layers = nn.ModuleDict()
            for i, mult in enumerate(channel_expansion):
                feat_channels = model_channels * mult
                scale = self.scales[min(i, len(self.scales) - 1)]
                self.film_layers[f'enc_{i}'] = FiLMLayer(
                    cond_channels=condition_embed_dim,
                    feature_channels=feat_channels,
                    dimension=dimension
                )
                self.film_layers[f'dec_{i}'] = FiLMLayer(
                    cond_channels=condition_embed_dim,
                    feature_channels=feat_channels,
                    dimension=dimension
                )
            # Bottleneck FiLM
            bottleneck_channels = model_channels * channel_expansion[-1]
            self.film_layers['bottleneck'] = FiLMLayer(
                cond_channels=condition_embed_dim,
                feature_channels=bottleneck_channels,
                dimension=dimension
            )
        else:
            self.film_layers = None

        # Classifier-free guidance dropout for conditioning
        if cond_drop_p > 0:
            self.cond_drop = commonlayers.ConditionDrop(
                p=cond_drop_p,
                hidden_dim=condition_embed_dim,
                null_is_learnable=True
            )
        else:
            self.cond_drop = None

        # Override base model's forward to inject our conditioning
        self._setup_hooks()

    def _setup_hooks(self):
        """Setup forward hooks for FiLM injection without modifying base model."""
        self._film_embeddings = None  # Will be set during forward
        self._current_level = 0

    def forward(self, x, t=None, y=None):
        """
        Enhanced forward pass with FiLM conditioning.

        Args:
            x: Input tensor
            t: Timestep
            y: Conditioning dict with 'porosity' key
        """
        # Prepare multi-scale conditioning embeddings
        if y is not None and 'porosity' in y and self.condition_encoder is not None:
            porosity = y['porosity']
            self._film_embeddings = self.condition_encoder(porosity)

            # Apply CFG dropout
            if self.cond_drop is not None and self.training:
                for scale in self._film_embeddings:
                    emb = self._film_embeddings[scale]
                    # Reshape for ConditionDrop: [B, C, D, H, W] -> [B, C, -1] -> dropout -> reshape back
                    B, C = emb.shape[:2]
                    spatial = emb.shape[2:]
                    emb_flat = emb.view(B, C, -1).transpose(1, 2)  # [B, N, C]
                    emb_flat = self.cond_drop(emb_flat)
                    self._film_embeddings[scale] = emb_flat.transpose(1, 2).view(B, C, *spatial)
        else:
            self._film_embeddings = None

        # Use base model's forward, but we'll intercept via our enhanced encode/decode
        return self._enhanced_forward(x, t, y)

    def _enhanced_forward(self, x, t=None, y=None):
        """Forward with FiLM injection at each level."""
        config = self.config
        base = self.base_model

        # Input processing (same as base)
        if not config.bias:
            xe_shape = list(x.shape)
            xe_shape[1] = 1
            xe = torch.ones(xe_shape, device=x.device, dtype=x.dtype)
            x = torch.cat([x, xe], dim=1)

        x = base.convin(x)

        # Time embedding
        if t is not None:
            te = base.time_projection(t)
        else:
            te = torch.zeros(x.shape[0], config.model_channels, device=x.device, dtype=x.dtype)

        # Conditional embedding (original path)
        if y is not None:
            if base.conditional_embedding is not None:
                ye = base.conditional_embedding(y)
            else:
                ye = None

            if ye is not None:
                # Apply our amplifier
                ye = self.condition_amplifier(ye)

                if ye.ndim > te.ndim:
                    new_te_shape = list(te.shape) + [1] * (ye.ndim - te.ndim)
                    te = te.reshape(new_te_shape)

                if base.cond_drop is not None:
                    ye = base.cond_drop(ye)

                te = te + base.cond_dropout(ye)

        # Enhanced encode with FiLM
        x, intermediate_outputs = self._enhanced_encode(x, te)

        # Enhanced bottleneck with FiLM
        x = self._enhanced_bottom_forward(x, te)

        # Enhanced decode with FiLM
        x = self._enhanced_decode(x, te, intermediate_outputs)

        x = base.convout(x)
        return x

    def _enhanced_encode(self, x, te):
        """Encode with FiLM injection."""
        base = self.base_model
        intermediate_outputs = []

        for i, (resnet_block, downsampler) in enumerate(
            zip(base.downward_blocks, base.downsamplers)
        ):
            x = base.resnet_block_forward(x, te, resnet_block)

            # Apply FiLM
            if self.use_film and self._film_embeddings is not None:
                scale = self._get_scale_for_level(x)
                if scale in self._film_embeddings:
                    x = self.film_layers[f'enc_{i}'](x, self._film_embeddings[scale])

            intermediate_outputs.append(x.clone())
            x = downsampler(x)

        return x, intermediate_outputs

    def _enhanced_bottom_forward(self, x, te):
        """Bottleneck with FiLM injection."""
        base = self.base_model

        x = base.resnet_block_forward(x, te, base.before_block)

        # FiLM at bottleneck
        if self.use_film and self._film_embeddings is not None:
            scale = self._get_scale_for_level(x)
            if scale in self._film_embeddings:
                x = self.film_layers['bottleneck'](x, self._film_embeddings[scale])

        xa = base.resnet_attn_block_forward(x, te, base.attn_resnet_block, base.attn_block)
        x = x + xa
        x = base.resnet_block_forward(x, te, base.after_block)

        return x

    def _enhanced_decode(self, x, te, intermediate_outputs):
        """Decode with FiLM injection."""
        base = self.base_model

        for i, (resnet_block, upsampler) in enumerate(
            zip(base.upward_blocks, base.upsamplers)
        ):
            x = upsampler(x)
            x = x + intermediate_outputs.pop()
            x = base.resnet_block_forward(x, te, resnet_block)

            # Apply FiLM
            if self.use_film and self._film_embeddings is not None:
                scale = self._get_scale_for_level(x)
                if scale in self._film_embeddings:
                    # Decoder levels go in reverse order
                    dec_idx = len(base.upward_blocks) - 1 - i
                    x = self.film_layers[f'dec_{dec_idx}'](x, self._film_embeddings[scale])

        return x

    def _get_scale_for_level(self, x: torch.Tensor) -> int:
        """Get the closest matching scale for the current feature resolution."""
        spatial_size = x.shape[2]  # Assuming cubic
        # Find closest scale
        for scale in sorted(self.scales, reverse=True):
            if spatial_size >= scale:
                return scale
        return self.scales[-1]

    def get_trainable_params(self) -> list:
        """Return only the NEW trainable parameters (not base model)."""
        params = []

        if self.condition_encoder is not None:
            params.extend(self.condition_encoder.parameters())

        if self.film_layers is not None:
            params.extend(self.film_layers.parameters())

        params.extend(self.condition_amplifier.parameters())

        if self.cond_drop is not None:
            params.extend(self.cond_drop.parameters())

        return params

    def get_all_params_with_lr_groups(
        self,
        base_lr: float,
        new_params_lr_mult: float = 10.0
    ) -> list[dict]:
        """
        Return parameter groups with different learning rates.

        Base model params get base_lr, new conditioning params get higher LR
        for faster adaptation.
        """
        return [
            {'params': self.base_model.parameters(), 'lr': base_lr},
            {'params': self.get_trainable_params(), 'lr': base_lr * new_params_lr_mult},
        ]


def wrap_model_with_enhanced_conditioning(
    model: nn.Module,
    condition_embed_dim: int = 64,
    use_film: bool = True,
    use_multiscale: bool = True,
    use_gradient: bool = True,
    condition_amplification: float = 1.0,
    cond_drop_p: float = 0.1,
) -> EnhancedConditioningWrapper:
    """
    Convenience function to wrap an existing model with enhanced conditioning.

    Args:
        model: Existing PUNetG with trained weights
        condition_embed_dim: Embedding dimension for conditioning
        use_film: Whether to use FiLM modulation
        use_multiscale: Whether to prepare conditioning at multiple scales
        use_gradient: Whether to include porosity gradient features
        condition_amplification: Initial conditioning scale (1.0 = no change)
        cond_drop_p: Probability of dropping conditioning for CFG training

    Returns:
        EnhancedConditioningWrapper ready for fine-tuning

    Example:
        >>> model = load_flow_model('Estaillades')
        >>> enhanced = wrap_model_with_enhanced_conditioning(model)
        >>> # Fine-tune with higher LR on new params
        >>> optimizer = Adam(enhanced.get_all_params_with_lr_groups(2e-5, 10.0))
    """
    return EnhancedConditioningWrapper(
        base_model=model,
        condition_embed_dim=condition_embed_dim,
        use_film=use_film,
        use_multiscale=use_multiscale,
        use_gradient=use_gradient,
        condition_amplification=condition_amplification,
        cond_drop_p=cond_drop_p,
    )

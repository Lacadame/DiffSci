from typing import List
import pathlib
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class DASCConfig:
    """Configuration class for DASC video clustering network."""

    def __init__(
        self,
        dimension: int = 2,                    # Spatial dimensions (typically 2 for video frames)
        in_channels: int = 3,                  # RGB channels
        frame_height: int = 48,                # Frame height
        frame_width: int = 42,                 # Frame width
        frames_per_video: int = 10,            # Number of frames sampled per video
        latent_dim: int = 128,                 # Latent feature dimension (d in paper)
        num_videos: int = 100,                 # Number of videos in dataset (n in paper)
        num_clusters: int = 10,                # Number of clusters (k in paper)
        # Encoder/Decoder architecture
        encoder_channels: List[int] = [32, 64, 128],  # Channel progression
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        # Video Modeling Module
        vmm_hidden_dim: int = 128,
        vmm_num_layers: int = 2,               # Number of attention layers
        # Self-representation Module
        srm_lambda1: float = 1.0,              # Sparsity regularization
        srm_lambda2: float = 1.0,              # Reconstruction weight
        # Training
        dropout: float = 0.0,
        use_skip_connections: bool = True,
    ):
        assert dimension in [2, 3], f"Dimension must be 2 or 3, got {dimension}"

        self.dimension = dimension
        self.in_channels = in_channels
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frames_per_video = frames_per_video
        self.latent_dim = latent_dim
        self.num_videos = num_videos
        self.num_clusters = num_clusters
        self.encoder_channels = encoder_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.vmm_hidden_dim = vmm_hidden_dim
        self.vmm_num_layers = vmm_num_layers
        self.srm_lambda1 = srm_lambda1
        self.srm_lambda2 = srm_lambda2
        self.dropout = dropout
        self.use_skip_connections = use_skip_connections

    def export_description(self) -> dict:
        """Export configuration as a dictionary."""
        return {
            "dimension": self.dimension,
            "in_channels": self.in_channels,
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "frames_per_video": self.frames_per_video,
            "latent_dim": self.latent_dim,
            "num_videos": self.num_videos,
            "num_clusters": self.num_clusters,
            "encoder_channels": self.encoder_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "vmm_hidden_dim": self.vmm_hidden_dim,
            "vmm_num_layers": self.vmm_num_layers,
            "srm_lambda1": self.srm_lambda1,
            "srm_lambda2": self.srm_lambda2,
            "dropout": self.dropout,
            "use_skip_connections": self.use_skip_connections,
        }

    @classmethod
    def from_description(cls, description: dict):
        return cls(**description)

    @classmethod
    def from_config_file(cls, config_file: pathlib.Path | str):
        with open(config_file, "r") as f:
            description = yaml.safe_load(f)
        return cls.from_description(description)


class DimensionHelper:
    """Helper class for dimension-specific operations (similar to VAENet)."""

    @staticmethod
    def get_conv_cls(dimension):
        if dimension == 2:
            return nn.Conv2d
        elif dimension == 3:
            return nn.Conv3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    @staticmethod
    def get_convtranspose_cls(dimension):
        if dimension == 2:
            return nn.ConvTranspose2d
        elif dimension == 3:
            return nn.ConvTranspose3d
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")


class AutoEncoderBackbone(nn.Module):
    """Auto-encoder backbone for frame-level feature extraction."""

    def __init__(self, config: DASCConfig):
        super().__init__()
        self.config = config

        Conv = DimensionHelper.get_conv_cls(config.dimension)
        ConvTranspose = DimensionHelper.get_convtranspose_cls(config.dimension)

        # Encoder layers
        encoder_layers = []
        in_ch = config.in_channels
        for out_ch in config.encoder_channels:
            encoder_layers.append(
                Conv(in_ch, out_ch, config.kernel_size, config.stride, config.padding)
            )
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                encoder_layers.append(nn.Dropout2d(config.dropout))
            in_ch = out_ch

        # Final encoder layer to latent dimension
        encoder_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(config.encoder_channels[-1], config.latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []

        # Project back from latent dimension
        decoder_layers.append(nn.Linear(config.latent_dim, config.encoder_channels[-1] * 4 * 4))
        decoder_layers.append(nn.ReLU(inplace=True))
        decoder_layers.append(nn.Unflatten(1, (config.encoder_channels[-1], 4, 4)))

        # Transposed convolutions
        reversed_channels = list(reversed(config.encoder_channels))
        for i in range(len(reversed_channels) - 1):
            decoder_layers.append(
                ConvTranspose(reversed_channels[i], reversed_channels[i + 1], config.kernel_size, config.stride,
                              config.padding, output_padding=1)
            )
            decoder_layers.append(nn.ReLU(inplace=True))

        # Final reconstruction layer
        decoder_layers.append(
            ConvTranspose(reversed_channels[-1], config.in_channels, config.kernel_size, config.stride,
                          config.padding, output_padding=1)
        )
        # Adjust to match input size
        decoder_layers.append(nn.AdaptiveAvgPool2d((config.frame_height, config.frame_width)))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode frames to latent features."""
        # x shape: [batch_size * n_frames, channels, height, width]
        return self.encoder(x)

    def decode(self, z):
        """Decode latent features to reconstructed frames."""
        return self.decoder(z)


class VideoModelingModule(nn.Module):
    """Video Modeling Module with attention mechanism."""

    def __init__(self, config: DASCConfig):
        super().__init__()
        self.config = config

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, config.latent_dim))

        # Linear transformations for multi-layer attention
        self.attention_layers = nn.ModuleList([
            nn.Linear(config.latent_dim, config.latent_dim)
            for _ in range(config.vmm_num_layers - 1)
        ])

    def forward(self, frame_features):
        """
        Args:
            frame_features: [batch_size, n_frames, latent_dim]
        Returns:
            video_features: [batch_size, latent_dim]
        """
        batch_size, n_frames, latent_dim = frame_features.shape

        # Initial query expanded for batch
        q = self.query.expand(batch_size, -1)  # [batch_size, latent_dim]

        # First attention layer
        scores = torch.bmm(frame_features, q.unsqueeze(-1)).squeeze(-1)  # [batch_size, n_frames]
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted aggregation
        video_repr = torch.bmm(attention_weights.unsqueeze(1), frame_features).squeeze(1)

        # Additional attention layers
        for layer in self.attention_layers:
            # Update query
            q = torch.tanh(layer(video_repr))

            # Compute new attention scores
            scores = torch.bmm(frame_features, q.unsqueeze(-1)).squeeze(-1)
            attention_weights = F.softmax(scores, dim=-1)

            # Update video representation
            video_repr = torch.bmm(attention_weights.unsqueeze(1), frame_features).squeeze(1)

        return video_repr, attention_weights


class SelfRepresentationModule(nn.Module):
    """Self-representation module for subspace clustering."""

    def __init__(self, config: DASCConfig):
        super().__init__()
        self.config = config

        # Self-representation layer (without bias and activation)
        # This acts as the coefficient matrix A
        self.self_repr = nn.Linear(config.num_videos, config.num_videos, bias=False)

        # Initialize to promote self-expression
        nn.init.xavier_uniform_(self.self_repr.weight)

    def forward(self, O):
        """
        Args:
            O: [num_videos, latent_dim] - all video features
        Returns:
            OA: [num_videos, latent_dim] - self-represented features
            A: [num_videos, num_videos] - coefficient matrix
        """
        # Get coefficient matrix
        A = self.self_repr.weight

        # Zero diagonal constraint
        A = A - torch.diag(torch.diag(A))

        # Self-representation: O = OA + E
        OA = torch.matmul(A.T, O)  # [num_videos, latent_dim]

        return OA, A


class FeatureRecoveredModule(nn.Module):
    """Feature recovery module to restore frame-level features."""

    def __init__(self, config: DASCConfig):
        super().__init__()
        self.config = config
        self.use_skip = config.use_skip_connections

        if not self.use_skip:
            # Optional: Add a transformation layer when not using skip connections
            self.transform = nn.Linear(config.latent_dim, config.latent_dim)

    def forward(self, video_features, original_frame_features=None):
        """
        Args:
            video_features: [batch_size, latent_dim]
            original_frame_features: [batch_size, n_frames, latent_dim] (optional)
        Returns:
            recovered_features: [batch_size * n_frames, latent_dim]
        """
        batch_size, latent_dim = video_features.shape
        n_frames = self.config.frames_per_video

        # Replicate video features for each frame
        replicated = video_features.unsqueeze(1).expand(-1, n_frames, -1)

        if self.use_skip and original_frame_features is not None:
            # Add skip connections
            recovered = replicated + original_frame_features
        else:
            recovered = replicated
            if hasattr(self, 'transform'):
                recovered = self.transform(recovered)

        # Reshape for decoder input
        recovered = recovered.reshape(batch_size * n_frames, latent_dim)

        return recovered


class DASC(nn.Module):
    """Deep Aggregation Subspace Clustering network."""

    def __init__(self, config: DASCConfig):
        super().__init__()
        self.config = config

        # Four main components
        self.auto_encoder = AutoEncoderBackbone(config)
        self.vmm = VideoModelingModule(config)
        self.srm = SelfRepresentationModule(config)
        self.frm = FeatureRecoveredModule(config)

    def forward(self, x, all_videos_mode=False):
        """
        Args:
            x: Input video frames
               - Training mode: [batch_size, n_frames, channels, height, width]
               - All videos mode: [num_videos, n_frames, channels, height, width]
            all_videos_mode: If True, process all videos for self-representation

        Returns:
            Dict with:
                - reconstructed: Reconstructed frames
                - frame_features: Frame-level latent features
                - video_features: Video-level features
                - coefficient_matrix: Self-representation matrix (if all_videos_mode)
                - attention_weights: VMM attention weights
        """
        if all_videos_mode:
            batch_size = x.shape[0]
            assert batch_size == self.config.num_videos, \
                f"Expected {self.config.num_videos} videos, got {batch_size}"
        else:
            batch_size = x.shape[0]

        n_frames = x.shape[1]

        # Reshape for encoder: combine batch and frames dimensions
        x_reshaped = x.reshape(batch_size * n_frames, *x.shape[2:])

        # 1. Auto-encoder backbone: Extract frame-level features
        frame_features_flat = self.auto_encoder.encode(x_reshaped)
        frame_features = frame_features_flat.reshape(batch_size, n_frames, -1)

        # 2. Video Modeling Module: Aggregate to video-level features
        video_features, attention_weights = self.vmm(frame_features)

        outputs = {
            'frame_features': frame_features,
            'video_features': video_features,
            'attention_weights': attention_weights,
        }

        if all_videos_mode:
            # 3. Self-representation Module (only in all-videos mode)
            # video_features shape: [num_videos, latent_dim]
            self_repr_features, coeff_matrix = self.srm(video_features)
            outputs['coefficient_matrix'] = coeff_matrix
            outputs['self_represented_features'] = self_repr_features

            # 4. Feature Recovery Module
            recovered_features = self.frm(self_repr_features, frame_features)
        else:
            # In batch mode, skip SRM
            recovered_features = self.frm(video_features, frame_features)

        # 5. Decode to reconstruct frames
        reconstructed = self.auto_encoder.decode(recovered_features)
        reconstructed = reconstructed.reshape(batch_size, n_frames, *reconstructed.shape[1:])

        outputs['reconstructed'] = reconstructed

        return outputs

    def compute_loss(self, outputs, original_frames, stage='second'):
        """
        Compute loss based on training stage.

        Args:
            outputs: Model outputs dictionary
            original_frames: Original input frames
            stage: 'first' (reconstruction only) or 'second' (all losses)
        """
        losses = {}

        # Reconstruction loss (MSE)
        loss_mse = F.mse_loss(outputs['reconstructed'], original_frames)
        losses['mse'] = loss_mse

        total_loss = loss_mse

        if stage == 'second' and 'coefficient_matrix' in outputs:
            # Self-representation loss
            video_features = outputs['video_features']
            self_repr_features = outputs['self_represented_features']
            coeff_matrix = outputs['coefficient_matrix']

            # ||O - OA||_F^2
            loss_self_repr = F.mse_loss(self_repr_features, video_features)

            # Sparsity regularization ||A||_1
            loss_sparsity = torch.norm(coeff_matrix, p=1)

            losses['self_repr'] = loss_self_repr
            losses['sparsity'] = loss_sparsity

            # Total loss with weights
            total_loss = loss_mse + \
                self.config.srm_lambda2 * loss_self_repr + \
                self.config.srm_lambda1 * loss_sparsity

        losses['total'] = total_loss
        return losses

    def export_description(self) -> dict:
        """Export model configuration."""
        return {
            "config": self.config.export_description(),
            "model_type": "DASC",
        }

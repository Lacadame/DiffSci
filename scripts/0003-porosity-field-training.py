#!/usr/bin/env python
"""
Fine-tune a 3D latent diffusion model with porosity field conditioning.

=============================================================================
OVERVIEW
=============================================================================
This script fine-tunes a pretrained 3D diffusion model (e.g., Estaillades)
to generate volumes conditioned on LOCAL porosity fields (spatial conditioning).

The key difference from the original scalar-porosity conditioning:
- Original: Single scalar porosity value conditions the entire volume
- This script: 3D porosity field provides spatially-varying conditioning

Workflow:
1. Load a binary volume (1000x1000x1000) and its corresponding porosity field
   (also 1000x1000x1000, computed as a local convolution mean of the binary volume).

2. Split both volumes identically:
   - Training: first 700 slices along z-axis (1000x1000x700)
   - Validation: last 300 slices along z-axis (1000x1000x300)

3. During training:
   - Sample random 256^3 subvolumes from the binary volume
   - Crop the SAME location from the porosity field volume
   - Apply the SAME cube symmetry to both (48 symmetries of the cube)
   - Downsample the porosity subvolume using 3D avg_pool with window L (L=8 -> 32^3)
   - Pass the downsampled porosity as spatial conditioning to the network

The network learns to generate microstructures that match the local porosity field,
enabling controlled generation with spatially-varying porosity.

=============================================================================
KEY DESIGN DECISIONS
=============================================================================

1. SPATIAL CONDITIONING:
   The ScalarEmbedder in diffsci2.nets can handle multidimensional inputs.
   When given a 3D porosity field [B, D, H, W], it embeds each spatial position
   and returns [B, dembed, D, H, W], providing spatially-varying conditioning.

2. DOWNSAMPLING RATIONALE:
   - The autoencoder has a compression ratio of 8x (256 -> 32 latent)
   - The porosity field must match the latent spatial resolution
   - We use AvgPool3d with kernel_size=8 to match this compression

3. SYMMETRY AUGMENTATION:
   The 48 cube symmetries (rotations and reflections) provide data augmentation.
   CRITICAL: The same symmetry must be applied to both the volume AND the porosity
   field to maintain spatial correspondence.

4. TRAINING IN LATENT SPACE:
   Like the original training, we train the flow model in the VAE's latent space.
   The autoencoder is frozen and used only for encoding/decoding.

5. GRADIENT ACCUMULATION:
   For single-GPU training with limited memory, use batch_size=1 with
   accumulate_grad_batches=8 for an effective batch size of 8.

=============================================================================
USAGE
=============================================================================
    # Single GPU with gradient accumulation (memory efficient)
    python scripts/0003-porosity-field-training.py \\
        --stone Estaillades \\
        --devices 6 \\
        --batch-size 1 \\
        --accumulate-grad-batches 8

    # Multi-GPU DDP (faster, needs more GPUs)
    python scripts/0003-porosity-field-training.py \\
        --stone Estaillades \\
        --devices 0,1,2,3 \\
        --batch-size 2

=============================================================================
OPTIMIZER NOTES (Best Practices)
=============================================================================

We use AdamW with the following considerations:

1. LEARNING RATE: 2e-5 (10x lower than training from scratch)
   - Fine-tuning requires careful updates to not destroy pretrained features
   - If loss spikes, reduce further to 1e-5

2. WEIGHT DECAY: 0.01
   - Helps prevent overfitting during fine-tuning
   - AdamW decouples weight decay from gradient updates (better than L2 reg)

3. BETAS: (0.9, 0.999)
   - Standard Adam values work well for diffusion models
   - Some papers use (0.9, 0.95) for faster adaptation

4. GRADIENT CLIPPING: 1.0
   - Prevents gradient explosion during training
   - Essential for stable diffusion model training

5. WARMUP: Linear warmup over 1000 steps
   - Stabilizes early training when gradients are noisy
   - Especially important for fine-tuning

6. EMA (Exponential Moving Average): decay=0.9999
   - Maintains a smoothed version of model weights
   - Often produces better samples than raw trained weights
   - Standard practice for diffusion models

=============================================================================
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import lightning
import lightning.pytorch.callbacks as pl_callbacks
from torch.utils.data import Dataset, DataLoader

# Add the aux directory for model_loaders (contains load_flow_model, load_autoencoder)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
import diffsci2.data

# Import model loaders from the aux directory
# These functions handle the exact architecture and checkpoint loading
from model_loaders import load_flow_model, load_autoencoder


# =============================================================================
# DATA PATHS (Hardcoded for convenience, can be overridden via CLI)
# =============================================================================
# Raw binary volumes from Imperial College dataset
DATA_DIR = '/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/'
VOLUME_PATHS = {
    'Bentheimer': DATA_DIR + 'Bentheimer_1000c_3p0035um.raw',
    'Doddington': DATA_DIR + 'Doddington_1000c_2p6929um.raw',
    'Estaillades': DATA_DIR + 'Estaillades_1000c_3p31136um.raw',
    'Ketton': DATA_DIR + 'Ketton_1000c_3p00006um.raw',
}

# Porosity field volumes (precomputed local mean porosity)
# gpdata2: original format with full resolution info in filename
# gpdata4-129: new format with simpler naming (conv kernel 129)
POROSITY_BASE_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/'

POROSITY_PATHS_GPDATA2 = {
    'Bentheimer': 'gpdata2/bentheimer/Bentheimer_1000c_3p0035um_porosity_field_full.npy',
    'Doddington': 'gpdata2/doddington/Doddington_1000c_2p6929um_porosity_field_full.npy',
    'Estaillades': 'gpdata2/estaillades/Estaillades_1000c_3p31136um_porosity_field_full.npy',
    'Ketton': 'gpdata2/ketton/Ketton_1000c_3p00006um_porosity_field_full.npy',
}

POROSITY_PATHS_GPDATA4_129 = {
    'Bentheimer': 'gpdata4-129/bentheimer/bentheimer_porosity_field_full.npy',
    'Doddington': 'gpdata4-129/doddington/doddington_porosity_field_full.npy',
    'Estaillades': 'gpdata4-129/estaillades/estaillades_porosity_field_full.npy',
    'Ketton': 'gpdata4-129/ketton/ketton_porosity_field_full.npy',
}


# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================

class VolumeSubvolumeWithPorosityDataset(Dataset):
    """
    Dataset that samples paired subvolumes from a binary volume and its porosity field.

    This is a specialized version of VolumeSubvolumeDataset that:
    1. Takes both a binary volume and a porosity field volume
    2. Samples the SAME spatial location from both
    3. Applies the SAME cube symmetry to both
    4. Downsamples the porosity field to match latent resolution

    Args:
        volume: Binary volume as numpy array [D, H, W]
        porosity_volume: Porosity field as numpy array [D, H, W] (same shape as volume)
        dataset_size: Number of samples per epoch (arbitrary, defines __len__)
        subvolume_size: Size of cubic subvolumes to extract (default 256)
        downsample_factor: Factor to downsample porosity (default 8, matching VAE compression)
        cube_symmetry: CubeSymmetry instance for data augmentation (optional)

    Returns:
        dict with:
            'x': Binary subvolume tensor [1, D, H, W]
            'y': dict containing:
                'porosity': Downsampled porosity field [D//L, H//L, W//L]
    """

    def __init__(
        self,
        volume: np.ndarray,
        porosity_volume: np.ndarray,
        dataset_size: int,
        subvolume_size: int = 256,
        downsample_factor: int = 8,
        cube_symmetry: diffsci2.data.CubeSymmetry | None = None
    ):
        # Validate that volumes have the same shape
        assert volume.shape == porosity_volume.shape, (
            f"Volume shape {volume.shape} != Porosity shape {porosity_volume.shape}"
        )

        self.volume = volume
        self.porosity_volume = porosity_volume
        self.dataset_size = dataset_size
        self.subvolume_size = subvolume_size
        self.downsample_factor = downsample_factor
        self.cube_symmetry = cube_symmetry

        # Random number generator for reproducibility
        self.rng = np.random.default_rng()

        # Precompute the valid sampling ranges
        # (where we can extract a full subvolume without going out of bounds)
        self.max_d = volume.shape[0] - subvolume_size
        self.max_h = volume.shape[1] - subvolume_size
        self.max_w = volume.shape[2] - subvolume_size

        assert self.max_d >= 0 and self.max_h >= 0 and self.max_w >= 0, (
            f"Volume {volume.shape} too small for subvolume size {subvolume_size}"
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Sample a random subvolume with matched porosity conditioning.

        The key insight is that we sample ONCE and use the SAME indices
        for both the binary volume and the porosity field.
        """
        # =====================================================================
        # STEP 1: Sample random position (same for both volumes)
        # =====================================================================
        start_d = self.rng.integers(0, self.max_d + 1) if self.max_d > 0 else 0
        start_h = self.rng.integers(0, self.max_h + 1) if self.max_h > 0 else 0
        start_w = self.rng.integers(0, self.max_w + 1) if self.max_w > 0 else 0

        # =====================================================================
        # STEP 2: Extract subvolumes from the SAME location
        # =====================================================================
        subvolume = self.volume[
            start_d:start_d + self.subvolume_size,
            start_h:start_h + self.subvolume_size,
            start_w:start_w + self.subvolume_size
        ].copy()  # copy() for contiguous memory

        porosity_subvolume = self.porosity_volume[
            start_d:start_d + self.subvolume_size,
            start_h:start_h + self.subvolume_size,
            start_w:start_w + self.subvolume_size
        ].copy()

        # =====================================================================
        # STEP 3: Convert to tensors
        # =====================================================================
        subvolume_tensor = torch.from_numpy(subvolume).float()
        porosity_tensor = torch.from_numpy(porosity_subvolume).float()

        # =====================================================================
        # STEP 4: Apply the SAME random symmetry to BOTH tensors
        # =====================================================================
        # This is critical! If we apply different symmetries, the spatial
        # correspondence between the binary volume and porosity field is lost.
        if self.cube_symmetry is not None:
            # Sample a random symmetry index (0-47)
            symmetry_id = self.rng.integers(0, 48)

            # Apply the SAME symmetry to both tensors
            subvolume_tensor = self.cube_symmetry.apply(subvolume_tensor, symmetry_id)
            porosity_tensor = self.cube_symmetry.apply(porosity_tensor, symmetry_id)

        # =====================================================================
        # STEP 5: Add channel dimension to subvolume
        # =====================================================================
        # Binary volume: [D, H, W] -> [1, D, H, W]
        subvolume_tensor = subvolume_tensor.unsqueeze(0)

        # =====================================================================
        # STEP 6: Downsample porosity field to match latent resolution
        # =====================================================================
        # The autoencoder compresses 256 -> 32 (8x), so we need the porosity
        # field at 32^3 resolution to match the latent space.
        #
        # We use avg_pool3d which requires [N, C, D, H, W] format
        # Add batch and channel dims: [D, H, W] -> [1, 1, D, H, W]
        porosity_for_pool = porosity_tensor.unsqueeze(0).unsqueeze(0)

        # Apply 3D average pooling with kernel_size = downsample_factor
        # This computes the mean over each L x L x L block
        porosity_downsampled = F.avg_pool3d(
            porosity_for_pool,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        )

        # Remove batch dim: [1, 1, D', H', W'] -> [D', H', W']
        # We keep it as [D', H', W'] because ScalarEmbedder expects this shape
        porosity_downsampled = porosity_downsampled.squeeze(0).squeeze(0)

        # =====================================================================
        # STEP 7: Return in the format expected by SIModule
        # =====================================================================
        return {
            'x': subvolume_tensor,
            'y': {
                'porosity': porosity_downsampled
            }
        }

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)


# =============================================================================
# MODEL LOADING (using model_loaders.py from aux/)
# =============================================================================
# Note: load_flow_model and load_autoencoder are imported from model_loaders.py
# They handle the exact architecture, checkpoint paths, and weight loading.
#
# The key insight for spatial conditioning:
# ScalarEmbedder can handle multidimensional inputs:
# - For scalar porosity [B], it returns [B, dembed]
# - For spatial porosity [B, D, H, W], it returns [B, dembed, D, H, W]
#
# This means the SAME model architecture works for both scalar and spatial
# conditioning - we just pass a 3D tensor instead of a scalar!


# =============================================================================
# LEARNING RATE SCHEDULER WITH WARMUP
# =============================================================================

class WarmupConstantCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup, constant plateau, then cosine decay to zero.

    Timeline (in optimizer steps):
        [0, warmup_steps)         -> linear warmup from 0 to base_lr
        [warmup_steps, decay_start) -> constant at base_lr
        [decay_start, total_steps]  -> cosine decay from base_lr to 0

    If cosine_decay_steps <= 0 or total_steps is None, there is no cosine
    phase and the scheduler stays constant after warmup.
    """

    def __init__(self, optimizer, warmup_steps, total_steps=None, cosine_decay_steps=0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine_decay_steps = cosine_decay_steps
        if total_steps is not None and cosine_decay_steps > 0:
            self.decay_start = total_steps - cosine_decay_steps
        else:
            self.decay_start = None
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = (step + 1) / self.warmup_steps
        elif self.decay_start is not None and step >= self.decay_start:
            progress = (step - self.decay_start) / max(1, self.cosine_decay_steps)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        else:
            factor = 1.0
        return [base_lr * factor for base_lr in self.base_lrs]


# =============================================================================
# EMA (EXPONENTIAL MOVING AVERAGE) CALLBACK
# =============================================================================

class EMACallback(pl_callbacks.Callback):
    """
    Exponential Moving Average of model weights.

    EMA maintains a smoothed version of the model weights:
        ema_weight = decay * ema_weight + (1 - decay) * current_weight

    Why EMA is important for diffusion models:
    - Training can be noisy, causing weight fluctuations
    - EMA weights average out these fluctuations
    - Usually produces higher quality samples than raw weights
    - decay=0.9999 is standard (slow averaging, very smooth)

    Usage:
    - Training uses regular weights (for gradient updates)
    - Inference/validation uses EMA weights (for best quality)
    """

    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema_weights = {}

    def on_train_start(self, trainer, pl_module):
        """Initialize EMA weights as copy of current weights."""
        for name, param in pl_module.model.named_parameters():
            self.ema_weights[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA weights after each training step."""
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self.ema_weights:
                    self.ema_weights[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def on_validation_start(self, trainer, pl_module):
        """Swap to EMA weights for validation."""
        self._swap_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        """Swap back to training weights after validation."""
        self._swap_weights(pl_module)

    def _swap_weights(self, pl_module):
        """Swap model weights with EMA weights."""
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self.ema_weights:
                    tmp = param.data.clone()
                    param.data.copy_(self.ema_weights[name])
                    self.ema_weights[name].copy_(tmp)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_devices(devices_str: str) -> list[int]:
    """Parse device string like '0,1,2,3' into list [0, 1, 2, 3]."""
    return [int(d.strip()) for d in devices_str.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune 3D diffusion model with porosity field conditioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Stone type (uses predefined paths)
    parser.add_argument('--stone', type=str, default='Estaillades',
                        choices=['Bentheimer', 'Doddington', 'Estaillades', 'Ketton'],
                        help='Stone type to use (determines data/model paths)')

    # Data source selection
    parser.add_argument('--data-source', type=str, default='gpdata2',
                        help='Porosity field data source (e.g., gpdata2, gpdata4-129, gpdata4-257)')

    # Center crop option
    parser.add_argument('--center-crop', type=int, default=None,
                        help='Center crop margin (e.g., 64 crops [64:-64, 64:-64, 64:-64] from volumes)')

    # Override paths (optional, will use defaults based on --stone if not specified)
    parser.add_argument('--data-path', type=str, default=None,
                        help='Override: Path to binary volume .raw file (uint8, 1000x1000x1000)')
    parser.add_argument('--porosity-path', type=str, default=None,
                        help='Override: Path to porosity field .npy file (float32, 1000x1000x1000)')
    parser.add_argument('--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000],
                        help='Shape of the volumes [D, H, W]')

    # Training parameters
    parser.add_argument('--subvolume-size', type=int, default=256,
                        help='Size of cubic subvolumes to extract')
    parser.add_argument('--downsample-factor', type=int, default=8,
                        help='Factor to downsample porosity field (should match VAE compression)')
    parser.add_argument('--dataset-size', type=int, default=int(2**12),
                        help='Number of samples per epoch')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU (use 1 with gradient accumulation for memory efficiency)')
    parser.add_argument('--accumulate-grad-batches', type=int, default=8,
                        help='Accumulate gradients over N batches (effective batch = batch_size * N)')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--train-split', type=int, default=700,
                        help='Number of z-slices for training (rest for validation)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Linear warmup over this many epochs, then constant LR (default: 2)')
    parser.add_argument('--cosine-decay-epochs', type=int, default=0,
                        help='Cosine decay to zero over the last N epochs (0 = no decay, constant LR)')
    parser.add_argument('--ema-decay', type=float, default=0.99,
                        help='EMA decay rate (0 to disable)')
    parser.add_argument('--no-validation', action='store_true',
                        help='Skip validation and use full volume for training')

    # Hardware
    parser.add_argument('--devices', type=str, default='0',
                        help='GPU devices, comma-separated (e.g., "0,1,2,3")')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints (default: models/experimental/YYYYMMDD-dfn-{stone}-porosity-field/)')
    parser.add_argument('--log-every-n-steps', type=int, default=10,
                        help='Log metrics every N steps')

    # TensorBoard profiling
    parser.add_argument('--profile', action='store_true',
                        help='Enable TensorBoard profiling (first few batches)')

    return parser.parse_args()


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # STEP 0: Resolve paths based on stone type and data source
    # =========================================================================
    stone = args.stone
    data_path = args.data_path or VOLUME_PATHS[stone]

    # Select porosity path based on data source
    if args.porosity_path:
        porosity_path = args.porosity_path
    elif args.data_source == 'gpdata2':
        porosity_path = POROSITY_BASE_DIR + POROSITY_PATHS_GPDATA2[stone]
    elif args.data_source == 'gpdata4-129':
        porosity_path = POROSITY_BASE_DIR + POROSITY_PATHS_GPDATA4_129[stone]
    else:
        # Generic pattern: {data_source}/{stone_lower}/{stone_lower}_porosity_field_full.npy
        sl = stone.lower()
        porosity_path = POROSITY_BASE_DIR + f'{args.data_source}/{sl}/{sl}_porosity_field_full.npy'

    # Default checkpoint directory with date
    if args.checkpoint_dir is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d')
        ds_suffix = f'-{args.data_source}' if args.data_source != 'gpdata2' else ''
        checkpoint_dir = f'models/experimental/{date_str}-dfn-{stone.lower()}{ds_suffix}-porosity-field'
    else:
        checkpoint_dir = args.checkpoint_dir

    print("=" * 60)
    print(f"Porosity Field Training: {stone}")
    print("=" * 60)
    print(f"  Data source: {args.data_source}")
    print(f"  Data path: {data_path}")
    print(f"  Porosity path: {porosity_path}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Center crop: {args.center_crop}")
    print(f"  Effective batch size: {args.batch_size} x {args.accumulate_grad_batches} = {args.batch_size * args.accumulate_grad_batches}")
    print()

    # =========================================================================
    # STEP 1: Load volumes
    # =========================================================================
    print("=" * 60)
    print("Loading volumes...")
    print("=" * 60)

    # Load binary volume (uint8, raw format)
    print(f"Loading binary volume from {data_path}")
    volume_data = np.fromfile(data_path, dtype=np.uint8).reshape(args.volume_shape)
    print(f"  Shape: {volume_data.shape}")
    print(f"  Dtype: {volume_data.dtype}")
    print(f"  Range: [{volume_data.min()}, {volume_data.max()}]")

    # Load porosity field (float32, .npy format)
    # The porosity field is precomputed as a local mean of the binary volume
    print(f"Loading porosity field from {porosity_path}")
    porosity_data = np.load(porosity_path)
    print(f"  Shape: {porosity_data.shape}")
    print(f"  Dtype: {porosity_data.dtype}")
    print(f"  Range: [{porosity_data.min():.4f}, {porosity_data.max():.4f}]")

    # Validate shapes match
    assert volume_data.shape == porosity_data.shape, (
        f"Volume shape {volume_data.shape} != Porosity shape {porosity_data.shape}"
    )

    # =========================================================================
    # STEP 1.5: Apply center crop if specified
    # =========================================================================
    if args.center_crop is not None:
        c = args.center_crop
        print(f"\nApplying center crop [{c}:-{c}, {c}:-{c}, {c}:-{c}]")
        print(f"  Original shape: {volume_data.shape}")
        volume_data = volume_data[c:-c, c:-c, c:-c]
        porosity_data = porosity_data[c:-c, c:-c, c:-c]
        print(f"  Cropped shape: {volume_data.shape}")

    # =========================================================================
    # STEP 2: Split volumes into train/val
    # =========================================================================
    if args.no_validation:
        print(f"\nNo validation — using full volume for training")
        volume_train = volume_data
        porosity_train = porosity_data
        volume_val = None
        porosity_val = None
        print(f"  Training volume: {volume_train.shape}")
    else:
        # Split along the first axis (z-axis)
        train_split = args.train_split
        print(f"\nSplitting volumes at z={train_split}")
        volume_train = volume_data[:, :, :train_split]
        volume_val = volume_data[:, :, train_split:]
        porosity_train = porosity_data[:, :, :train_split]
        porosity_val = porosity_data[:, :, train_split:]
        print(f"  Training volume: {volume_train.shape}")
        print(f"  Validation volume: {volume_val.shape}")

    # =========================================================================
    # STEP 3: Create datasets
    # =========================================================================
    print("\nCreating datasets...")

    # Cube symmetry for data augmentation (48 symmetries)
    cube_symmetry = diffsci2.data.CubeSymmetry()

    # Training dataset
    train_dataset = VolumeSubvolumeWithPorosityDataset(
        volume=volume_train,
        porosity_volume=porosity_train,
        dataset_size=args.dataset_size,
        subvolume_size=args.subvolume_size,
        downsample_factor=args.downsample_factor,
        cube_symmetry=cube_symmetry
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Avoid batch size issues with DDP
    )

    print(f"  Training samples per epoch: {len(train_dataset)}")

    if args.no_validation:
        val_loader = None
        print("  Validation: disabled")
    else:
        val_dataset = VolumeSubvolumeWithPorosityDataset(
            volume=volume_val,
            porosity_volume=porosity_val,
            dataset_size=args.dataset_size // 4,
            subvolume_size=args.subvolume_size,
            downsample_factor=args.downsample_factor,
            cube_symmetry=None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print(f"  Validation samples per epoch: {len(val_dataset)}")
    
    sample_data = next(iter(train_loader))
    print("Volume shape: ", sample_data['x'].shape)
    print("Porosity shape: ", sample_data['y']['porosity'].shape)

    # =========================================================================
    # STEP 4: Load models (using model_loaders.py)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    # Load pretrained flow model for this stone type
    # The model already has ScalarEmbedder which works for spatial conditioning too!
    print(f"Loading flow model for {stone}...")
    flow_model = load_flow_model(stone)
    print("  Flow model loaded successfully")

    # Load autoencoder (shared across all stone types)
    print("Loading VAE autoencoder...")
    autoencoder = load_autoencoder()
    print("  Autoencoder loaded successfully")

    # =========================================================================
    # STEP 5: Create flow module (SIModule)
    # =========================================================================
    # Use EDM-style sigma-space configuration (matching original training)
    flow_module_config = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        initial_norm=20.0,
        loss_formulation='denoiser'
    )

    flow_module = diffsci2.models.SIModule(
        config=flow_module_config,
        model=flow_model,
        autoencoder=autoencoder
    )

    # =========================================================================
    # STEP 6: Setup optimizer and scheduler
    # =========================================================================
    print("\nConfiguring optimizer...")

    # AdamW with weight decay (better than Adam + L2 regularization)
    optimizer = torch.optim.AdamW(
        flow_module.model.parameters(),  # Only train the flow model, not autoencoder
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8  # Numerical stability
    )

    # Compute optimizer steps per epoch (accounts for accumulation + DDP)
    batches_per_epoch = len(train_loader)
    num_devices = len(parse_devices(args.devices))
    # In DDP, each GPU processes a shard of the data
    batches_per_gpu = batches_per_epoch // num_devices
    optimizer_steps_per_epoch = batches_per_gpu // args.accumulate_grad_batches
    total_steps = args.max_epochs * optimizer_steps_per_epoch
    warmup_steps = args.warmup_epochs * optimizer_steps_per_epoch
    cosine_decay_steps = args.cosine_decay_epochs * optimizer_steps_per_epoch

    print(f"  Batches per epoch: {batches_per_epoch} total, {batches_per_gpu} per GPU ({num_devices} GPUs)")
    print(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch} (acc_grad={args.accumulate_grad_batches})")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup: {args.warmup_epochs} epochs = {warmup_steps} steps")
    if cosine_decay_steps > 0:
        print(f"  Cosine decay: last {args.cosine_decay_epochs} epochs = {cosine_decay_steps} steps (to zero)")
    else:
        print(f"  LR after warmup: constant")

    scheduler = WarmupConstantCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        cosine_decay_steps=cosine_decay_steps,
    )

    flow_module.set_optimizer_and_scheduler(
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval="step"
    )

    # =========================================================================
    # STEP 7: Setup callbacks
    # =========================================================================
    print("\nConfiguring callbacks...")
    callbacks = []

    # NaN gradient handling (converts NaN gradients to zero)
    callbacks.append(diffsci2.models.NanToZeroGradCallback())

    # Checkpoint saving
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.no_validation:
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
            save_top_k=0,
            save_last=True,
        )
    else:
        checkpoint_callback = pl_callbacks.ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
            filename='porosity-field-{epoch:03d}-{val_loss:.6f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True,
        )
    callbacks.append(checkpoint_callback)

    # EMA (if enabled)
    if args.ema_decay > 0:
        print(f"  EMA enabled with decay={args.ema_decay}")
        callbacks.append(EMACallback(decay=args.ema_decay))

    # Learning rate monitor (logs LR to TensorBoard)
    callbacks.append(pl_callbacks.LearningRateMonitor(logging_interval='step'))

    # =========================================================================
    # STEP 8: Setup trainer
    # =========================================================================
    print("\nConfiguring trainer...")

    devices = parse_devices(args.devices)
    strategy = 'ddp' if len(devices) > 1 else 'auto'

    print(f"  Devices: {devices}")
    print(f"  Strategy: {strategy}")
    print(f"  Gradient accumulation: {args.accumulate_grad_batches}")
    print(f"  Effective batch size: {args.batch_size * args.accumulate_grad_batches}")

    # Logger configuration
    tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=checkpoint_dir,
        name='logs',
        default_hp_metric=False
    )

    # Profiler (if enabled)
    profiler = None
    if args.profile:
        print("  TensorBoard profiling enabled")
        profiler = lightning.pytorch.profilers.PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(checkpoint_dir, 'profiler')
            ),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            profile_memory=True,
            with_stack=True
        )

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=checkpoint_dir,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        enable_checkpointing=True,
        devices=devices,
        strategy=strategy,
        logger=tb_logger,
        log_every_n_steps=args.log_every_n_steps,
        profiler=profiler,
        precision='16-mixed',  # Mixed precision for faster training and less memory
        accumulate_grad_batches=args.accumulate_grad_batches,  # Gradient accumulation for effective batch size
        check_val_every_n_epoch=1
    )

    # =========================================================================
    # STEP 9: Train!
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"\nTensorBoard logs: {checkpoint_dir}/logs")
    print(f"Checkpoints: {checkpoint_dir}/checkpoints")
    print("\nTo monitor training:")
    print(f"  tensorboard --logdir {checkpoint_dir}/logs")
    print()

    trainer.fit(
        flow_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    if not args.no_validation:
        print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")


if __name__ == '__main__':
    main()

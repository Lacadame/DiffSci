#!/usr/bin/env python
"""
Fine-tune a 3D latent diffusion model with VALID porosity field conditioning.

=============================================================================
OVERVIEW
=============================================================================
This is a corrected version of 0003-porosity-field-training.py that uses
valid-convolution porosity fields instead of same-convolution (full) fields.

The problem with 0003:
  The porosity field was computed with "same" padding (the _full.npy files),
  meaning boundary/corner voxels have porosity values computed from zero-padded
  regions that are NOT part of the training data. This introduces a systematic
  artifact at the edges of the porosity field.

The fix in 0003c:
  Use the valid-convolution porosity field (the non-_full .npy files) and crop
  the binary volume by [r:-r, r:-r, r:-r] (where r = convolution radius) so
  that both volumes match exactly. Every porosity value now corresponds to a
  region fully contained in the original data.

The script is parameterized by --radius r (default 64). This determines:
  - kernel_size = 2*r + 1 (e.g., r=64 -> kernel 129)
  - porosity data directory: gpdata4-{kernel_size}/
  - binary volume crop: [r:-r, r:-r, r:-r]
  - for r=64 on a 1000^3 volume, the result is 872^3

=============================================================================
USAGE
=============================================================================
    python scripts/0003c-porosity-field-training-vol-correction.py \\
        --stone Estaillades \\
        --radius 64 \\
        --devices 6 \\
        --batch-size 1 \\
        --accumulate-grad-batches 8

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
import diffsci2.data

from model_loaders import load_flow_model, load_autoencoder


# =============================================================================
# DATA PATHS
# =============================================================================
DATA_DIR = '/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/'
VOLUME_PATHS = {
    'Bentheimer': DATA_DIR + 'Bentheimer_1000c_3p0035um.raw',
    'Doddington': DATA_DIR + 'Doddington_1000c_2p6929um.raw',
    'Estaillades': DATA_DIR + 'Estaillades_1000c_3p31136um.raw',
    'Ketton': DATA_DIR + 'Ketton_1000c_3p00006um.raw',
}

POROSITY_BASE_DIR = '/home/ubuntu/repos/DiffSci2/notebooks/exploratory/dfn/data/'

# Porosity path pattern: gpdata4-{kernel_size}/{stone_lower}/{stone_lower}_porosity_field.npy
# (the valid-convolution version, NOT _full)
STONE_LOWER = {
    'Bentheimer': 'bentheimer',
    'Doddington': 'doddington',
    'Estaillades': 'estaillades',
    'Ketton': 'ketton',
}


def get_valid_porosity_path(stone: str, radius: int) -> str:
    kernel_size = 2 * radius + 1
    sl = STONE_LOWER[stone]
    return f'{POROSITY_BASE_DIR}gpdata4-{kernel_size}/{sl}/{sl}_porosity_field.npy'


# =============================================================================
# DATASET CLASS (same as 0003)
# =============================================================================

class VolumeSubvolumeWithPorosityDataset(Dataset):
    def __init__(
        self,
        volume: np.ndarray,
        porosity_volume: np.ndarray,
        dataset_size: int,
        subvolume_size: int = 256,
        downsample_factor: int = 8,
        cube_symmetry: diffsci2.data.CubeSymmetry | None = None
    ):
        assert volume.shape == porosity_volume.shape, (
            f"Volume shape {volume.shape} != Porosity shape {porosity_volume.shape}"
        )

        self.volume = volume
        self.porosity_volume = porosity_volume
        self.dataset_size = dataset_size
        self.subvolume_size = subvolume_size
        self.downsample_factor = downsample_factor
        self.cube_symmetry = cube_symmetry
        self.rng = np.random.default_rng()

        self.max_d = volume.shape[0] - subvolume_size
        self.max_h = volume.shape[1] - subvolume_size
        self.max_w = volume.shape[2] - subvolume_size
        assert self.max_d >= 0 and self.max_h >= 0 and self.max_w >= 0, (
            f"Volume {volume.shape} too small for subvolume size {subvolume_size}"
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        start_d = self.rng.integers(0, self.max_d + 1) if self.max_d > 0 else 0
        start_h = self.rng.integers(0, self.max_h + 1) if self.max_h > 0 else 0
        start_w = self.rng.integers(0, self.max_w + 1) if self.max_w > 0 else 0

        subvolume = self.volume[
            start_d:start_d + self.subvolume_size,
            start_h:start_h + self.subvolume_size,
            start_w:start_w + self.subvolume_size
        ].copy()

        porosity_subvolume = self.porosity_volume[
            start_d:start_d + self.subvolume_size,
            start_h:start_h + self.subvolume_size,
            start_w:start_w + self.subvolume_size
        ].copy()

        subvolume_tensor = torch.from_numpy(subvolume).float()
        porosity_tensor = torch.from_numpy(porosity_subvolume).float()

        if self.cube_symmetry is not None:
            symmetry_id = self.rng.integers(0, 48)
            subvolume_tensor = self.cube_symmetry.apply(subvolume_tensor, symmetry_id)
            porosity_tensor = self.cube_symmetry.apply(porosity_tensor, symmetry_id)

        subvolume_tensor = subvolume_tensor.unsqueeze(0)

        porosity_for_pool = porosity_tensor.unsqueeze(0).unsqueeze(0)
        porosity_downsampled = F.avg_pool3d(
            porosity_for_pool,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        ).squeeze(0).squeeze(0)

        return {
            'x': subvolume_tensor,
            'y': {'porosity': porosity_downsampled}
        }


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then constant LR. Works in epochs when scheduler_interval='epoch'."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


# =============================================================================
# EMA CALLBACK
# =============================================================================

class EMACallback(pl_callbacks.Callback):
    def __init__(self, decay: float = 0.9999):
        super().__init__()
        self.decay = decay
        self.ema_weights = {}

    def on_train_start(self, trainer, pl_module):
        for name, param in pl_module.model.named_parameters():
            self.ema_weights[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for name, param in pl_module.model.named_parameters():
                if name in self.ema_weights:
                    self.ema_weights[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def on_validation_start(self, trainer, pl_module):
        self._swap_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap_weights(pl_module)

    def _swap_weights(self, pl_module):
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
    return [int(d.strip()) for d in devices_str.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune 3D diffusion model with valid porosity field conditioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--stone', type=str, default='Estaillades',
                        choices=['Bentheimer', 'Doddington', 'Estaillades', 'Ketton'])

    parser.add_argument('--radius', type=int, default=64,
                        help='Convolution radius r. Kernel size = 2r+1. '
                             'Loads from gpdata4-{2r+1}/ and crops binary volume by [r:-r]. '
                             'Default: 64 (kernel 129, volume 1000->872)')

    parser.add_argument('--data-path', type=str, default=None,
                        help='Override: Path to binary volume .raw file')
    parser.add_argument('--porosity-path', type=str, default=None,
                        help='Override: Path to valid porosity field .npy file')
    parser.add_argument('--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000])

    parser.add_argument('--subvolume-size', type=int, default=256)
    parser.add_argument('--downsample-factor', type=int, default=8)
    parser.add_argument('--dataset-size', type=int, default=int(2**12))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulate-grad-batches', type=int, default=8)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--train-split', type=int, default=600,
                        help='Train/val split along last axis (default: 600, for 872^3 volumes)')

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Linear warmup over this many epochs, then constant LR')
    parser.add_argument('--ema-decay', type=float, default=0.9999)

    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--log-every-n-steps', type=int, default=10)
    parser.add_argument('--profile', action='store_true')

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    stone = args.stone
    radius = args.radius
    kernel_size = 2 * radius + 1

    data_path = args.data_path or VOLUME_PATHS[stone]
    porosity_path = args.porosity_path or get_valid_porosity_path(stone, radius)

    if args.checkpoint_dir is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d')
        checkpoint_dir = f'savedmodels/experimental/{date_str}-dfn-{stone.lower()}-{kernel_size}-valid-porosity-field'
    else:
        checkpoint_dir = args.checkpoint_dir

    print("=" * 60)
    print(f"Porosity Field Training (Valid Convolution): {stone}")
    print("=" * 60)
    print(f"  Radius: {radius} (kernel size {kernel_size})")
    print(f"  Data path: {data_path}")
    print(f"  Porosity path (valid): {porosity_path}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Effective batch size: {args.batch_size} x {args.accumulate_grad_batches} = {args.batch_size * args.accumulate_grad_batches}")
    print()

    # =========================================================================
    # Load volumes
    # =========================================================================
    print("=" * 60)
    print("Loading volumes...")
    print("=" * 60)

    print(f"Loading binary volume from {data_path}")
    volume_data = np.fromfile(data_path, dtype=np.uint8).reshape(args.volume_shape)
    print(f"  Shape: {volume_data.shape}")

    print(f"Loading valid porosity field from {porosity_path}")
    porosity_data = np.load(porosity_path)
    print(f"  Shape: {porosity_data.shape}")
    print(f"  Range: [{porosity_data.min():.4f}, {porosity_data.max():.4f}]")

    # =========================================================================
    # Crop binary volume to match valid porosity field
    # =========================================================================
    r = radius
    print(f"\nCropping binary volume by [{r}:-{r}, {r}:-{r}, {r}:-{r}] to match valid porosity field")
    print(f"  Original binary shape: {volume_data.shape}")
    volume_data = volume_data[r:-r, r:-r, r:-r]
    print(f"  Cropped binary shape:  {volume_data.shape}")

    assert volume_data.shape == porosity_data.shape, (
        f"Shape mismatch after crop! Volume {volume_data.shape} != Porosity {porosity_data.shape}. "
        f"Expected both to be ({args.volume_shape[0] - 2*r}, {args.volume_shape[1] - 2*r}, {args.volume_shape[2] - 2*r})"
    )
    print(f"  Shapes match: {volume_data.shape}")

    # =========================================================================
    # Split into train/val
    # =========================================================================
    train_split = args.train_split
    vol_size_last = volume_data.shape[2]

    assert train_split >= args.subvolume_size, (
        f"train_split ({train_split}) must be >= subvolume_size ({args.subvolume_size})"
    )
    assert vol_size_last - train_split >= args.subvolume_size, (
        f"Validation portion ({vol_size_last - train_split}) must be >= subvolume_size ({args.subvolume_size}). "
        f"Reduce --train-split (max {vol_size_last - args.subvolume_size})"
    )

    print(f"\nSplitting at z={train_split} (volume last axis = {vol_size_last})")
    volume_train = volume_data[:, :, :train_split]
    volume_val = volume_data[:, :, train_split:]
    porosity_train = porosity_data[:, :, :train_split]
    porosity_val = porosity_data[:, :, train_split:]

    print(f"  Training: {volume_train.shape}")
    print(f"  Validation: {volume_val.shape}")

    # =========================================================================
    # Create datasets
    # =========================================================================
    cube_symmetry = diffsci2.data.CubeSymmetry()

    train_dataset = VolumeSubvolumeWithPorosityDataset(
        volume=volume_train,
        porosity_volume=porosity_train,
        dataset_size=args.dataset_size,
        subvolume_size=args.subvolume_size,
        downsample_factor=args.downsample_factor,
        cube_symmetry=cube_symmetry
    )

    val_dataset = VolumeSubvolumeWithPorosityDataset(
        volume=volume_val,
        porosity_volume=porosity_val,
        dataset_size=args.dataset_size // 4,
        subvolume_size=args.subvolume_size,
        downsample_factor=args.downsample_factor,
        cube_symmetry=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    sample_data = next(iter(train_loader))
    print("Volume shape: ", sample_data['x'].shape)
    print("Porosity shape: ", sample_data['y']['porosity'].shape)

    # =========================================================================
    # Load models
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    print(f"Loading flow model for {stone}...")
    flow_model = load_flow_model(stone)
    print("  Flow model loaded successfully")

    print("Loading VAE autoencoder...")
    autoencoder = load_autoencoder()
    print("  Autoencoder loaded successfully")

    # =========================================================================
    # Create flow module
    # =========================================================================
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
    # Optimizer and scheduler
    # =========================================================================
    print("\nConfiguring optimizer...")

    optimizer = torch.optim.AdamW(
        flow_module.model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Compute optimizer steps per epoch (accounts for accumulation + DDP)
    batches_per_epoch = len(train_loader)
    num_devices = len(parse_devices(args.devices))
    # In DDP, each GPU processes a shard of the data
    batches_per_gpu = batches_per_epoch // num_devices
    optimizer_steps_per_epoch = batches_per_gpu // args.accumulate_grad_batches
    warmup_steps = args.warmup_epochs * optimizer_steps_per_epoch

    print(f"  Batches per epoch: {batches_per_epoch} total, {batches_per_gpu} per GPU ({num_devices} GPUs)")
    print(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch} (acc_grad={args.accumulate_grad_batches})")
    print(f"  Warmup: {args.warmup_epochs} epochs = {warmup_steps} optimizer steps, then constant")

    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
    )

    flow_module.set_optimizer_and_scheduler(
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval="step"
    )

    # =========================================================================
    # Callbacks
    # =========================================================================
    callbacks = []
    callbacks.append(diffsci2.models.NanToZeroGradCallback())

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
        filename='porosity-field-valid-{epoch:03d}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    if args.ema_decay > 0:
        print(f"  EMA enabled with decay={args.ema_decay}")
        callbacks.append(EMACallback(decay=args.ema_decay))

    callbacks.append(pl_callbacks.LearningRateMonitor(logging_interval='step'))

    # =========================================================================
    # Trainer
    # =========================================================================
    devices = parse_devices(args.devices)
    strategy = 'ddp' if len(devices) > 1 else 'auto'

    tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=checkpoint_dir,
        name='logs',
        default_hp_metric=False
    )

    profiler = None
    if args.profile:
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
        precision='16-mixed',
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=1
    )

    # =========================================================================
    # Train
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"\nTensorBoard: tensorboard --logdir {checkpoint_dir}/logs")
    print()

    trainer.fit(
        flow_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")


if __name__ == '__main__':
    main()

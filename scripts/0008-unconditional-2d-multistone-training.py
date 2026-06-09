#!/usr/bin/env python
"""
Unconditional 2D diffusion model training on multiple stone types.

Trains an unconditional UNet (no porosity conditioning) on 2D slices from
Bentheimer and Estaillades volumes simultaneously. The goal is to explore
what happens when generating at large scales — whether there is a phase
transition between the two stone structures, since in the infinite limit
the learned distribution must be ergodic.

Theoretical motivation:
- A convolutional (shift-equivariant) score network on an infinite domain
  can only represent ergodic distributions
- Training on two structurally different stones (Bentheimer=sandstone,
  Estaillades=carbonate) means the model must find an ergodic approximation
  to a non-ergodic mixture
- At generation time, we expect a "mosaic" structure with domains of each
  stone type, whose characteristic scale relates to the receptive field
- Generating at very large scales should reveal this phase structure

Usage:
    # Test run (fast dev run)
    python scripts/0008-unconditional-2d-multistone-training.py --fast-dev-run --devices 0

    # Full training on 6 GPUs
    python scripts/0008-unconditional-2d-multistone-training.py --devices 0,1,2,3,4,5

    # Single GPU
    python scripts/0008-unconditional-2d-multistone-training.py --devices 0
"""
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import lightning
import lightning.pytorch.callbacks as pl_callbacks

import diffsci2.models
import diffsci2.nets
import diffsci2.data
from torch.utils.data import Dataset


# =============================================================================
# UNCONDITIONAL WRAPPER
# =============================================================================

class UnconditionalWrapper(Dataset):
    """Wraps a VolumeSubsliceDataset to return {'x': tensor} without 'y' key.

    SIModule.training_step does batch.get('y', None), so omitting the key
    ensures y=None is passed to the model (empty dict {} would cause errors
    when no conditional embedding is present).
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, dict):
            return {'x': item['x']}
        return {'x': item}


# =============================================================================
# DATA PATHS
# =============================================================================
DATA_DIR = '/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/'
VOLUME_PATHS = {
    'Bentheimer': DATA_DIR + 'Bentheimer_1000c_3p0035um.raw',
    'Estaillades': DATA_DIR + 'Estaillades_1000c_3p31136um.raw',
}


# =============================================================================
# WARMUP + COSINE SCHEDULER
# =============================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            lr_factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            return [base_lr * lr_factor for base_lr in self.base_lrs]


# =============================================================================
# EMA CALLBACK
# =============================================================================

class EMACallback(pl_callbacks.Callback):
    """Exponential Moving Average of model weights."""

    def __init__(self, decay=0.9999):
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

def parse_devices(devices_str):
    """Parse device string like '0,1,2,3' into list [0, 1, 2, 3]."""
    return [int(d.strip()) for d in devices_str.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unconditional 2D diffusion model on Bentheimer + Estaillades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data
    parser.add_argument('--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000],
                        help='Shape of the volumes (D H W)')
    parser.add_argument('--subslice-size', type=int, default=256,
                        help='Size of 2D slices to extract (square)')
    parser.add_argument('--train-split', type=int, default=700,
                        help='Z-axis split: [:split] for train, [split:] for val')

    # Dataset
    parser.add_argument('--dataset-size', type=int, default=2**15,
                        help='Number of samples per epoch')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU')

    # Model architecture
    parser.add_argument('--model-channels', type=int, default=64,
                        help='Base channel count for UNet')
    parser.add_argument('--channel-expansion', type=int, nargs='+', default=[2, 4],
                        help='Channel expansion factors at each level')

    # Training
    parser.add_argument('--max-epochs', type=int, default=30,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Number of epochs for LR warmup')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                        help='EMA decay rate (0 to disable)')

    # Hardware
    parser.add_argument('--devices', type=str, default='0',
                        help='GPU devices, comma-separated (e.g., "0,1,2,3,4,5")')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--log-every-n-steps', type=int, default=10,
                        help='Log metrics every N steps')

    # Precision
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision')

    # Dev/test
    parser.add_argument('--fast-dev-run', action='store_true',
                        help='Run a single train+val batch for testing')

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    # =========================================================================
    # Resolve checkpoint dir
    # =========================================================================
    if args.checkpoint_dir is None:
        date_str = datetime.now().strftime('%Y%m%d')
        checkpoint_dir = f'savedmodels/experimental/{date_str}-unconditional-2d-bentheimer-estaillades'
    else:
        checkpoint_dir = args.checkpoint_dir

    print("=" * 70)
    print("Unconditional 2D Diffusion: Bentheimer + Estaillades")
    print("=" * 70)
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Subslice size: {args.subslice_size}x{args.subslice_size}")
    print(f"  Model channels: {args.model_channels}")
    print(f"  Channel expansion: {args.channel_expansion}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Precision: {args.precision}")
    print()

    # =========================================================================
    # Load data — both stones
    # =========================================================================
    print("Loading volumes...")
    volumes_train = []
    volumes_val = []

    for stone_name, path in VOLUME_PATHS.items():
        print(f"  Loading {stone_name} from {path}")
        vol = np.fromfile(path, dtype=np.uint8).reshape(args.volume_shape)
        print(f"    Shape: {vol.shape}, porosity: {1 - vol.mean():.4f}")

        # Split along last axis (z) for train/val
        volumes_train.append(vol[:, :, :args.train_split])
        volumes_val.append(vol[:, :, args.train_split:])

    print(f"  Train volumes: {len(volumes_train)} stones, "
          f"each ~ {volumes_train[0].shape}")
    print(f"  Val volumes: {len(volumes_val)} stones, "
          f"each ~ {volumes_val[0].shape}")

    # =========================================================================
    # Create datasets — multi-volume, unconditional
    # =========================================================================
    print("\nCreating datasets...")
    symmetry = diffsci2.data.SquareSymmetry()

    # VolumeSubsliceDataset supports a list of volumes and samples uniformly
    train_dataset = UnconditionalWrapper(diffsci2.data.VolumeSubsliceDataset(
        volumes=volumes_train,
        dataset_size=args.dataset_size,
        subslice_size=[args.subslice_size, args.subslice_size],
        square_symmetry=symmetry,
        return_as_dict=True,
    ))

    val_dataset = UnconditionalWrapper(diffsci2.data.VolumeSubsliceDataset(
        volumes=volumes_val,
        dataset_size=args.dataset_size // 4,
        subslice_size=[args.subslice_size, args.subslice_size],
        square_symmetry=None,  # No augmentation for validation
        return_as_dict=True,
    ))

    print(f"  Train samples per epoch: {len(train_dataset)}")
    print(f"  Val samples per epoch: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Quick sanity check
    sample = next(iter(train_loader))
    print(f"  Sample batch shape: {sample['x'].shape}")

    # =========================================================================
    # Create model — unconditional UNet
    # =========================================================================
    print("\nCreating model...")
    flow_model_config = diffsci2.nets.PUNetGConfig(
        input_channels=1,
        output_channels=1,
        dimension=2,
        model_channels=args.model_channels,
        channel_expansion=args.channel_expansion,
        number_resnet_downward_block=2,
        number_resnet_upward_block=2,
        number_resnet_attn_block=0,   # No attention — keeps the model strictly shift-equivariant
        number_resnet_before_attn_block=2,
        number_resnet_after_attn_block=2,
        dropout=0.0,
    )
    # No conditional embedding — purely unconditional
    flow_model = diffsci2.nets.PUNetG(
        config=flow_model_config,
        conditional_embedding=None,
    )

    total_params = sum(p.numel() for p in flow_model.parameters())
    trainable_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Create flow module (SIModule) — EDM sigma-space config
    # =========================================================================
    flow_module_config = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        initial_norm=20.0,
        loss_formulation='denoiser',
    )

    flow_module = diffsci2.models.SIModule(
        config=flow_module_config,
        model=flow_model,
        autoencoder=None,  # No autoencoder — direct pixel space
    )

    # =========================================================================
    # Optimizer + scheduler
    # =========================================================================
    print("\nConfiguring optimizer...")
    optimizer = torch.optim.AdamW(
        flow_module.model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.max_epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.01,
    )

    flow_module.set_optimizer_and_scheduler(
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval="step",
    )

    # =========================================================================
    # Callbacks
    # =========================================================================
    print("\nConfiguring callbacks...")
    callbacks = [diffsci2.models.NanToZeroGradCallback()]

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
        filename='uncond-2d-{epoch:03d}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    if args.ema_decay > 0:
        print(f"  EMA: decay={args.ema_decay}")
        callbacks.append(EMACallback(decay=args.ema_decay))

    callbacks.append(pl_callbacks.LearningRateMonitor(logging_interval='step'))

    # =========================================================================
    # Trainer
    # =========================================================================
    print("\nConfiguring trainer...")
    devices = parse_devices(args.devices)
    strategy = 'ddp' if len(devices) > 1 else 'auto'
    print(f"  Devices: {devices}")
    print(f"  Strategy: {strategy}")

    tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir=checkpoint_dir,
        name='logs',
        default_hp_metric=False,
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
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
    )

    # =========================================================================
    # Train
    # =========================================================================
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    print(f"\nTensorBoard: tensorboard --logdir {checkpoint_dir}/logs")
    print(f"Checkpoints: {checkpoint_dir}/checkpoints")
    print()

    trainer.fit(
        flow_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    if not args.fast_dev_run:
        print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")


if __name__ == '__main__':
    main()

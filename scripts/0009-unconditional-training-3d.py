#!/usr/bin/env python
"""
Unconditional 3D latent diffusion model training from scratch.

Trains a 3D flow model in latent space (using the pretrained VAE autoencoder)
without any conditioning. This is useful for studying diversity at different
generation sizes — e.g. 128^3 avoids the diversity collapse seen at 256^3.

Usage:
    # Single GPU with gradient accumulation
    python scripts/0009-unconditional-training-3d.py \
        --stone Estaillades \
        --subvolume-size 128 \
        --devices 6 \
        --batch-size 1 \
        --accumulate-grad-batches 8

    # Multi-GPU
    python scripts/0009-unconditional-training-3d.py \
        --stone Estaillades \
        --subvolume-size 128 \
        --devices 0,1,2,3 \
        --batch-size 2
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import lightning
import lightning.pytorch.callbacks as pl_callbacks
from torch.utils.data import Dataset, DataLoader

# Add the aux directory for model_loaders (for load_autoencoder)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
import diffsci2.data

from model_loaders import load_autoencoder


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


# =============================================================================
# UNCONDITIONAL WRAPPER
# =============================================================================

class UnconditionalWrapper(Dataset):
    """Wraps a VolumeSubvolumeDataset to return {'x': tensor} without 'y' key.

    SIModule.training_step does batch.get('y', None), so omitting the key
    ensures y=None is passed to the model.
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
# LR SCHEDULER
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
        description='Unconditional 3D latent diffusion model training from scratch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Stone type
    parser.add_argument('--stone', type=str, default='Estaillades',
                        choices=['Bentheimer', 'Doddington', 'Estaillades', 'Ketton'],
                        help='Stone type to train on')

    # Data
    parser.add_argument('--data-path', type=str, default=None,
                        help='Override: path to binary volume .raw file')
    parser.add_argument('--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000],
                        help='Shape of the volume [D, H, W]')
    parser.add_argument('--subvolume-size', type=int, default=128,
                        help='Size of cubic subvolumes to extract')
    parser.add_argument('--train-split', type=int, default=700,
                        help='Number of z-slices for training (rest for validation)')

    # Dataset / training
    parser.add_argument('--dataset-size', type=int, default=int(34560),
                        help='Number of samples per epoch')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--accumulate-grad-batches', type=int, default=8,
                        help='Accumulate gradients over N batches')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum number of training epochs')

    # Model architecture
    parser.add_argument('--model-channels', type=int, default=64,
                        help='Base channel count for UNet')
    parser.add_argument('--channel-expansion', type=int, nargs='+', default=[2, 4],
                        help='Channel expansion factors at each level')

    # Optimizer
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
                        help='GPU devices, comma-separated (e.g., "0,1,2,3")')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Override checkpoint directory')
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

    stone = args.stone
    data_path = args.data_path or VOLUME_PATHS[stone]

    # Default checkpoint directory: {date}-dfn-unconditional-0009-{stone}-{size}
    if args.checkpoint_dir is None:
        date_str = datetime.now().strftime('%Y%m%d')
        checkpoint_dir = (
            f'savedmodels/experimental/'
            f'{date_str}-dfn-unconditional-0009-{stone.lower()}-{args.subvolume_size}'
        )
    else:
        checkpoint_dir = args.checkpoint_dir

    print("=" * 70)
    print(f"Unconditional 3D Latent Diffusion: {stone} @ {args.subvolume_size}^3")
    print("=" * 70)
    print(f"  Data path: {data_path}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Subvolume size: {args.subvolume_size}^3")
    print(f"  Model channels: {args.model_channels}")
    print(f"  Channel expansion: {args.channel_expansion}")
    print(f"  Effective batch size: {args.batch_size} x {args.accumulate_grad_batches}"
          f" = {args.batch_size * args.accumulate_grad_batches}")
    print(f"  Precision: {args.precision}")
    print()

    # =========================================================================
    # Load volume
    # =========================================================================
    print("Loading volume...")
    volume_data = np.fromfile(data_path, dtype=np.uint8).reshape(args.volume_shape)
    print(f"  Shape: {volume_data.shape}")
    print(f"  Porosity: {1 - volume_data.mean():.4f}")

    # Train/val split along last axis (z)
    train_split = args.train_split
    volume_train = volume_data[:, :, :train_split]
    volume_val = volume_data[:, :, train_split:]
    print(f"  Train volume: {volume_train.shape}")
    print(f"  Val volume: {volume_val.shape}")

    # =========================================================================
    # Create datasets
    # =========================================================================
    print("\nCreating datasets...")
    cube_symmetry = diffsci2.data.CubeSymmetry()

    train_dataset = UnconditionalWrapper(diffsci2.data.VolumeSubvolumeDataset(
        volumes=volume_train,
        dataset_size=args.dataset_size,
        subvolume_size=args.subvolume_size,
        cube_symmetry=cube_symmetry,
        return_as_dict=True,
    ))

    val_dataset = UnconditionalWrapper(diffsci2.data.VolumeSubvolumeDataset(
        volumes=volume_val,
        dataset_size=args.dataset_size // 4,
        subvolume_size=args.subvolume_size,
        cube_symmetry=None,
        return_as_dict=True,
    ))

    print(f"  Train samples per epoch: {len(train_dataset)}")
    print(f"  Val samples per epoch: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Sanity check
    sample = next(iter(train_loader))
    print(f"  Sample batch shape: {sample['x'].shape}")

    # =========================================================================
    # Create model — unconditional, from scratch
    # =========================================================================
    print("\nCreating model (from scratch, unconditional)...")
    flow_model_config = diffsci2.nets.PUNetGConfig(
        input_channels=4,
        output_channels=4,
        dimension=3,
        model_channels=args.model_channels,
        channel_expansion=args.channel_expansion,
        number_resnet_downward_block=2,
        number_resnet_upward_block=2,
        number_resnet_attn_block=0,
        number_resnet_before_attn_block=3,
        number_resnet_after_attn_block=3,
        kernel_size=3,
        in_out_kernel_size=3,
        in_embedding=False,
        time_projection_scale=10.0,
        input_projection_scale=1.0,
        transition_scale_factor=2,
        transition_kernel_size=3,
        dropout=0.1,
        cond_dropout=0.0,
        first_resblock_norm="GroupLN",
        second_resblock_norm="GroupRMS",
        affine_norm=True,
        convolution_type="default",
        num_groups=1,
        attn_residual=False,
        attn_type="default",
        bias=True,
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
    # Load autoencoder (frozen, for encoding/decoding only)
    # =========================================================================
    print("\nLoading VAE autoencoder...")
    autoencoder = load_autoencoder()
    print("  Autoencoder loaded successfully")

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
        autoencoder=autoencoder,
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

    # With scheduler_interval="step", Lightning calls scheduler.step() per
    # optimizer step, not per batch. With gradient accumulation the number of
    # optimizer steps per epoch is len(train_loader) // accumulate_grad_batches.
    # In DDP, each GPU processes a shard, so divide by num_devices.
    batches_per_epoch = len(train_loader)
    num_devices = len(parse_devices(args.devices))
    batches_per_gpu = batches_per_epoch // num_devices
    optimizer_steps_per_epoch = batches_per_gpu // args.accumulate_grad_batches
    total_steps = args.max_epochs * optimizer_steps_per_epoch
    warmup_steps = args.warmup_epochs * optimizer_steps_per_epoch

    print(f"  Batches per epoch: {batches_per_epoch} total, {batches_per_gpu} per GPU ({num_devices} GPUs)")
    print(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps} ({args.warmup_epochs} epochs)")

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
        filename='uncond-3d-{epoch:03d}-{val_loss:.6f}',
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
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=1,
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

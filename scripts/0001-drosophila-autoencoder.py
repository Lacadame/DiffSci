#!/usr/bin/env python
"""
Simple 2D diffusion model training on binary volume slices.

This script trains a conditional flow model on 2D slices from 3D binary volumes,
using porosity as a conditioning variable.

Usage:
    # Single GPU
    python scripts/0001-drosophila-training.py --data-path /path/to/volume.raw --devices 0

    # Multi-GPU DDP
    python scripts/0001-drosophila-training.py --data-path /path/to/volume.raw --devices 0,1,2,3,4,5
"""
import argparse
import os

import numpy as np
import torch
import lightning
import lightning.pytorch.callbacks as pl_callbacks

import diffsci2.models
import diffsci2.nets
import diffsci2.data


def parse_devices(devices_str):
    """Parse device string like '0,1,2,3' into list [0, 1, 2, 3]."""
    return [int(d.strip()) for d in devices_str.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(description='Train 2D diffusion model on volume slices')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to .raw volume file')
    parser.add_argument('--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000],
                        help='Shape of the volume (D H W)')
    parser.add_argument('--subslice-size', type=int, nargs=2, default=[256, 256],
                        help='Size of 2D slices to extract (H W)')
    parser.add_argument('--dataset-size', type=int, default=2**15,
                        help='Number of samples per epoch')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--max-epochs', type=int, default=20,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--model-channels', type=int, default=32,
                        help='Base channel count for UNet')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--devices', type=str, default='0',
                        help='GPU devices, comma-separated (e.g., "0,1,2,3")')
    parser.add_argument('--gradient-clip', type=float, default=0.5,
                        help='Gradient clipping value')
    return parser.parse_args()


def porosity_extractor(x):
    """Extract porosity from a binary tensor (0=pore, 1=solid)."""
    return {'porosity': 1 - x.mean()}


def main():
    args = parse_args()

    # Load data
    print(f"Loading volume from {args.data_path}")
    data = np.fromfile(args.data_path, dtype=np.uint8).reshape(args.volume_shape)
    print(f"Volume shape: {data.shape}")

    # Create dataset
    symmetry = diffsci2.data.SquareSymmetry()
    train_dataset = diffsci2.data.VolumeSubsliceDataset(
        data[:, :, :700],
        subslice_size=args.subslice_size,
        dataset_size=args.dataset_size,
        square_symmetry=symmetry,
        extractor=porosity_extractor,
        return_as_dict=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataset = diffsci2.data.VolumeSubsliceDataset(
        data[:, :, 700:],
        subslice_size=args.subslice_size,
        dataset_size=args.dataset_size,
        square_symmetry=symmetry,
        extractor=porosity_extractor,
        return_as_dict=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    vaenet_config_args = {
        'dimension': 2,
        'in_channels': 1,
        'out_channels': 1,
        'z_channels': 4,
        'z_dim': 4,
        'ch': 32,
        'ch_mult': [1, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
        'resolution': 256,
        'has_mid_attn': False,
        'resamp_with_conv': True,
        'attn_type': 'vanilla',
        'tanh_out': False,
        'input_bias': True,
        'output_bias': True,
        'with_time_emb': False,
        'double_z': True,
        'num_groups': 32,
        'patch_size': None,
        'memory_efficient_variant': False,
        'use_flash_attention': True,
        'minimal_rf_mode': False,
    }

    vaenet_config = diffsci2.nets.VAENetConfig(**vaenet_config_args)
    vaenet_model = diffsci2.nets.VAENet(config=vaenet_config)
    vaemoduleconfig = diffsci2.models.VAEModuleConfig()
    vaemodule = diffsci2.models.VAEModule(config=vaemoduleconfig, encdec=vaenet_model)

    # Setup optimizer
    optimizer = torch.optim.AdamW(vaemodule.parameters(), lr=args.lr)
    vaemodule.set_optimizer_and_scheduler(optimizer)

    # Callbacks
    callbacks = [diffsci2.models.NanToZeroGradCallback()]

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, 'checkpoints'),
        filename='model-{epoch:03d}-{val_loss:.6f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Parse devices
    devices = parse_devices(args.devices)
    strategy = 'ddp' if len(devices) > 1 else 'auto'

    # Create trainer
    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.checkpoint_dir,
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        enable_checkpointing=True,
        devices=devices,
        strategy=strategy,
    )

    # Train
    print("Starting training...")
    trainer.fit(vaemodule, train_dataloader, val_dataloader)
    print("Training complete!")


if __name__ == '__main__':
    main()

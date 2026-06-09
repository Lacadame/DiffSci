#!/usr/bin/env python
"""
Enhanced porosity field conditioning training with FiLM modulation.

This script fine-tunes an existing porosity-conditioned model with:
1. FiLM (Feature-wise Linear Modulation) for stronger conditioning
2. Multi-scale conditioning injection at each UNet level
3. Porosity gradient features for boundary awareness
4. Classifier-free guidance training via condition dropout

The key insight: new layers are initialized to IDENTITY, so the model starts
exactly where the pretrained model left off, then learns stronger conditioning.

=============================================================================
DESIGN PHILOSOPHY
=============================================================================

This is a POST-TRAINING approach:
1. Load an existing checkpoint (e.g., from 0003-porosity-field-training.py)
2. Wrap the model with EnhancedConditioningWrapper
3. Fine-tune with:
   - Low LR on base model (preserve learned features)
   - High LR on new FiLM/conditioning layers (learn fast)
   - CFG dropout enabled

The wrapper adds:
- FiLM layers at each encoder/decoder level (per-pixel, no attention)
- Multi-scale conditioning embeddings
- Gradient features (where porosity changes = boundaries)

All new layers start as identity, so initial behavior = pretrained model.

=============================================================================
USAGE
=============================================================================
    python scripts/0003b-porosity-field-training-enhanced.py \\
        --checkpoint /path/to/existing/checkpoint.ckpt \\
        --stone Estaillades \\
        --devices 6 \\
        --batch-size 1 \\
        --accumulate-grad-batches 8 \\
        --lr 1e-5 \\
        --new-params-lr-mult 10.0

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

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
import diffsci2.data
from diffsci2.nets.enhanced_conditioning import wrap_model_with_enhanced_conditioning

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
        assert volume.shape == porosity_volume.shape
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
        assert self.max_d >= 0 and self.max_h >= 0 and self.max_w >= 0

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
# LR SCHEDULER
# =============================================================================

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup-only scheduler: ramps up during warmup, then stays constant."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + cosine decay scheduler."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01
    ):
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
        description='Enhanced porosity field conditioning training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Source checkpoint (required for post-training)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to existing model checkpoint to enhance')

    parser.add_argument('--stone', type=str, default='Estaillades',
                        choices=['Bentheimer', 'Doddington', 'Estaillades', 'Ketton'])

    parser.add_argument('--data-source', type=str, default='gpdata2',
                        choices=['gpdata2', 'gpdata4-129'])

    parser.add_argument('--center-crop', type=int, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--porosity-path', type=str, default=None)
    parser.add_argument('--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000])

    parser.add_argument('--subvolume-size', type=int, default=256)
    parser.add_argument('--downsample-factor', type=int, default=8)
    parser.add_argument('--dataset-size', type=int, default=int(2**12))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulate-grad-batches', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--train-split', type=int, default=700)

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Base learning rate (for existing model params)')
    parser.add_argument('--new-params-lr-mult', type=float, default=10.0,
                        help='LR multiplier for new conditioning params (default: 10x)')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Override warmup steps directly (takes precedence over --warmup-epochs)')
    parser.add_argument('--has-cosine-decay', action='store_true',
                        help='Enable cosine decay after warmup (default: constant LR after warmup)')
    parser.add_argument('--ema-decay', type=float, default=0.9999)

    # Enhanced conditioning options
    parser.add_argument('--condition-embed-dim', type=int, default=64,
                        help='Dimension for FiLM conditioning embeddings')
    parser.add_argument('--no-film', action='store_true',
                        help='Disable FiLM modulation')
    parser.add_argument('--no-multiscale', action='store_true',
                        help='Disable multi-scale conditioning')
    parser.add_argument('--no-gradient', action='store_true',
                        help='Disable porosity gradient features')
    parser.add_argument('--condition-amplification', type=float, default=1.0,
                        help='Initial conditioning amplification factor')
    parser.add_argument('--cond-drop-p', type=float, default=0.1,
                        help='Probability of dropping conditioning for CFG (0.1 = 10%)')

    # Hardware
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=4)

    # Output
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
    data_path = args.data_path or VOLUME_PATHS[stone]

    if args.porosity_path:
        porosity_path = args.porosity_path
    elif args.data_source == 'gpdata2':
        porosity_path = POROSITY_BASE_DIR + POROSITY_PATHS_GPDATA2[stone]
    else:
        porosity_path = POROSITY_BASE_DIR + POROSITY_PATHS_GPDATA4_129[stone]

    if args.checkpoint_dir is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y%m%d')
        suffix = '-129' if args.data_source == 'gpdata4-129' else ''
        checkpoint_dir = f'savedmodels/experimental/{date_str}-dfn-{stone.lower()}{suffix}-enhanced-cond'
    else:
        checkpoint_dir = args.checkpoint_dir

    print("=" * 70)
    print("Enhanced Porosity Field Conditioning Training")
    print("=" * 70)
    print(f"  Source checkpoint: {args.checkpoint}")
    print(f"  Stone: {stone}")
    print(f"  Data source: {args.data_source}")
    print(f"  Output dir: {checkpoint_dir}")
    print()
    print("Enhanced Conditioning Options:")
    print(f"  FiLM: {not args.no_film}")
    print(f"  Multi-scale: {not args.no_multiscale}")
    print(f"  Gradient features: {not args.no_gradient}")
    print(f"  Condition embed dim: {args.condition_embed_dim}")
    print(f"  Condition amplification: {args.condition_amplification}")
    print(f"  CFG dropout: {args.cond_drop_p}")
    print()
    print("Learning Rates:")
    print(f"  Base model: {args.lr}")
    print(f"  New params: {args.lr * args.new_params_lr_mult} ({args.new_params_lr_mult}x)")
    print()
    print("Training Config:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulate grad batches: {args.accumulate_grad_batches}")
    print(f"  Effective batch size: {args.batch_size * args.accumulate_grad_batches}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Cosine decay: {args.has_cosine_decay}")
    print()

    # =========================================================================
    # Load volumes
    # =========================================================================
    print("=" * 70)
    print("Loading volumes...")
    print("=" * 70)

    volume_data = np.fromfile(data_path, dtype=np.uint8).reshape(args.volume_shape)
    porosity_data = np.load(porosity_path)

    if args.center_crop is not None:
        c = args.center_crop
        volume_data = volume_data[c:-c, c:-c, c:-c]
        porosity_data = porosity_data[c:-c, c:-c, c:-c]
        print(f"  Center cropped to: {volume_data.shape}")

    train_split = args.train_split
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

    # =========================================================================
    # Load models and wrap with enhanced conditioning
    # =========================================================================
    print("\n" + "=" * 70)
    print("Loading and wrapping model...")
    print("=" * 70)

    print(f"  Loading from: {args.checkpoint}")

    # Load checkpoint and detect if enhanced or not
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Extract model state - split by '.', drop first element ('model'), rejoin
    model_state = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if k.startswith('model.')}

    # Check if this is an enhanced checkpoint
    is_enhanced_checkpoint = any(
        k.startswith('film_layers') or k.startswith('condition_encoder') or k.startswith('base_model.')
        for k in model_state.keys()
    )

    # Create base PUNetG architecture - EXACTLY as in model_loaders.py
    config = diffsci2.nets.PUNetGConfig(
        input_channels=4,
        output_channels=4,
        dimension=3,
        model_channels=64,
        channel_expansion=[2, 4],
        number_resnet_downward_block=2,
        number_resnet_upward_block=2,
        number_resnet_attn_block=0,  # NO attention blocks
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
        cond_dropout=0.1,
        first_resblock_norm="GroupLN",
        second_resblock_norm="GroupRMS",
        affine_norm=True,
        convolution_type="default",
        num_groups=1,
        attn_residual=False,
        attn_type="default",
        bias=True
    )
    conditional_embedding = diffsci2.nets.ScalarEmbedder(dembed=64, key='porosity')
    base_model = diffsci2.nets.PUNetG(config, conditional_embedding=conditional_embedding)

    # Wrap with enhanced conditioning
    enhanced_model = wrap_model_with_enhanced_conditioning(
        model=base_model,
        condition_embed_dim=args.condition_embed_dim,
        use_film=not args.no_film,
        use_multiscale=not args.no_multiscale,
        use_gradient=not args.no_gradient,
        condition_amplification=args.condition_amplification,
        cond_drop_p=args.cond_drop_p,
    )

    if is_enhanced_checkpoint:
        # Load directly into enhanced model
        print("  Detected ENHANCED checkpoint - loading full state")
        enhanced_model.load_state_dict(model_state)
        print("  Enhanced weights loaded successfully")
    else:
        # Load into base model, enhanced layers stay at identity init
        print("  Detected STANDARD checkpoint - loading into base model only")
        base_model.load_state_dict(model_state)
        print("  Base weights loaded, enhanced layers initialized to identity")

    # Count parameters
    base_params = sum(p.numel() for p in enhanced_model.base_model.parameters())
    new_params = sum(p.numel() for p in enhanced_model.get_trainable_params())
    print(f"  Base model params: {base_params:,}")
    print(f"  New conditioning params: {new_params:,}")
    print(f"  Total: {base_params + new_params:,}")

    # Load autoencoder
    autoencoder = load_autoencoder()

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
        model=enhanced_model,
        autoencoder=autoencoder
    )

    # =========================================================================
    # Setup optimizer with different LRs for base vs new params
    # =========================================================================
    print("\nConfiguring optimizer...")

    param_groups = enhanced_model.get_all_params_with_lr_groups(
        base_lr=args.lr,
        new_params_lr_mult=args.new_params_lr_mult
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # In Lightning with scheduler_interval="step", the scheduler is called
    # after each OPTIMIZER step (i.e., after gradient accumulation completes).
    # So we must calculate steps in terms of optimizer steps, not batches.
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = batches_per_epoch // args.accumulate_grad_batches

    total_steps = args.max_epochs * optimizer_steps_per_epoch

    # Warmup: use --warmup-steps if provided, otherwise compute from --warmup-epochs
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
        warmup_source = f"{args.warmup_steps} steps (explicit)"
    else:
        warmup_steps = args.warmup_epochs * optimizer_steps_per_epoch
        warmup_source = f"{args.warmup_epochs} epochs = {warmup_steps} steps"

    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch} (with acc_grad={args.accumulate_grad_batches})")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup: {warmup_source}")

    if args.has_cosine_decay:
        print(f"  Scheduler: WarmupCosineScheduler (cosine decay after warmup)")
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.01
        )
    else:
        print(f"  Scheduler: WarmupScheduler (constant LR after warmup)")
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
    # Setup callbacks
    # =========================================================================
    callbacks = []
    callbacks.append(diffsci2.models.NanToZeroGradCallback())

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
        filename='enhanced-{epoch:03d}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    if args.ema_decay > 0:
        callbacks.append(EMACallback(decay=args.ema_decay))

    callbacks.append(pl_callbacks.LearningRateMonitor(logging_interval='step'))

    # =========================================================================
    # Setup trainer
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
    print("\n" + "=" * 70)
    print("Starting enhanced conditioning training...")
    print("=" * 70)
    print(f"\nTensorBoard: tensorboard --logdir {checkpoint_dir}/logs")
    print()

    trainer.fit(
        flow_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()

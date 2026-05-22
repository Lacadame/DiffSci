"""CLI entrypoint for VAE supervised fine-tuning.

Drop-in replacement for `notebooks/exploratory/dfnai/scripts/vaeporesft/train.py`.

Usage (5-way DDP on GPUs 1..5):
    cd /home/ubuntu/repos/DiffSci2
    /opt/persistence/miniconda3/envs/ddpm_env/bin/python \
        pipelines/vae/train_vae_sft.py \
            --run-name run01 \
            --devices 1,2,3,4,5 \
            --total-steps 10000

Single-GPU:
    python pipelines/vae/train_vae_sft.py --run-name dev --devices 0 --total-steps 500
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from diffsci2.vaesft import SFTConfig, VAESFTModule

# Local: host-specific paths (CKPT_DIR / LOG_DIR roots).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, default=None,
                   help="If None, uses run_YYYYmmdd_HHMMSS.")
    # data
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-n-chunks", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    # input transform
    p.add_argument("--temperature", type=float, default=5.0)
    # loss
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--target-weights", type=float, nargs=4,
                   default=[1.0, 1.0, 1.0, 1.0],
                   help="Per-target weights for "
                        "(surface_area_density, mean_pore_size, "
                        "mean_curvature, euler_number_density).")
    p.add_argument("--base-bce", type=float, default=1.0)
    p.add_argument("--base-reg", type=float, default=1.0)
    # schedule
    p.add_argument("--warmup-bce-steps", type=int, default=1000)
    p.add_argument("--ramp-steps", type=int, default=1000)
    p.add_argument("--total-steps", type=int, default=10000)
    # optim
    p.add_argument("--lr-decoder", type=float, default=1e-5)
    p.add_argument("--lr-encoder", type=float, default=None,
                   help="If set, fine-tune the encoder too. Default = frozen.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cosine-decay", action="store_true",
                   help="Cosine LR decay over [warmup+ramp, total_steps].")
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    # pretrained VAE selection
    p.add_argument("--vae-ckpt", type=str, default=None,
                   help="Explicit path to a VAE checkpoint. Overrides --vae-variant.")
    p.add_argument("--vae-variant", type=str, default=None,
                   help="Variant name from diffsci2.vaesft.VARIANT_REGISTRY "
                        "(e.g. vae_pixnorm_s8_raw). Defaults to the registry "
                        "default if both --vae-ckpt and --vae-variant are unset.")
    # devices / precision
    p.add_argument("--devices", type=str, default="1,2,3,4,5")
    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["32-true", "16-mixed", "bf16-mixed"])
    # logging / eval / ckpt cadence
    p.add_argument("--log-every-n-steps", type=int, default=20)
    p.add_argument("--val-check-interval", type=int, default=200)
    p.add_argument("--ckpt-every-n-train-steps", type=int, default=500)
    # roots
    p.add_argument("--ckpt-root", type=str, default=paths.CKPT_DIR)
    p.add_argument("--log-root", type=str, default=paths.LOG_DIR)
    return p.parse_args()


def _parse_devices(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    devices = _parse_devices(args.devices)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    cfg = SFTConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_n_chunks=args.eval_n_chunks,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        temperature=args.temperature,
        huber_beta=args.huber_beta,
        target_weights=tuple(args.target_weights),
        base_bce=args.base_bce,
        base_reg=args.base_reg,
        warmup_bce_steps=args.warmup_bce_steps,
        ramp_steps=args.ramp_steps,
        total_steps=args.total_steps,
        lr_decoder=args.lr_decoder,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        cosine_decay=args.cosine_decay,
        min_lr_ratio=args.min_lr_ratio,
        vae_ckpt=args.vae_ckpt,
        vae_variant=args.vae_variant,
    )
    module = VAESFTModule(cfg)

    ckpt_dir = os.path.join(args.ckpt_root, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_logger = CSVLogger(save_dir=args.log_root, name=run_name)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-step{step}-z{val/mae_true_z_mean:.4f}",
            monitor="val/mae_true_z_mean",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="periodic-step{step}",
            every_n_train_steps=args.ckpt_every_n_train_steps,
            save_top_k=-1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_steps=args.total_steps,
        devices=devices,
        accelerator="gpu",
        strategy="ddp" if len(devices) > 1 else "auto",
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_check_interval,
        limit_train_batches=args.total_steps,
        num_sanity_val_steps=0,
        deterministic=False,
        benchmark=True,
    )

    trainer.fit(module)
    if trainer.is_global_zero:
        best = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if best:
            print(f"Training done. Best ckpt: {best}")


if __name__ == "__main__":
    main()

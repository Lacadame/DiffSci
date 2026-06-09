"""Single-GPU smoke check for the SFT engine.

Builds the LightningModule + trainer with a tiny config and runs ~5 training
batches + 1 validation pass to confirm the wiring.

Usage:
    CUDA_VISIBLE_DEVICES=0 \
        /opt/persistence/miniconda3/envs/ddpm_env/bin/python \
        pipelines/vae/smoke_test_sft.py
"""
from __future__ import annotations

import tempfile

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger

from diffsci2.vaesft import SFTConfig, VAESFTModule


def main():
    L.seed_everything(0)
    torch.set_float32_matmul_precision("high")

    cfg = SFTConfig(
        batch_size=4,
        num_workers=2,
        eval_n_chunks=8,
        eval_batch_size=4,
        temperature=5.0,
        warmup_bce_steps=2,
        ramp_steps=2,
        total_steps=6,
        lr_decoder=1e-5,
    )
    module = VAESFTModule(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        logger = CSVLogger(save_dir=tmp, name="smoke")
        trainer = L.Trainer(
            max_steps=cfg.total_steps,
            devices=[0],
            accelerator="gpu",
            precision="bf16-mixed",
            gradient_clip_val=1.0,
            logger=logger,
            log_every_n_steps=1,
            check_val_every_n_epoch=None,
            val_check_interval=3,
            limit_train_batches=cfg.total_steps,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
        )
        trainer.fit(module)

    print("smoke ok")


if __name__ == "__main__":
    main()

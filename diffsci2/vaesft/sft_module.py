"""LightningModule wrapping the SFT setup.

One training step:

  1. Real chunk x_real  ->  (frozen, or unfrozen if --lr-encoder is set)
     encoder              ->  latent mean z
  2. Trainable decoder    ->  continuous field x_hat
  3. logits = T * z(x_hat) (per-chunk z-score, clamped) -- shared with BCE
  4. x_soft = sigmoid(logits)                 # near-binary, differentiable
  5. y_pred = regressor.predict_raw(x_soft)   # raw morphology, differentiable
  6. loss   = w_bce * BCE_with_logits(logits, x_real)
            + w_reg * Huber_z(y_pred, y_true, y_std, target_weights)
            [+ optional anti-drift terms -- all default-off; see SFTConfig
             fields `zscore_mode` / `w_field_flatness` / `w_raw_anchor` and
             diffsci2/vaesft/{loss,regressor}.py for the s8-blob diagnosis]
     where (w_bce, w_reg) follow the BCE-warmup -> ramp schedule.

Validation step:
  1. Encoder mean -> z, decoder -> x_hat, deterministic threshold -> x_recon.
  2. y_pred = regressor.predict_raw(x_recon).
  3. Log per-target MAE vs cached truth and the aggregate
     `val/mae_true_z_mean` (mean absolute z-scored error). The aggregate is
     what `ModelCheckpoint` monitors, with `mode="min"`.

Subdivision of frozen vs trainable parameters:
  - `self.regressor`  : FrozenRegressor wrapper, all params requires_grad=False.
  - `self.encoder`    : requires_grad=False by default; True if `lr_encoder` is set.
  - `self.decoder`    : trainable.

DDP-safe because every parameter with requires_grad=True is touched in the
training step (encoder when unfrozen, decoder always), so the default
`find_unused_parameters=False` works without modification.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader

# `_paths` puts the poreregressor scripts dir on sys.path so TARGET_NAMES
# imports cleanly.
from . import _paths  # noqa: F401
from poreregressor.data_constants import TARGET_NAMES

from .chunk_loader import CachedChunkSampler, EvalPackDataset, collate_cached
from .loaders import load_autoencoder, DEFAULT_VARIANT
from .loss import (
    HuberCfg,
    bce_pixel_anchor,
    field_flatness_penalty,
    raw_consistency_anchor,
    regressor_huber,
    regressor_z_err_mean,
)
from .regressor import (
    LOGIT_CLAMP,
    deterministic_binary,
    deterministic_binary_local,
    load_frozen_regressor,
    normalize_to_logits,
    normalize_to_logits_local,
)
from .schedule import LossWeights, ScheduleCfg, weights_at_step


@dataclass
class SFTConfig:
    # data
    batch_size: int = 16
    num_workers: int = 4
    eval_n_chunks: int = 64
    eval_batch_size: int = 8
    seed: int = 42
    # input transform
    temperature: float = 5.0
    logit_clamp: float = LOGIT_CLAMP
    eps_std: float = 1e-6
    # loss
    huber_beta: float = 1.0
    target_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    base_bce: float = 1.0
    base_reg: float = 1.0
    # schedule
    warmup_bce_steps: int = 1000
    ramp_steps: int = 1000
    total_steps: int = 10000
    # optim
    lr_decoder: float = 1e-5
    lr_encoder: Optional[float] = None     # None == encoder frozen
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    # LR schedule. cosine_decay=False keeps the run01 flat-LR behavior.
    cosine_decay: bool = False
    min_lr_ratio: float = 0.05
    # Pretrained VAE selection. Exactly one of (vae_ckpt, vae_variant) is
    # respected; if both are None, falls back to `DEFAULT_VARIANT` from
    # `diffsci2.vaesft.loaders`. `vae_ckpt` is a full path; `vae_variant`
    # is a registry key like 'vae_pixnorm_s8_sft'.
    vae_ckpt: Optional[str] = None
    vae_variant: Optional[str] = None
    # --- anti-drift / s8 binarization-blob fixes -----------------------------
    # Three independent knobs. ALL default to the run04 behavior, so leaving
    # them untouched reproduces the existing pipeline bit-for-bit. The
    # diagnosis lives in diffsci2/vaesft/regressor.py::normalize_to_logits.
    #
    #   Option C -- z-score mode for the transform feeding BCE + regressor.
    #   "global" = current behavior; "local" subtracts a locally-pooled
    #   mean/std so the low-frequency field is removed from `logits` and the
    #   loss can no longer ignore it. Eval binarization switches to the
    #   matching local threshold automatically.
    zscore_mode: str = "global"            # "global" | "local"
    local_zscore_kernel: int = 17          # odd; used when zscore_mode="local"
    #   Option B -- field-flatness penalty. Adds
    #   w * Var(pool(x_hat)) / Var(x_hat): 0 for a flat field (+ any global
    #   offset), growing with a ramp. Self-contained; eval unchanged.
    w_field_flatness: float = 0.0
    field_flatness_kernel: int = 16        # must divide the training chunk size
    #   Option A -- consistency anchor to the frozen raw decoder. Adds
    #   w * MSE(x_hat, x_hat_raw). lowpass kernel 0 = full MSE; >0 = pin only
    #   the low-frequency component (recommended).
    w_raw_anchor: float = 0.0
    raw_anchor_lowpass_kernel: int = 0


def _encode_mean(encoder: nn.Module, x: torch.Tensor, z_dim: int) -> torch.Tensor:
    """Encoder produces [B, 2*z_dim, *spatial]; strip logvar, return the mean."""
    return encoder(x)[:, :z_dim]


class VAESFTModule(L.LightningModule):
    def __init__(self, cfg: SFTConfig):
        super().__init__()
        # save_hyperparameters() needs json-serializable values; the
        # `target_weights` tuple and Nones in lr_encoder are fine.
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg

        if cfg.vae_ckpt is not None:
            vae = load_autoencoder(path=cfg.vae_ckpt)
        else:
            vae = load_autoencoder(cfg.vae_variant or DEFAULT_VARIANT)
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.z_dim = vae.config.z_dim

        # Freeze policy
        if cfg.lr_encoder is None:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
        else:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
        for p in self.decoder.parameters():
            p.requires_grad_(True)

        # Option A (anti-drift): a frozen copy of the *raw*, pre-SFT decoder,
        # built only when the consistency anchor is enabled. The deepcopy
        # happens here -- before any optimizer step -- so it captures the
        # pretrained weights. All params requires_grad=False, so DDP ignores
        # it in the gradient sync. Stays None when the anchor is off.
        self.decoder_raw: Optional[nn.Module] = None
        if cfg.w_raw_anchor > 0.0:
            self.decoder_raw = copy.deepcopy(self.decoder)
            for p in self.decoder_raw.parameters():
                p.requires_grad_(False)
            self.decoder_raw.eval()

        # Frozen regressor (held as a real submodule so Lightning moves it
        # to the right device; all params have requires_grad=False so DDP
        # never tries to sync gradients on them).
        self.regressor = load_frozen_regressor(map_location="cpu")

        self.huber_cfg = HuberCfg(
            beta=cfg.huber_beta,
            target_weights=tuple(cfg.target_weights),
        )
        self.schedule_cfg = ScheduleCfg(
            warmup_bce_steps=cfg.warmup_bce_steps,
            ramp_steps=cfg.ramp_steps,
            base_bce=cfg.base_bce,
            base_reg=cfg.base_reg,
        )

    # ------------------------------------------------------------------
    # Checkpoint helper (compensates for `save_hyperparameters(cfg.__dict__)`
    # flattening the dataclass, which breaks the default `load_from_checkpoint`).
    # ------------------------------------------------------------------

    @classmethod
    def load_sft_checkpoint(
        cls,
        ckpt_path: str,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "VAESFTModule":
        blob = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        hp = blob.get("hyper_parameters") or blob.get("hparams") or {}
        cfg_fields = set(SFTConfig.__dataclass_fields__.keys())
        cfg_kwargs = {k: v for k, v in hp.items() if k in cfg_fields}
        # Tuples get serialized as lists in some logger paths; restore.
        for tk in ("target_weights", "betas"):
            if tk in cfg_kwargs and not isinstance(cfg_kwargs[tk], tuple):
                cfg_kwargs[tk] = tuple(cfg_kwargs[tk])
        cfg = SFTConfig(**cfg_kwargs)
        module = cls(cfg)
        module.load_state_dict(blob["state_dict"], strict=strict)
        return module

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _forward_decoder(self, x_real: torch.Tensor):
        if self.cfg.lr_encoder is None:
            with torch.no_grad():
                z = _encode_mean(self.encoder, x_real, self.z_dim)
        else:
            z = _encode_mean(self.encoder, x_real, self.z_dim)
        x_hat = self.decoder(z)
        # Option C: global (default) vs local z-score for the transform that
        # feeds BOTH the BCE anchor (`logits`) and the regressor (`x_soft`).
        if self.cfg.zscore_mode == "local":
            logits = normalize_to_logits_local(
                x_hat, self.cfg.temperature,
                kernel=self.cfg.local_zscore_kernel,
                eps_std=self.cfg.eps_std,
                logit_clamp=self.cfg.logit_clamp,
            )
        else:
            logits = normalize_to_logits(
                x_hat, self.cfg.temperature,
                eps_std=self.cfg.eps_std,
                logit_clamp=self.cfg.logit_clamp,
            )
        x_soft = torch.sigmoid(logits)
        # `z` is returned too so training_step can feed the frozen raw decoder
        # for the Option-A anchor without recomputing the encoder.
        return x_hat, logits, x_soft, z

    # ------------------------------------------------------------------
    # train / val
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x_real, y_true = batch
        x_hat, logits, x_soft, z = self._forward_decoder(x_real)
        y_pred_raw = self.regressor.predict_raw(x_soft)

        l_bce = bce_pixel_anchor(logits, x_real)
        l_reg = regressor_huber(
            y_pred_raw, y_true,
            y_std=self.regressor.y_std,
            cfg=self.huber_cfg,
        )
        w: LossWeights = weights_at_step(self.global_step, self.schedule_cfg)
        loss = w.bce * l_bce + w.reg * l_reg

        bs = x_real.size(0)

        # --- anti-drift terms (s8 binarization-blob fixes) -------------------
        # All default-off (weight 0.0). They are added at a CONSTANT weight,
        # i.e. outside the BCE-warmup->ramp schedule, because the drift they
        # fight is present throughout the regressor phase. See loss.py.
        if self.cfg.w_field_flatness > 0.0:
            # Option B: penalize the low-frequency field directly.
            l_flat = field_flatness_penalty(
                x_hat, kernel=self.cfg.field_flatness_kernel)
            loss = loss + self.cfg.w_field_flatness * l_flat
            self.log("train/loss_flat", l_flat, on_step=True,
                     batch_size=bs, sync_dist=False)
        if self.cfg.w_raw_anchor > 0.0:
            # Option A: pin x_hat to the frozen pre-SFT decoder output. `z` is
            # reused from the forward pass; decoder_raw is frozen and the call
            # is under no_grad, so the reference carries no gradient.
            with torch.no_grad():
                x_hat_raw = self.decoder_raw(z)
            l_anchor = raw_consistency_anchor(
                x_hat, x_hat_raw,
                lowpass_kernel=self.cfg.raw_anchor_lowpass_kernel)
            loss = loss + self.cfg.w_raw_anchor * l_anchor
            self.log("train/loss_anchor", l_anchor, on_step=True,
                     batch_size=bs, sync_dist=False)

        self.log("train/loss", loss, on_step=True, prog_bar=True,
                 batch_size=bs, sync_dist=False)
        self.log("train/loss_bce", l_bce, on_step=True,
                 batch_size=bs, sync_dist=False)
        self.log("train/loss_reg", l_reg, on_step=True,
                 batch_size=bs, sync_dist=False)
        self.log("train/w_bce", w.bce, on_step=True, batch_size=bs)
        self.log("train/w_reg", w.reg, on_step=True, batch_size=bs)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x_real, y_true = batch
        z = _encode_mean(self.encoder, x_real, self.z_dim)
        x_hat = self.decoder(z)
        # Eval binarization must match the training z-score mode: a model
        # trained with the local z-score has its decision surface at the
        # local mean, not the global one (see regressor.py Option C).
        if self.cfg.zscore_mode == "local":
            x_recon = deterministic_binary_local(
                x_hat, kernel=self.cfg.local_zscore_kernel)
        else:
            x_recon = deterministic_binary(x_hat, eps_std=self.cfg.eps_std)
        y_pred_raw = self.regressor.predict_raw(x_recon)

        mae = (y_pred_raw - y_true).abs()  # [B, T] raw units
        z_err = regressor_z_err_mean(y_pred_raw, y_true, self.regressor.y_std)
        por_orig = (1.0 - x_real.mean(dim=(1, 2, 3, 4))).mean()
        por_recon = (1.0 - x_recon.mean(dim=(1, 2, 3, 4))).mean()

        bs = x_real.size(0)
        self.log("val/mae_true_z_mean", z_err,
                 on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        for ti, name in enumerate(TARGET_NAMES):
            self.log(f"val/mae_true_{name}", mae[:, ti].mean(),
                     on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("val/porosity_orig", por_orig,
                 on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("val/porosity_recon", por_recon,
                 on_epoch=True, batch_size=bs, sync_dist=True)

    # ------------------------------------------------------------------
    # optim
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        params = [{
            "params": [p for p in self.decoder.parameters() if p.requires_grad],
            "lr": self.cfg.lr_decoder,
            "weight_decay": self.cfg.weight_decay,
        }]
        if self.cfg.lr_encoder is not None:
            params.append({
                "params": [p for p in self.encoder.parameters() if p.requires_grad],
                "lr": self.cfg.lr_encoder,
                "weight_decay": self.cfg.weight_decay,
            })
        opt = torch.optim.AdamW(params, betas=self.cfg.betas)
        if not self.cfg.cosine_decay:
            return opt

        import math
        hold = self.cfg.warmup_bce_steps + self.cfg.ramp_steps
        total = max(hold + 1, self.cfg.total_steps)
        mr = self.cfg.min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < hold:
                return 1.0
            t = (step - hold) / max(1, total - hold)
            t = min(max(t, 0.0), 1.0)
            return mr + 0.5 * (1.0 - mr) * (1.0 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        rank = self.trainer.global_rank if self.trainer is not None else 0
        world_size = self.trainer.world_size if self.trainer is not None else 1
        ds = CachedChunkSampler(
            split="train",
            augment=True,
            seed_base=self.cfg.seed,
            rank=rank,
            world_size=world_size,
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_cached,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self):
        ds = EvalPackDataset(n=self.cfg.eval_n_chunks, seed=0)
        return DataLoader(
            ds,
            batch_size=self.cfg.eval_batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )

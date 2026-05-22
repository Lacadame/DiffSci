"""Held-out evaluation of an SFT-tuned decoder against the regressor's cached
true morphological targets.

Standalone CLI (no Lightning trainer needed). Loads the decoder + regressor
state from a `*.ckpt` produced by `pipelines/vae/train_vae_sft.py`
(or any equivalent SFT trainer), runs the deterministic threshold roundtrip
on N held-out chunks, and reports:

  - per-target MAE between regressor(x_recon) and cached truth (TRUE error)
  - per-target MAE between regressor(x_recon) and regressor(x_real) (PROXY)
  - per-target MAE between regressor(x_real)  and cached truth (REGRESSOR own)
  - mean original / reconstructed porosity
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from diffsci2.vaesft import VAESFTModule, deterministic_binary
from diffsci2.vaesft._paths import (
    DEFAULT_REGRESSOR_TEST_CHUNKS, DEFAULT_REGRESSOR_TEST_TARGETS,
)
from poreregressor.chunk_index import load_chunk_index
from poreregressor.data_constants import STONES, CHUNK_SIZE, TARGET_NAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Default: alongside ckpt.")
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _select_test_chunks(n: int, rng: np.random.Generator):
    cs = load_chunk_index(DEFAULT_REGRESSOR_TEST_CHUNKS)
    targets = np.load(DEFAULT_REGRESSOR_TEST_TARGETS).astype(np.float32)
    keep = np.isfinite(targets).all(axis=1)
    idx = np.flatnonzero(keep)
    take = rng.choice(idx, size=min(n, len(idx)), replace=False)
    take.sort()
    return {
        "stone_idx": cs.stone_idx[take],
        "i": cs.i[take], "j": cs.j[take], "k": cs.k[take],
        "targets": targets[take],
    }


def _open_volumes():
    return [np.memmap(c.raw_path, dtype=np.uint8, mode="r", shape=c.shape)
            for c in STONES]


def _load_chunk(volumes, s, i, j, k):
    sub = np.asarray(
        volumes[s][i:i + CHUNK_SIZE, j:j + CHUNK_SIZE, k:k + CHUNK_SIZE]
    ).copy()
    if sub.dtype != np.bool_ and sub.max() > 1:
        sub = (sub > 0).astype(np.uint8)
    return sub.astype(np.float32)


def main():
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or os.path.dirname(args.ckpt)
    os.makedirs(out_dir, exist_ok=True)

    module = VAESFTModule.load_sft_checkpoint(args.ckpt, map_location="cpu")
    module.eval()
    module.to(device)
    encoder = module.encoder
    decoder = module.decoder
    regressor = module.regressor
    z_dim = module.z_dim
    eps_std = module.cfg.eps_std

    sel = _select_test_chunks(args.n_samples, rng)
    volumes = _open_volumes()
    n = len(sel["stone_idx"])

    y_true = sel["targets"]
    y_pred_orig_list, y_pred_recon_list = [], []
    por_orig_list, por_recon_list = [], []

    bs = 8
    for start in range(0, n, bs):
        idxs = range(start, min(start + bs, n))
        chunks_np = np.stack([
            _load_chunk(volumes, int(sel["stone_idx"][i]),
                        int(sel["i"][i]), int(sel["j"][i]), int(sel["k"][i]))
            for i in idxs
        ], axis=0)
        x_real = torch.from_numpy(chunks_np).unsqueeze(1).to(device)

        with torch.no_grad():
            z = encoder(x_real)[:, :z_dim]
            x_hat = decoder(z)
            x_recon = deterministic_binary(x_hat, eps_std=eps_std)
            y_orig = regressor.predict_raw(x_real)
            y_rec = regressor.predict_raw(x_recon)

        y_pred_orig_list.append(y_orig.cpu().numpy())
        y_pred_recon_list.append(y_rec.cpu().numpy())
        por_orig_list.append((1.0 - x_real.mean(dim=(1, 2, 3, 4))).cpu().numpy())
        por_recon_list.append((1.0 - x_recon.mean(dim=(1, 2, 3, 4))).cpu().numpy())

    y_pred_orig = np.concatenate(y_pred_orig_list, axis=0)
    y_pred_recon = np.concatenate(y_pred_recon_list, axis=0)
    por_orig = np.concatenate(por_orig_list, axis=0)
    por_recon = np.concatenate(por_recon_list, axis=0)

    mae_true = np.abs(y_pred_recon - y_true).mean(axis=0)
    mae_proxy = np.abs(y_pred_recon - y_pred_orig).mean(axis=0)
    mae_regressor = np.abs(y_pred_orig - y_true).mean(axis=0)

    print("\n=== Held-out eval ===")
    print(f"  N chunks: {n}")
    print(f"  porosity   orig mean={por_orig.mean():.4f} "
          f"recon mean={por_recon.mean():.4f}")
    for t, name in enumerate(TARGET_NAMES):
        print(f"  {name:>22s}  MAE_true={mae_true[t]:.4e}  "
              f"MAE_proxy={mae_proxy[t]:.4e}  "
              f"MAE_regressor={mae_regressor[t]:.4e}")

    np.savez(
        os.path.join(out_dir, "eval_predictions.npz"),
        y_true=y_true, y_pred_orig=y_pred_orig, y_pred_recon=y_pred_recon,
        porosity_orig=por_orig, porosity_recon=por_recon,
    )
    with open(os.path.join(out_dir, "eval_metrics.csv"), "w") as f:
        f.write("target,MAE_true,MAE_proxy,MAE_regressor\n")
        for t, name in enumerate(TARGET_NAMES):
            f.write(f"{name},{mae_true[t]},{mae_proxy[t]},{mae_regressor[t]}\n")
    print(f"\nSaved CSV + npz to {out_dir}")


if __name__ == "__main__":
    main()

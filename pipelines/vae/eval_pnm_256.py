"""256^3 reconstruction + PNM-permeability evaluation of an SFT-tuned (or
raw) decoder.

Adapts `notebooks/exploratory/dfnai/0003-vae-reconstruction-metrics-fine-tuned.ipynb`
into a CLI script. For each of the four Imperial College stones, it picks
the first `--n-samples` rows from
`notebooks/exploratory/dfnai/0003-data/<stone>_metrics.csv` (matching origins
by `sample_idx`), encode-decodes each 256^3 subvolume through the chosen
decoder, and computes the six scalar metrics:

    porosity, surface_area_density, mean_pore_size, mean_curvature,
    euler_number_density, K_abs_mean_physical (PNM via SNOW2)

The "original" and pre-trained "reconstructed" rows are reused from the
existing CSV — we only run the SFT (or raw) decoder + recompute its recon
metrics.

Output:
    <PLOT_DIR>/<run-name>/eval_pnm_256/<stone>_<label>_metrics.csv  (per-stone)
    <PLOT_DIR>/<run-name>/eval_pnm_256/all_stones_three_way.csv     (merged)
    <PLOT_DIR>/<run-name>/eval_pnm_256/*.png                        (figures)

Usage:
    /opt/persistence/miniconda3/envs/ddpm_env/bin/python \
        pipelines/vae/eval_pnm_256.py \
            --ckpt savedmodels/vae/vae_pixnorm_s8_sft.ckpt \
            --mode raw --n-samples 8 --run-name run04_pixnorm_s8
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffsci2.extra.pore.morphological_metrics import MorphologicalMetrics
from diffsci2.extra.pore.permeability_from_pnm import PoreNetworkPermeability
from diffsci2.vaesft import VAESFTModule, load_autoencoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402  (host-specific PLOT_DIR / EXISTING_CSV_DIR)

# `diffsci2.vaesft._paths` bootstraps poreregressor onto sys.path.
from diffsci2.vaesft._paths import _POREREGRESSOR_PARENT  # noqa: F401
from poreregressor.data_constants import STONES


# ---------------------------------------------------------------------------
# Stone configs (mirror the notebook — physical voxel sizes, not voxel=1.0)
# ---------------------------------------------------------------------------

STONES_PHYSICAL = {
    "Bentheimer":  {"voxel_size": 3.0035e-6},
    "Doddington":  {"voxel_size": 2.6929e-6},
    "Estaillades": {"voxel_size": 3.31136e-6},
    "Ketton":      {"voxel_size": 3.00006e-6},
}

METRIC_COLS = [
    "porosity",
    "surface_area_density",
    "mean_pore_size",
    "mean_curvature",
    "euler_number_density",
    "K_abs_mean_physical",
]

EXISTING_CSV_DIR = Path(paths.EXISTING_CSV_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="SFT Lightning ckpt (mode=sft) or raw VAE ckpt (mode=raw).")
    p.add_argument("--mode", choices=["sft", "raw"], default="sft")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None,
                   help="Sub-folder under <PLOT_DIR>. Defaults to ckpt parent name.")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--subvol-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NP_INT_RE = re.compile(r"np\.int(?:32|64)\((-?\d+)\)")


def parse_origin(s: str) -> tuple[int, int, int]:
    """Parse strings like '(np.int64(66), np.int64(576), np.int64(487))'."""
    clean = _NP_INT_RE.sub(r"\1", s)
    o = ast.literal_eval(clean)
    return tuple(int(x) for x in o)


def load_raw_volume(path: str, shape: tuple, dtype=np.uint8) -> np.ndarray:
    vol = np.fromfile(path, dtype=dtype).reshape(shape)
    if vol.max() > 1:
        vol = (vol > 0)
    return vol.astype(bool)


def extract_subvolume(volume: np.ndarray, origin: tuple, size: int) -> np.ndarray:
    i, j, k = origin
    return volume[i:i + size, j:j + size, k:k + size]


@torch.no_grad()
def encode_decode(encoder: torch.nn.Module, decoder: torch.nn.Module,
                  z_dim: int, binary_subvol: np.ndarray,
                  device: torch.device) -> np.ndarray:
    x = torch.from_numpy(binary_subvol.astype(np.float32))
    x = x.unsqueeze(0).unsqueeze(0).to(device)
    z = encoder(x)[:, :z_dim]
    x_hat = decoder(z)
    x_hat_np = x_hat.squeeze().detach().cpu().numpy()
    return (x_hat_np > x_hat_np.mean()).astype(bool)


def compute_metrics(binary_vol: np.ndarray, voxel_size: float) -> dict:
    vol_int = binary_vol.astype(np.uint8)
    morph = MorphologicalMetrics(vol_int, voxel_size=voxel_size)
    out = {}
    out["porosity"] = morph.porosity()
    out["surface_area_density"] = morph.surface_area_density()
    out["mean_pore_size"] = morph.mean_pore_size().mean
    out["mean_curvature"] = morph.curvature().mean
    out["euler_number_density"] = morph.euler_number_density()
    try:
        pnp = PoreNetworkPermeability.from_binary_volume(vol_int, voxel_size=voxel_size)
        K = pnp.calculate_absolute_permeability()
        out["K_abs_mean_physical"] = K.K_mean_physical
    except Exception as e:
        warnings.warn(f"PNM permeability failed: {e}")
        out["K_abs_mean_physical"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sft_module(ckpt_path: str, device: torch.device) -> VAESFTModule:
    module = VAESFTModule.load_sft_checkpoint(ckpt_path, map_location="cpu")
    module.eval()
    return module.to(device)


def load_raw_vae(ckpt_path: str, device: torch.device):
    """Return ``(encoder, decoder, z_dim)`` for a raw VAE ckpt (any of the
    five formats handled by `diffsci2.vaesft.load_autoencoder`)."""
    vae = load_autoencoder(path=ckpt_path, device=device)
    return vae.encoder, vae.decoder, vae.config.z_dim


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(three_way: pd.DataFrame, out_dir: Path,
                 new_label: str = "sft_reconstructed",
                 new_label_display: str = "SFT"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stones = sorted(three_way["stone"].unique())
    stone_color = dict(zip(stones, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, metric in zip(axes.flat, METRIC_COLS):
        for stone in stones:
            sub = three_way[three_way["stone"] == stone]
            o = sub[sub["label"] == "original"].set_index("sample_idx")[metric]
            p = sub[sub["label"] == "reconstructed"].set_index("sample_idx")[metric]
            s = sub[sub["label"] == new_label].set_index("sample_idx")[metric]
            common = o.index.intersection(p.index).intersection(s.index)
            if len(common) == 0:
                continue
            ax.scatter(o.loc[common], p.loc[common], marker="o", s=50,
                       edgecolor=stone_color[stone], facecolor="none",
                       linewidth=1.4,
                       label=f"{stone} (pretrained)" if metric == METRIC_COLS[0] else None)
            ax.scatter(o.loc[common], s.loc[common], marker="x", s=55,
                       color=stone_color[stone], linewidth=1.5,
                       label=f"{stone} ({new_label_display})" if metric == METRIC_COLS[0] else None)
        lims_x = ax.get_xlim()
        lims_y = ax.get_ylim()
        lo = min(lims_x[0], lims_y[0])
        hi = max(lims_x[1], lims_y[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("original"); ax.set_ylabel("reconstructed")
        ax.set_title(metric); ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle(f"256^3 reconstruction: original vs reconstructed "
                 f"(circles = pretrained, x = {new_label_display})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_three_way.png", dpi=140)
    plt.close(fig)

    rows = []
    for stone in stones:
        sub = three_way[three_way["stone"] == stone]
        for lab in ["reconstructed", new_label]:
            for metric in METRIC_COLS:
                o = sub[sub["label"] == "original"].set_index("sample_idx")[metric]
                r = sub[sub["label"] == lab].set_index("sample_idx")[metric]
                idx = o.index.intersection(r.index)
                if len(idx) == 0:
                    continue
                a = o.loc[idx].values.astype(float)
                b = r.loc[idx].values.astype(float)
                if metric == "K_abs_mean_physical":
                    mask = (a > 0) & (b > 0)
                    a, b = np.log(a[mask]), np.log(b[mask])
                    rel = (b - a) / np.where(np.abs(a) > 0, np.abs(a), np.nan)
                else:
                    rel = (b - a) / np.where(np.abs(a) > 0, np.abs(a), np.nan)
                for v in rel:
                    rows.append({"stone": stone, "label": lab, "metric": metric,
                                 "rel_error": float(v)})
    rel_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, metric in zip(axes.flat, METRIC_COLS):
        sub = rel_df[rel_df["metric"] == metric]
        positions, ticks = [], []
        pos = 0
        for stone in stones:
            for lab, color, w in (("reconstructed", "#888", 0.0),
                                  (new_label, "#1f77b4", 0.45)):
                vals = sub[(sub["stone"] == stone) & (sub["label"] == lab)]["rel_error"].dropna().values
                if len(vals) == 0:
                    continue
                ax.boxplot([vals], positions=[pos + w], widths=0.4,
                           patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.6),
                           medianprops=dict(color="black"),
                           flierprops=dict(marker=".", markersize=4))
            positions.append(pos + 0.22); ticks.append(stone)
            pos += 1.2
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(positions); ax.set_xticklabels(ticks, rotation=20)
        ax.set_title(metric)
        ax.set_ylabel("rel. error (recon - orig)/|orig|")
        ax.grid(alpha=0.25, axis="y")
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#888", alpha=0.6, label="pretrained decoder"),
              Patch(facecolor="#1f77b4", alpha=0.6, label=f"{new_label_display} decoder")]
    fig.legend(handles=legend, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"Paired relative error per metric and stone "
                 f"(pretrained vs {new_label_display})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "rel_error_boxes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    agg = (
        rel_df.assign(abs_rel=lambda d: d["rel_error"].abs())
              .groupby(["stone", "label", "metric"])["abs_rel"]
              .agg(["mean", "std", "count"])
              .reset_index()
    )
    agg.to_csv(out_dir / "rel_error_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)

    run_name = args.run_name or Path(args.ckpt).parent.name
    out_dir = Path(paths.PLOT_DIR) / run_name / "eval_pnm_256"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_pnm_256] output -> {out_dir}")
    print(f"[eval_pnm_256] ckpt   -> {args.ckpt}  (mode={args.mode})")

    if args.mode == "sft":
        module = load_sft_module(args.ckpt, device)
        encoder, decoder, z_dim = module.encoder, module.decoder, module.z_dim
        default_label = "sft_reconstructed"
        print(f"[eval_pnm_256] SFT module loaded on {device}  z_dim={z_dim}")
    else:
        encoder, decoder, z_dim = load_raw_vae(args.ckpt, device)
        default_label = "raw_reconstructed"
        print(f"[eval_pnm_256] raw VAE loaded on {device}  z_dim={z_dim}")
    label = args.label or default_label

    sft_rows: list[dict] = []
    three_way_rows: list[dict] = []

    for stone_name, phys in STONES_PHYSICAL.items():
        stone_csv = EXISTING_CSV_DIR / f"{stone_name}_metrics.csv"
        if not stone_csv.exists():
            print(f"[skip] {stone_csv} missing")
            continue
        df = pd.read_csv(stone_csv)

        orig = df[df["label"] == "original"].set_index("sample_idx")
        pre = df[df["label"] == "reconstructed"].set_index("sample_idx")
        idxs = sorted(orig.index.intersection(pre.index).tolist())[:args.n_samples]
        if not idxs:
            print(f"[skip] {stone_name}: no paired samples in {stone_csv}")
            continue

        stone_cfg = next((c for c in STONES if c.name == stone_name), None)
        if stone_cfg is None:
            print(f"[skip] {stone_name}: not found in STONES")
            continue

        print(f"\n=== {stone_name} ({len(idxs)} samples, voxel={phys['voxel_size']*1e6:.4f} um) ===")
        vol = load_raw_volume(stone_cfg.raw_path, stone_cfg.shape)

        for k, sample_idx in enumerate(idxs):
            origin = parse_origin(orig.loc[sample_idx, "origin"])
            t0 = time.time()
            subvol = extract_subvolume(vol, origin, args.subvol_size).copy()
            print(f"  [{k+1}/{len(idxs)}] sample_idx={sample_idx} origin={origin}",
                  flush=True)

            recon = encode_decode(encoder, decoder, z_dim, subvol, device)
            recon_metrics = compute_metrics(recon, phys["voxel_size"])
            elapsed = time.time() - t0
            print(f"    recon  phi={recon_metrics['porosity']:.4f}  "
                  f"K={recon_metrics['K_abs_mean_physical']:.3e}  "
                  f"({elapsed:.1f}s)",
                  flush=True)

            row_sft = {
                "stone": stone_name, "sample_idx": int(sample_idx),
                "origin": str(origin), "label": label,
                **recon_metrics,
            }
            sft_rows.append(row_sft)

            three_way_rows.append({
                "stone": stone_name, "sample_idx": int(sample_idx),
                "origin": str(origin), "label": "original",
                **{m: float(orig.loc[sample_idx, m]) for m in METRIC_COLS},
            })
            three_way_rows.append({
                "stone": stone_name, "sample_idx": int(sample_idx),
                "origin": str(origin), "label": "reconstructed",
                **{m: float(pre.loc[sample_idx, m]) for m in METRIC_COLS},
            })
            three_way_rows.append(row_sft)

        stone_sft_df = pd.DataFrame([r for r in sft_rows if r["stone"] == stone_name])
        per_stone_csv = out_dir / f"{stone_name}_{label}_metrics.csv"
        stone_sft_df.to_csv(per_stone_csv, index=False)
        print(f"    wrote {per_stone_csv}")
        del vol

    three_way = pd.DataFrame(three_way_rows)
    three_way.to_csv(out_dir / "all_stones_three_way.csv", index=False)
    print(f"\n[eval_pnm_256] wrote {out_dir / 'all_stones_three_way.csv'} "
          f"({len(three_way)} rows)")

    display = {"sft_reconstructed": "SFT",
               "raw_reconstructed": "pixnorm raw"}.get(label, label)
    make_figures(three_way, out_dir,
                 new_label=label, new_label_display=display)
    print(f"[eval_pnm_256] figures saved to {out_dir}")


if __name__ == "__main__":
    main()

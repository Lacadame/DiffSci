#!/usr/bin/env python3
"""Simple, editable-in-file EDM CIFAR-10 trainer launcher.

Edit the configuration variables below, then run:
    python scripts/training/train-edm-cifar10.py
"""

import math
import pathlib
import shlex
import shutil
import subprocess
import sys
import io
import json
import os
import zipfile
import time
import shutil as shutil_files

import torchvision
from torch.utils.tensorboard import SummaryWriter
# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

# Data and output.
# Your scripts/testing/data currently contains CIFAR python batches, which are
# not directly consumable by EDM train.py. Keep AUTO_PREPARE_DATASET=True to
# auto-download CIFAR-10 and build an EDM-compatible ZIP when needed.
AUTO_PREPARE_DATASET = True
RAW_CIFAR_ROOT = "/home/ubuntu/repos/DiffSci/scripts/testing/data"
DATA_PATH = "/home/ubuntu/repos/DiffSci/scripts/testing/data/cifar10-32x32.zip"  # EDM-format dataset ZIP|DIR.
CHECKPOINT_DIR = "/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260323-bps-karras-cifar10"

# Distributed setup.
TORCHRUN_BIN = "torchrun"  # If unavailable, script falls back to "python -m torch.distributed.run".
NPROC_PER_NODE = 5
MASTER_PORT = 29500
USE_STANDALONE = True
CUDA_VISIBLE_DEVICES = "1,2,3,4,5"  # Example: "1,2,3,4,5" on DGX.

# Model/training setup (mirrors EDM train.py options).
COND = False
ARCH = "ddpmpp"  # ddpmpp | ncsnpp | adm
PRECOND = "edm"  # vp | ve | edm
EPOCHS = 200
BATCH = 500
BATCH_GPU = None
CBASE = None
CRES = None  # Example: "1,2,2,2"
LR = 1e-3
EMA = 0.5
DROPOUT = 0.13
AUGMENT = 0.12
XFLIP = False
FP16 = False
LS = 1.0
BENCH = True
CACHE = True
WORKERS = 1
SEED = None
TRANSFER = None
RESUME = None
DESC = None

# CIFAR-10 train split size; used to convert epochs -> EDM kimg schedule.
TRAINSET_SIZE = 50_000
INCLUDE_TEST_SPLIT = False  # False => standard 50k train images.

# Save checkpoints every K epochs.
EVERY_K_EPOCHS = 20

# Utility flags.
DRY_RUN = False             # For EDM --dry-run.
PRINT_COMMAND_ONLY = False  # Print resolved command and exit.

# Testing-script compatibility helpers.
EXPORT_LATEST_SNAPSHOT_ALIAS = True
LATEST_SNAPSHOT_ALIAS_NAME = "network-snapshot-latest.pkl"

# TensorBoard logging bridge (EDM stats.jsonl -> TensorBoard events).
ENABLE_TENSORBOARD_LOGS = True
TENSORBOARD_POLL_SECONDS = 2.0

# -----------------------------------------------------------------------------


def _bool01(value: bool) -> str:
    return "1" if value else "0"


def _append_opt(cmd: list[str], name: str, value) -> None:
    if value is None:
        return
    cmd.append(f"--{name}={value}")


def _epoch_to_kimg(epochs: float, trainset_size: int) -> int:
    return max(1, math.ceil(epochs * trainset_size / 1000.0))


def _is_edm_compatible_dataset(path: pathlib.Path, require_labels: bool) -> bool:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    if path.is_dir():
        image_count = sum(
            1
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )
        if image_count == 0:
            return False
        if require_labels and not (path / "dataset.json").is_file():
            return False
        return True

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
        image_count = sum(1 for n in names if pathlib.Path(n).suffix.lower() in image_exts)
        if image_count == 0:
            return False
        if require_labels and "dataset.json" not in names:
            return False
        return True

    return False


def _prepare_edm_cifar10_zip(
    raw_root: pathlib.Path,
    output_zip: pathlib.Path,
    include_test_split: bool,
) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    train_ds = torchvision.datasets.CIFAR10(root=str(raw_root), train=True, download=True)
    datasets = [train_ds]
    if include_test_split:
        test_ds = torchvision.datasets.CIFAR10(root=str(raw_root), train=False, download=True)
        datasets.append(test_ds)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        labels = []
        idx = 0
        for ds in datasets:
            for image, label in ds:
                img_name = f"{idx // 10000:05d}/img{idx:08d}.png"
                with io.BytesIO() as buffer:
                    image.save(buffer, format="PNG")
                    zf.writestr(img_name, buffer.getvalue())
                labels.append([img_name, int(label)])
                idx += 1

        zf.writestr("dataset.json", json.dumps({"labels": labels}))

    print(f"Prepared EDM dataset ZIP: {output_zip} ({idx} images)")


def _next_lightning_version_dir(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    base = checkpoint_dir / "lightning_logs"
    base.mkdir(parents=True, exist_ok=True)
    version_ids = []
    for p in base.glob("version_*"):
        try:
            version_ids.append(int(p.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    next_id = 0 if not version_ids else max(version_ids) + 1
    run_dir = base / f"version_{next_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _try_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stream_edm_stats_to_tensorboard(
    process: subprocess.Popen,
    stats_jsonl_path: pathlib.Path,
    writer: SummaryWriter,
    poll_seconds: float = 2.0,
) -> None:
    offset = 0
    while process.poll() is None:
        if stats_jsonl_path.is_file():
            with open(stats_jsonl_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    step = _try_float(rec.get("Progress/kimg"))
                    if step is None:
                        continue
                    global_step = int(step * 1000)

                    for k, v in rec.items():
                        if k == "timestamp":
                            continue
                        scalar = _try_float(v)
                        if scalar is not None:
                            writer.add_scalar(k, scalar, global_step=global_step)
                offset = f.tell()
                writer.flush()
        time.sleep(poll_seconds)

    # Final flush pass after process exits.
    if stats_jsonl_path.is_file():
        with open(stats_jsonl_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                step = _try_float(rec.get("Progress/kimg"))
                if step is None:
                    continue
                global_step = int(step * 1000)

                for k, v in rec.items():
                    if k == "timestamp":
                        continue
                    scalar = _try_float(v)
                    if scalar is not None:
                        writer.add_scalar(k, scalar, global_step=global_step)
        writer.flush()


def _write_configs_txt(checkpoint_dir: pathlib.Path, total_kimg: int, checkpoint_kimg: int) -> None:
    config_txt = checkpoint_dir / "configs.txt"
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write(f"Data path: {DATA_PATH}\n")
        f.write(f"Checkpoint directory: {CHECKPOINT_DIR}\n")
        f.write(f"Conditional: {COND}\n")
        f.write(f"Architecture: {ARCH}\n")
        f.write(f"Preconditioning: {PRECOND}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch size: {BATCH}\n")
        f.write(f"Batch per GPU: {BATCH_GPU}\n")
        f.write(f"Learning rate: {LR}\n")
        f.write(f"EMA: {EMA}\n")
        f.write(f"Dropout: {DROPOUT}\n")
        f.write(f"Augment: {AUGMENT}\n")
        f.write(f"FP16: {FP16}\n")
        f.write(f"Workers: {WORKERS}\n")
        f.write(f"NPROC_PER_NODE: {NPROC_PER_NODE}\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}\n")
        f.write(f"Total kimg: {total_kimg}\n")
        f.write(f"Checkpoint every kimg: {checkpoint_kimg}\n")
        f.write(f"Checkpoint every epochs: {EVERY_K_EPOCHS}\n")


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    edm_dir = repo_root / "external_repos" / "edm"
    train_script = edm_dir / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"EDM train script not found: {train_script}")

    total_kimg = _epoch_to_kimg(EPOCHS, TRAINSET_SIZE)
    checkpoint_kimg = _epoch_to_kimg(EVERY_K_EPOCHS, TRAINSET_SIZE)

    if BATCH % NPROC_PER_NODE != 0:
        lower = (BATCH // NPROC_PER_NODE) * NPROC_PER_NODE
        upper = ((BATCH + NPROC_PER_NODE - 1) // NPROC_PER_NODE) * NPROC_PER_NODE
        raise ValueError(
            "Invalid distributed batch config for EDM: "
            f"BATCH ({BATCH}) must be divisible by NPROC_PER_NODE ({NPROC_PER_NODE}). "
            f"Try BATCH={lower} or BATCH={upper}."
        )

    # Configure EDM so that one tick corresponds to checkpoint interval.
    # Then snapshot/state dump every tick.
    tick_kimg = checkpoint_kimg
    snap_ticks = 1
    dump_ticks = 1

    dataset_path = pathlib.Path(DATA_PATH)
    if AUTO_PREPARE_DATASET:
        is_compatible = dataset_path.exists() and _is_edm_compatible_dataset(dataset_path, COND)
        if not is_compatible:
            print(f"Dataset at {dataset_path} is missing/incompatible. Preparing EDM CIFAR-10 ZIP...")
            _prepare_edm_cifar10_zip(
                raw_root=pathlib.Path(RAW_CIFAR_ROOT),
                output_zip=dataset_path,
                include_test_split=INCLUDE_TEST_SPLIT,
            )
    else:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"DATA_PATH does not exist: {dataset_path}. "
                "Set AUTO_PREPARE_DATASET=True to generate it automatically."
            )
        if not _is_edm_compatible_dataset(dataset_path, COND):
            raise ValueError(
                f"DATA_PATH is not EDM-compatible: {dataset_path}. "
                "Expected images (ZIP|DIR), and dataset.json for class-conditional training."
            )

    if shutil.which(TORCHRUN_BIN):
        cmd: list[str] = [TORCHRUN_BIN]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run"]

    if USE_STANDALONE:
        cmd.append("--standalone")
    cmd.extend(
        [
            f"--nproc_per_node={NPROC_PER_NODE}",
            f"--master_port={MASTER_PORT}",
            str(train_script),
        ]
    )

    _append_opt(cmd, "outdir", CHECKPOINT_DIR)
    _append_opt(cmd, "data", str(dataset_path))
    _append_opt(cmd, "cond", _bool01(COND))
    _append_opt(cmd, "arch", ARCH)
    _append_opt(cmd, "precond", PRECOND)
    _append_opt(cmd, "duration", total_kimg / 1000.0)  # EDM expects MIMG.
    _append_opt(cmd, "batch", BATCH)
    _append_opt(cmd, "batch-gpu", BATCH_GPU)
    _append_opt(cmd, "cbase", CBASE)
    _append_opt(cmd, "cres", CRES)
    _append_opt(cmd, "lr", LR)
    _append_opt(cmd, "ema", EMA)
    _append_opt(cmd, "dropout", DROPOUT)
    _append_opt(cmd, "augment", AUGMENT)
    _append_opt(cmd, "xflip", _bool01(XFLIP))
    _append_opt(cmd, "fp16", _bool01(FP16))
    _append_opt(cmd, "ls", LS)
    _append_opt(cmd, "bench", _bool01(BENCH))
    _append_opt(cmd, "cache", _bool01(CACHE))
    _append_opt(cmd, "workers", WORKERS)
    _append_opt(cmd, "desc", DESC)
    cmd.append("--nosubdir")  # Save snapshots directly inside CHECKPOINT_DIR.
    _append_opt(cmd, "tick", tick_kimg)
    _append_opt(cmd, "snap", snap_ticks)
    _append_opt(cmd, "dump", dump_ticks)
    _append_opt(cmd, "seed", SEED)
    _append_opt(cmd, "transfer", TRANSFER)
    _append_opt(cmd, "resume", RESUME)
    if DRY_RUN:
        cmd.append("--dry-run")

    print(f"epochs={EPOCHS}, total_kimg={total_kimg}")
    print(f"checkpoint_every_k_epochs={EVERY_K_EPOCHS}, checkpoint_kimg={checkpoint_kimg}")
    if CUDA_VISIBLE_DEVICES is not None:
        visible_devices = [d.strip() for d in CUDA_VISIBLE_DEVICES.split(",") if d.strip()]
        if NPROC_PER_NODE > len(visible_devices):
            raise ValueError(
                "NPROC_PER_NODE is larger than number of CUDA_VISIBLE_DEVICES. "
                f"NPROC_PER_NODE={NPROC_PER_NODE}, visible={visible_devices}"
            )
        print(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}")
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd))

    if PRINT_COMMAND_ONLY:
        return

    run_env = os.environ.copy()
    if CUDA_VISIBLE_DEVICES is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    checkpoint_dir = pathlib.Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_configs_txt(checkpoint_dir, total_kimg, checkpoint_kimg)

    if ENABLE_TENSORBOARD_LOGS:
        tb_run_dir = _next_lightning_version_dir(checkpoint_dir)
        writer = SummaryWriter(log_dir=str(tb_run_dir))
        writer.add_text("run/command", " ".join(shlex.quote(x) for x in cmd), 0)
        writer.add_text("run/data_path", str(dataset_path), 0)
        writer.flush()
        print(f"TensorBoard log dir: {tb_run_dir}")
    else:
        writer = None

    process = subprocess.Popen(cmd, env=run_env)
    stats_jsonl_path = checkpoint_dir / "stats.jsonl"
    if writer is not None:
        _stream_edm_stats_to_tensorboard(
            process=process,
            stats_jsonl_path=stats_jsonl_path,
            writer=writer,
            poll_seconds=TENSORBOARD_POLL_SECONDS,
        )

    return_code = process.wait()
    if writer is not None:
        writer.close()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    # Convenience output for scripts/testing/test-diffusion-cifar10-karras.py.
    snapshot_dir = pathlib.Path(CHECKPOINT_DIR)
    snapshots = sorted(snapshot_dir.glob("network-snapshot-*.pkl"))
    if not snapshots:
        return

    latest_snapshot = snapshots[-1]
    print(f"Latest snapshot: {latest_snapshot}")
    print(
        "Use this in test script as network_url/path:\n"
        f"    {latest_snapshot}"
    )

    if EXPORT_LATEST_SNAPSHOT_ALIAS:
        alias_path = snapshot_dir / LATEST_SNAPSHOT_ALIAS_NAME
        shutil_files.copy2(latest_snapshot, alias_path)
        print(f"Updated alias snapshot: {alias_path}")

    if COND:
        print(
            "Note: training used COND=True (class-conditional). "
            "Your current test script does not pass explicit class one-hot labels."
        )
        print(
            "For strict compatibility with that script, prefer COND=False in training, "
            "or update the test script to provide class labels."
        )


if __name__ == "__main__":
    main()

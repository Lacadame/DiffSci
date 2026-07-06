"""
Parallelised FID-colormap (Langevin-interval grid sweep) for CIFAR-10
unconditional diffusion using the NVIDIA EDM checkpoint via DiffSci.

This is a multi-GPU drop-in replacement for
``scripts/testing/test-diffusion-cifar10karras-colormap.py``. It produces the
same output bundle (``fid_grid``, ``fid_ode``, the ``.npy`` sample previews
and the human-readable ``fid_scores.txt``), but distributes the work across
several GPUs and adds two big efficiency wins on top of the parallelism:

1. **Process-level parallelism across GPUs** (default ``cuda:0``..``cuda:5`` on
   a DGX A100). The ODE task plus all valid ``(s_min, s_max)`` SDE intervals
   are gathered into a single task list and round-robin-distributed across
   ``len(device_ids)`` worker processes, each pinned to its own GPU. Expected
   wall-clock speedup is roughly linear in ``len(device_ids)``.

2. **Cached real Inception features**. The sequential script feeds the full
   real CIFAR-10 dataset through Inception once per interval (so for
   ``ngrid=10`` it does it ~46 times). Here we feed the real images through
   Inception exactly once in the parent process and ship the resulting
   ``(real_features_sum, real_features_cov_sum, real_features_num_samples)``
   tensors to every worker, which copy them into a fresh
   ``FrechetInceptionDistance`` instance per task. This alone removes a large
   constant overhead from every interval evaluation.

3. **TF32 + cuDNN benchmark** are enabled on every worker to take advantage
   of A100 tensor cores during sampling (safe for inference).

The numerical contract matches the sequential script up to per-worker random
state (each worker uses ``seed + worker_id * 10_000``), so the FID grid
values will be statistically equivalent but not bit-identical.
"""

import os
import sys
import time
import pickle
import itertools
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torchmetrics.image import FrechetInceptionDistance
from tqdm import tqdm

# EDM repo for NVIDIA checkpoint loading (must be before importing diffsci if edm is needed)
_EDM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external_repos", "edm")
if os.path.isdir(_EDM_DIR):
    sys.path.insert(0, os.path.abspath(_EDM_DIR))

import diffsci.models  # noqa: E402
from diffsci.models.karras.integrators import EulerIntegrator, EulerMaruyamaIntegrator  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (kept identical in spirit to the sequential script)
# ---------------------------------------------------------------------------

def custom_spacing(min_val, max_val, N, alpha):
    """Power-law spaced points between ``min_val`` and ``max_val``."""
    min_alpha = min_val ** alpha
    max_alpha = max_val ** alpha
    lin_space = torch.linspace(min_alpha, max_alpha, N)
    return lin_space ** (1 / alpha)


def _extract_edm_net_from_pickle(loaded_obj):
    """Extract a torch.nn.Module from common EDM pickle formats."""
    if isinstance(loaded_obj, nn.Module):
        return loaded_obj
    if isinstance(loaded_obj, dict):
        for key in ("ema", "net", "model", "G_ema"):
            maybe_net = loaded_obj.get(key, None)
            if isinstance(maybe_net, nn.Module):
                return maybe_net
    return None


def load_nvidia_edm_cifar10(network_url_or_path, device):
    """Load EDM CIFAR-10 network from URL/path and return torch module on device."""
    import dnnlib

    with dnnlib.util.open_url(network_url_or_path) as f:
        loaded_obj = pickle.load(f)
    net = _extract_edm_net_from_pickle(loaded_obj)
    if net is not None:
        return net.to(device)

    if os.path.isfile(network_url_or_path) and network_url_or_path.endswith((".pt", ".pth")):
        loaded_obj = torch.load(network_url_or_path, map_location="cpu")
        net = _extract_edm_net_from_pickle(loaded_obj)
        if net is not None:
            return net.to(device)

    raise ValueError(
        "Unsupported checkpoint format. Expected an EDM/NVIDIA pickle with a model "
        "under one of keys {ema, net, model, G_ema}, or a raw torch.nn.Module."
    )


class NVIDIAWrapper(nn.Module):
    """Wrap NVIDIA EDM net so DiffSci can call it with (x, sigma) and NullPreconditioner."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, cond_noise, class_labels=None):
        batch_size = x.shape[0]
        if class_labels is None and getattr(self.net, "label_dim", 0) > 0:
            class_labels = torch.zeros((batch_size, self.net.label_dim), device=x.device, dtype=x.dtype)
        return self.net(x, cond_noise, class_labels)


def denormalize_for_fid(img_tensor):
    """Map model output [-1, 1] to [0, 1] for FID."""
    img = (img_tensor + 1) / 2
    return img.clamp(0, 1)


def prepare_for_fid(images_tensor):
    """Convert [0, 1] float tensor (B, C, H, W) to uint8 for FrechetInceptionDistance."""
    if images_tensor.shape[1] == 1:
        images_tensor = images_tensor.repeat(1, 3, 1, 1)
    return (images_tensor.clamp(0, 1) * 255).to(dtype=torch.uint8)


def _enable_a100_speedups():
    """Inference-safe A100 defaults: TF32 + cuDNN benchmark."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Real-feature caching (run once in the parent process)
# ---------------------------------------------------------------------------

def compute_real_features(device, batch_size, num_workers=4):
    """Compute real CIFAR-10 Inception features once and return CPU state.

    Returns
    -------
    state : dict
        CPU tensors with keys ``sum``, ``cov_sum``, ``num_samples`` corresponding
        to the FID metric's running real-feature statistics.
    real_samples_first10 : np.ndarray
        First 10 real samples (saved alongside the FID grid for reference).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    real_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    fid = FrechetInceptionDistance(normalize=False).to(device)
    real_samples_first10 = None
    print(f"[main] Processing real images on {device}...")
    for batch, _ in tqdm(real_loader, desc="real images"):
        batch = batch.to(device)
        fid.update(prepare_for_fid(batch), real=True)
        if real_samples_first10 is None:
            real_samples_first10 = batch[:10].detach().cpu().numpy()

    state = {
        "sum": fid.real_features_sum.detach().cpu().clone(),
        "cov_sum": fid.real_features_cov_sum.detach().cpu().clone(),
        "num_samples": fid.real_features_num_samples.detach().cpu().clone(),
    }
    print(
        f"[main] Cached real features: sum {tuple(state['sum'].shape)}, "
        f"cov_sum {tuple(state['cov_sum'].shape)}, "
        f"num_samples={int(state['num_samples'].item())}")

    del fid
    torch.cuda.empty_cache()
    return state, real_samples_first10


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker_process(
    worker_id,
    device_id,
    task_list,
    real_features_state,
    config_dict,
    output_queue,
):
    """Per-GPU worker: load model once, evaluate every assigned task.

    Each task is either ``{'kind': 'ode'}`` (uses Euler integrator, ignores
    ``smin``/``smax``) or ``{'kind': 'sde', 'smin': float, 'smax': float}``
    (uses Euler-Maruyama with the given Langevin interval).
    """
    try:
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device)
        _enable_a100_speedups()

        seed = int(config_dict["seed"]) + int(worker_id) * 10_000
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- Load model on this GPU ---
        net = load_nvidia_edm_cifar10(config_dict["network_url"], device)
        adapter = NVIDIAWrapper(net)
        moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
        moduleconfig.preconditioner = diffsci.models.karras.preconditioners.NullPreconditioner()
        module = diffsci.models.KarrasModule(adapter, moduleconfig, conditional=False)
        module = module.to(device).eval()
        module.config.noisescheduler.maximum_scale = config_dict["max_scale"]
        module.config.noisescheduler.langevin_const = config_dict["gamma"]

        ode_integrator = EulerIntegrator()
        sde_integrator = EulerMaruyamaIntegrator()

        n_samples = int(config_dict["n_samples"])
        batch_size = int(config_dict["batch_size"])
        nsteps = int(config_dict["nsteps"])
        shape = list(config_dict["shape"])
        num_batches = (n_samples + batch_size - 1) // batch_size

        # Place cached real features on this GPU once.
        real_sum_dev = real_features_state["sum"].to(device)
        real_cov_dev = real_features_state["cov_sum"].to(device)
        real_num_dev = real_features_state["num_samples"].to(device)

        for task in task_list:
            global_idx = task["global_idx"]
            kind = task["kind"]
            smin = task["smin"]
            smax = task["smax"]

            t0 = time.time()

            fid = FrechetInceptionDistance(normalize=False).to(device)
            fid.real_features_sum.copy_(real_sum_dev)
            fid.real_features_cov_sum.copy_(real_cov_dev)
            fid.real_features_num_samples.copy_(real_num_dev)

            if kind == "ode":
                integrator = ode_integrator
            else:
                integrator = sde_integrator
                module.config.noisescheduler.langevin_interval = [smin, smax]

            first10 = None
            with torch.no_grad():
                for i in range(num_batches):
                    curr_batch = min(batch_size, n_samples - i * batch_size)
                    gen_batch = module.sample(
                        nsamples=curr_batch,
                        shape=shape,
                        nsteps=nsteps,
                        integrator=integrator,
                    )
                    gen_batch = denormalize_for_fid(gen_batch)
                    fid.update(prepare_for_fid(gen_batch), real=False)
                    if first10 is None and i == 0:
                        first10 = gen_batch[:10].detach().cpu().numpy()

            score = float(fid.compute().item())
            elapsed = time.time() - t0

            output_queue.put({
                "type": "result",
                "global_idx": global_idx,
                "kind": kind,
                "smin": smin,
                "smax": smax,
                "score": score,
                "first10": first10,
                "worker_id": worker_id,
                "device_id": device_id,
                "elapsed_sec": elapsed,
            })

            del fid
            torch.cuda.empty_cache()

        output_queue.put({"type": "done", "worker_id": worker_id, "device_id": device_id})

    except Exception as exc:  # pragma: no cover - error reporting path
        output_queue.put({
            "type": "error",
            "worker_id": worker_id,
            "device_id": device_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


# ---------------------------------------------------------------------------
# Orchestrator (parent process)
# ---------------------------------------------------------------------------

def main(
    output_dir,
    n_samples=10000,
    device_ids=(0, 1, 2, 3, 4, 5),
    batch_size=500,
    gamma=1.0,
    seed=42,
    model_name="nvidia-edm-cifar10",
    nsteps=200,
    max_scale=80,
    initial_time=1.0,
    tmin=1e-3,
    ngrid=10,
    alpha=0.1,
    interval_grid=None,
    image_size=32,
    network_url="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
    real_data_workers=4,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    device_ids = list(device_ids)
    n_workers = len(device_ids)
    if n_workers < 1:
        raise ValueError("device_ids must contain at least one GPU id.")
    print(f"[main] Distributing work across {n_workers} GPU(s): {device_ids}")
    print(f"[main] {n_samples} samples per task (batch_size={batch_size}, nsteps={nsteps})")

    torch.manual_seed(seed)
    np.random.seed(seed)
    _enable_a100_speedups()

    # Spawn must be set before any worker is started; force=True allows
    # re-running this script in the same interpreter session.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 1. Compute real FID stats ONCE on the first GPU and cache the state.
    real_features_state, real_samples_first10 = compute_real_features(
        device=f"cuda:{device_ids[0]}",
        batch_size=batch_size,
        num_workers=real_data_workers,
    )

    # 2. Build the (s_min, s_max) interval grid (same logic as sequential script).
    if interval_grid is None:
        S_values = custom_spacing(tmin, initial_time, ngrid, alpha=alpha)
        S_values_list = [float(v) for v in S_values]
        interval_grid = list(itertools.product(S_values_list, S_values_list))
        manual_grid = False
    elif len(interval_grid) == ngrid:
        S_values = torch.as_tensor([float(v) for v in interval_grid])
        S_values_list = S_values.tolist()
        interval_grid = list(itertools.product(S_values_list, S_values_list))
        manual_grid = True
    elif len(interval_grid) == ngrid * ngrid:
        S_values = None
        manual_grid = True
        interval_grid = [(float(a), float(b)) for a, b in interval_grid]
    else:
        raise ValueError(
            f"interval_grid has {len(interval_grid)} entries, expected either "
            f"{ngrid} (1D discretization) or {ngrid * ngrid} (full product grid)")

    print(
        f"[main] Interval grid built with {len(interval_grid)} entries "
        f"(ngrid={ngrid}, manual={manual_grid})")

    # 3. Build task list: ODE + valid (smin < smax) SDE intervals.
    tasks = [{"global_idx": -1, "kind": "ode", "smin": None, "smax": None}]
    for idx, (smin, smax) in enumerate(interval_grid):
        if smin >= smax:
            continue
        tasks.append({"global_idx": idx, "kind": "sde", "smin": smin, "smax": smax})
    n_tasks = len(tasks)
    n_sde = n_tasks - 1
    print(f"[main] Total tasks: {n_tasks} (1 ODE + {n_sde} SDE intervals)")

    # 4. Round-robin distribution across workers.
    worker_tasks = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        worker_tasks[i % n_workers].append(task)
    for wid, tlist in enumerate(worker_tasks):
        print(f"  worker {wid} (cuda:{device_ids[wid]}): {len(tlist)} tasks")

    # 5. Spawn one process per GPU and collect results via a Queue.
    output_queue = mp.Queue()
    config_dict = {
        "network_url": network_url,
        "max_scale": max_scale,
        "gamma": gamma,
        "n_samples": n_samples,
        "batch_size": batch_size,
        "nsteps": nsteps,
        "shape": [3, image_size, image_size],
        "seed": seed,
    }

    processes = []
    for wid, dev_id in enumerate(device_ids):
        p = mp.Process(
            target=worker_process,
            args=(wid, dev_id, worker_tasks[wid], real_features_state, config_dict, output_queue),
        )
        p.start()
        processes.append(p)

    # 6. Collect results until every worker reports done (or errors out).
    fid_values = torch.full((ngrid * ngrid,), float("nan"), dtype=torch.float64)
    fid_score_ode = float("nan")
    ode_first10 = None
    elapsed_per_task = []
    n_done = 0
    pbar = tqdm(total=n_tasks, desc="Tasks completed")
    error_msgs = []
    overall_start = time.time()

    while n_done < n_workers:
        msg = output_queue.get()
        msg_type = msg.get("type")
        if msg_type == "done":
            n_done += 1
            print(f"[main] worker {msg['worker_id']} (cuda:{msg['device_id']}) finished")
        elif msg_type == "error":
            n_done += 1
            err = (
                f"worker {msg['worker_id']} (cuda:{msg['device_id']}) failed: "
                f"{msg['error']}\n{msg['traceback']}"
            )
            print(f"[main][ERROR] {err}")
            error_msgs.append(err)
        else:  # 'result'
            pbar.update(1)
            elapsed_per_task.append(msg["elapsed_sec"])
            if msg["kind"] == "ode":
                fid_score_ode = msg["score"]
                ode_first10 = msg["first10"]
                tag = "ODE"
            else:
                fid_values[msg["global_idx"]] = msg["score"]
                tag = f"SDE [{msg['smin']:.5f},{msg['smax']:.5f}]"
            print(
                f"[main] worker {msg['worker_id']} (cuda:{msg['device_id']}) "
                f"{tag} -> FID={msg['score']:.4f} ({msg['elapsed_sec']:.1f}s)"
            )

    pbar.close()
    for p in processes:
        p.join()

    if error_msgs:
        raise RuntimeError("One or more workers failed:\n" + "\n---\n".join(error_msgs))

    overall_elapsed = time.time() - overall_start
    if elapsed_per_task:
        print(
            f"[main] All workers done in {overall_elapsed:.1f}s "
            f"(per-task mean={np.mean(elapsed_per_task):.1f}s, "
            f"max={np.max(elapsed_per_task):.1f}s)"
        )

    fid_grid = fid_values.reshape(ngrid, ngrid)

    # 7. Save (output bundle matches the sequential script's format).
    output_dir = os.path.join(
        output_dir,
        f"fid_colormap_{n_samples}_samples_seed_{seed}_{model_name}_g={gamma}",
    )
    os.makedirs(output_dir, exist_ok=True)

    if real_samples_first10 is not None:
        np.save(os.path.join(output_dir, "real_samples.npy"), real_samples_first10)
    if ode_first10 is not None:
        np.save(os.path.join(output_dir, "gen_ode_samples.npy"), ode_first10)

    filename = f"fid_grid-g={gamma}.pt"
    save_obj = {
        "fid_grid": fid_grid,
        "fid_ode": fid_score_ode,
        "interval_grid": interval_grid,
        "manual_grid": manual_grid,
        "S_values": S_values,
        "gamma": gamma,
        "initial_time": initial_time,
        "tmin": tmin,
        "ngrid": ngrid,
        "alpha": alpha,
        "nsteps": nsteps,
        "max_scale": max_scale,
        "n_samples": n_samples,
        "seed": seed,
        "network_url": network_url,
        "device_ids": device_ids,
    }
    save_path = os.path.join(output_dir, filename)
    torch.save(save_obj, save_path)
    print(f"[main] Saved FID grid to {save_path}")

    with open(os.path.join(output_dir, "fid_scores.txt"), "w") as f:
        f.write(f"FID Score (ODE): {fid_score_ode}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"initial_time: {initial_time}, tmin: {tmin}, ngrid: {ngrid}, alpha: {alpha}\n")
        f.write(f"network_url: {network_url}\n")
        f.write(f"device_ids: {device_ids}\n")
        f.write(f"wall_clock_sec: {overall_elapsed:.2f}\n")
        f.write("FID grid (ngrid x ngrid, rows=smin, cols=smax, NaN where smin >= smax):\n")
        f.write(f"{fid_grid}\n")

    return fid_grid, fid_score_ode


if __name__ == "__main__":
    main(
        output_dir="/home/ubuntu/repos/DiffSci/notebooks/exploratory/bps/karras_cifar10_edm_stats",
        n_samples=50000,
        nsteps=1000,
        device_ids=(0, 1, 2, 3, 4, 5),
        batch_size=1000,
        gamma=2.0,
        seed=45,
        max_scale=80,
        initial_time=1.0,
        tmin=1e-3,
        ngrid=10,
        alpha=0.1,
        model_name="nvidia-vp-nsteps=1000-manual_grid-parallel",
        interval_grid=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        network_url="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl",
    )

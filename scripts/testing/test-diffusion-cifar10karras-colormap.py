"""
FID colormap (Langevin-interval grid sweep) for CIFAR-10 unconditional diffusion
using the NVIDIA EDM checkpoint via DiffSci.

This script mirrors the structure of
``scripts/testing/test-diffusion-mnist-colormap.py`` (which sweeps a grid of
(s_min, s_max) Langevin intervals and stores the resulting FID values in a
2-D colormap), but performs inference / FID computation in the same way as
``scripts/testing/test-diffusion-cifar10-karras.py``:

- CIFAR-10 (train + val) used as the real distribution
- NVIDIA EDM 32x32 unconditional checkpoint loaded via a pickle URL
- ``NullPreconditioner`` (the NVIDIA network already implements EDM preconditioning)
- Model outputs in [-1, 1] are denormalised to [0, 1] before FID
"""

import os
import sys
import itertools
import pickle
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torchmetrics.image import FrechetInceptionDistance
from tqdm import tqdm

# EDM repo for NVIDIA checkpoint loading (must be before importing diffsci if edm is needed)
_EDM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external_repos", "edm")
if os.path.isdir(_EDM_DIR):
    sys.path.insert(0, os.path.abspath(_EDM_DIR))

import diffsci.models
from diffsci.models.karras.integrators import EulerIntegrator, EulerMaruyamaIntegrator


def custom_spacing(min_val, max_val, N, alpha):
    """Power-law spaced points between ``min_val`` and ``max_val``.

    Mirrors the helper used in
    ``notebooks/exploratory/bps/046-bps-entropy_paper-investigating_scores.ipynb``.
    """
    min_alpha = min_val ** alpha
    max_alpha = max_val ** alpha
    lin_space = torch.linspace(min_alpha, max_alpha, N)
    return lin_space ** (1 / alpha)


def _extract_edm_net_from_pickle(loaded_obj):
    """Extract a torch.nn.Module from common EDM pickle formats."""
    if isinstance(loaded_obj, nn.Module):
        return loaded_obj

    if isinstance(loaded_obj, dict):
        # NVIDIA/EDM snapshots commonly store the model under "ema".
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

    # Optional local fallback: EDM training-state dumps (.pt/.pth) can store "net".
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
    """Map model output [-1, 1] to [0, 1] for FID (then to uint8 in prepare_for_fid)."""
    img = (img_tensor + 1) / 2
    return img.clamp(0, 1)


def prepare_for_fid(images_tensor):
    """Convert [0, 1] float tensor (B, C, H, W) to uint8 for FrechetInceptionDistance."""
    if images_tensor.shape[1] == 1:
        images_tensor = images_tensor.repeat(1, 3, 1, 1)
    return (images_tensor.clamp(0, 1) * 255).to(dtype=torch.uint8)


def main(
    output_dir,
    n_samples=10000,
    device_id=7,
    batch_size=500,
    gamma=1.0,
    seed=42,
    model_name='nvidia-edm-cifar10',
    nsteps=200,
    max_scale=80,
    initial_time=1.0,
    tmin=1e-3,
    ngrid=10,
    alpha=0.1,
    interval_grid=None,
    image_size=32,
    network_url="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
):
    # 0. Setup Device
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} with {n_samples} samples (Batch size: {batch_size})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load NVIDIA EDM and build DiffSci module
    print("Loading NVIDIA EDM CIFAR-10 model...")
    net = load_nvidia_edm_cifar10(network_url, device)
    adapter = NVIDIAWrapper(net)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
    moduleconfig.preconditioner = diffsci.models.karras.preconditioners.NullPreconditioner()
    module = diffsci.models.KarrasModule(adapter, moduleconfig, conditional=False)
    module = module.to(device)
    module.eval()

    print(module.config.noisescheduler.maximum_scale)
    module.config.noisescheduler.maximum_scale = max_scale
    print(module.config.noisescheduler.maximum_scale)

    # 2. Load Real Data (CIFAR-10 train + val)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    real_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    shape = [3, image_size, image_size]
    num_batches = (n_samples + batch_size - 1) // batch_size

    # Initialize Metrics
    fid_ode = FrechetInceptionDistance(normalize=False).to(device)

    # --- STEP 3: Process REAL Images in Batches ---
    print("Processing real images...")
    real_samples_all = []  # Keep light CPU copy for re-feeding FID per interval

    for batch, _ in tqdm(real_loader):
        batch = batch.to(device)
        fid_ode.update(prepare_for_fid(batch), real=True)
        real_samples_all.append(batch.cpu().numpy())

    real_samples_concat = np.concatenate(real_samples_all, axis=0)

    # --- STEP 4: Generate & Process ODE Samples (used as diagonal of FID grid) ---
    print("Generating ODE samples...")
    ode_integrator = EulerIntegrator()
    ode_samples_all = []

    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            curr_batch = min(batch_size, n_samples - (i * batch_size))
            gen_batch = module.sample(
                nsamples=curr_batch,
                shape=shape,
                nsteps=nsteps,
                integrator=ode_integrator,
            )
            gen_batch = denormalize_for_fid(gen_batch)
            fid_ode.update(prepare_for_fid(gen_batch), real=False)
            ode_samples_all.append(gen_batch.cpu().numpy())

    fid_score_ode = fid_ode.compute().item()
    print(f"ODE FID: {fid_score_ode}")

    # --- STEP 5: Build interval grid and compute SDE FID per interval ---
    if interval_grid is None:
        S_values = custom_spacing(tmin, initial_time, ngrid, alpha=alpha)
        S_values_list = [float(v) for v in S_values]
        interval_grid = list(itertools.product(S_values_list, S_values_list))
        manual_grid = False
    elif len(interval_grid) == ngrid:
        # User supplied a 1D discretization of the time interval; expand it to
        # the (S_min, S_max) product version.
        S_values = torch.as_tensor([float(v) for v in interval_grid])
        S_values_list = S_values.tolist()
        interval_grid = list(itertools.product(S_values_list, S_values_list))
        manual_grid = True
    elif len(interval_grid) == ngrid * ngrid:
        # User supplied the full product grid directly.
        S_values = None
        manual_grid = True
        interval_grid = [(float(a), float(b)) for a, b in interval_grid]
    else:
        raise ValueError(
            f"interval_grid has {len(interval_grid)} entries, expected either "
            f"{ngrid} (1D discretization) or {ngrid * ngrid} (full product grid)")

    print(f"Interval grid built with {len(interval_grid)} entries (ngrid={ngrid}, "
          f"manual={manual_grid})")

    sde_integrator = EulerMaruyamaIntegrator()
    module.config.noisescheduler.langevin_const = gamma

    fid_values = torch.full((ngrid * ngrid,), float('nan'), dtype=torch.float64)

    for idx, (smin, smax) in enumerate(interval_grid):
        if smin >= smax:
            # Lower-triangle / diagonal entries are not run; left as NaN.
            # The plotting helper replaces the diagonal NaNs with the ODE FID.
            continue

        print(f"[{idx + 1}/{len(interval_grid)}] interval=[{smin:.5f}, {smax:.5f}]")
        module.config.noisescheduler.langevin_interval = [smin, smax]

        # New FID instance for this interval
        fid_sde = FrechetInceptionDistance(normalize=False).to(device)

        # Re-feed real stats from cached numpy array
        for i in range(0, len(real_samples_concat), batch_size):
            batch_real_np = real_samples_concat[i:i + batch_size]
            batch_real_torch = torch.from_numpy(batch_real_np).to(device)
            fid_sde.update(prepare_for_fid(batch_real_torch), real=True)

        with torch.no_grad():
            for i in tqdm(range(num_batches), leave=False):
                curr_batch = min(batch_size, n_samples - (i * batch_size))
                gen_batch = module.sample(
                    nsamples=curr_batch,
                    shape=shape,
                    nsteps=nsteps,
                    integrator=sde_integrator,
                )
                gen_batch = denormalize_for_fid(gen_batch)
                fid_sde.update(prepare_for_fid(gen_batch), real=False)

        score = fid_sde.compute().item()
        fid_values[idx] = score
        print(f"  -> SDE FID = {score:.4f}")

    fid_grid = fid_values.reshape(ngrid, ngrid)

    # --- STEP 6: Save ---
    output_dir = os.path.join(
        output_dir,
        f"fid_colormap_{n_samples}_samples_seed_{seed}_{model_name}_g={gamma}")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "real_samples.npy"), real_samples_concat[:10])
    np.save(os.path.join(output_dir, "gen_ode_samples.npy"),
            np.concatenate(ode_samples_all, axis=0)[:10])

    filename = (
        f"fid_grid-g={gamma}.pt")
    save_obj = {
        "fid_grid": fid_grid,
        "fid_ode": fid_score_ode,
        "interval_grid": interval_grid,
        "manual_grid": manual_grid,
        "S_values": S_values,  # None if interval_grid was passed in manually
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
    }
    save_path = os.path.join(output_dir, filename)
    torch.save(save_obj, save_path)
    print(f"Saved FID grid to {save_path}")

    with open(os.path.join(output_dir, "fid_scores.txt"), "w") as f:
        f.write(f"FID Score (ODE): {fid_score_ode}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"initial_time: {initial_time}, tmin: {tmin}, ngrid: {ngrid}, alpha: {alpha}\n")
        f.write(f"network_url: {network_url}\n")
        f.write("FID grid (ngrid x ngrid, rows=smin, cols=smax, NaN where smin >= smax):\n")
        f.write(f"{fid_grid}\n")

    return fid_grid, fid_score_ode


if __name__ == "__main__":
    main(
        output_dir="/home/ubuntu/repos/DiffSci/notebooks/exploratory/bps/karras_cifar10_vp_stats",
        n_samples=10000,
        nsteps=1000,
        device_id=6,
        batch_size=500,
        gamma=5.0,
        seed=45,
        max_scale=80,
        initial_time=1.0,
        tmin=1e-3,
        ngrid=10,
        alpha=0.1,
        model_name='nvidia-vp-nsteps=1000-manual_grid',
        interval_grid=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        network_url="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl",
    )

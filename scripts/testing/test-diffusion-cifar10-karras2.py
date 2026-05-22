"""
FID computation for CIFAR-10 unconditional diffusion (NVIDIA EDM via DiffSci).

Structure mirrors scripts/testing/test-diffusion-mnist.py but uses:
- CIFAR-10 validation set and NVIDIA EDM 32x32 unconditional checkpoint
- Batched generation and FID updates to control memory
- ODE and SDE (multiple gamma) FID scores
"""

import os
import sys
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

    # Primary path: .pkl snapshots (NVIDIA public checkpoints and EDM training snapshots).
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
    return (images_tensor.clamp(0, 1) * 255).to(dtype=torch.uint8)


def main(
    n_samples=1000,
    device_id=7,
    batch_size=100,
    gamma_list=(0.3, 1.0, 2.0, 5.0),
    seed=42,
    nsteps=256,
    image_size=32,
    network_url="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
    output_dir=None,
    epoch=None,
):
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} with {n_samples} samples (batch_size={batch_size}, nsteps={nsteps})")

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

    # 2. Load real data (CIFAR-10 all images)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    real_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    shape = [3, image_size, image_size]
    num_batches = (n_samples + batch_size - 1) // batch_size

    # 3. Process real images and update FID
    fid_ode = FrechetInceptionDistance(normalize=False).to(device)
    real_samples_all = []
    print("Processing real images...")
    for batch, _ in tqdm(real_loader):
        batch = batch.to(device)
        fid_ode.update(prepare_for_fid(batch), real=True)
        real_samples_all.append(batch.cpu().numpy())
    real_samples_concat = np.concatenate(real_samples_all, axis=0)

    # 4. Generate ODE samples in batches
    ode_integrator = EulerIntegrator()
    ode_samples_all = []
    print("Generating ODE samples...")
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            curr_batch = min(batch_size, n_samples - i * batch_size)
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
    print(f"ODE FID: {fid_score_ode:.4f}")

    # 5. SDE loop (per gamma)
    sde_integrator = EulerMaruyamaIntegrator()
    sde_results = []
    for gamma in gamma_list:
        print(f"Processing SDE (gamma={gamma})...")
        module.config.noisescheduler.langevin_const = gamma
        fid_sde = FrechetInceptionDistance(normalize=False).to(device)
        for i in range(0, len(real_samples_concat), batch_size):
            batch_real_np = real_samples_concat[i:i + batch_size]
            batch_real_torch = torch.from_numpy(batch_real_np).to(device)
            fid_sde.update(prepare_for_fid(batch_real_torch), real=True)
        sde_samples_this_gamma = []
        with torch.no_grad():
            for i in tqdm(range(num_batches), leave=False):
                curr_batch = min(batch_size, n_samples - i * batch_size)
                gen_batch = module.sample(
                    nsamples=curr_batch,
                    shape=shape,
                    nsteps=nsteps,
                    integrator=sde_integrator,
                )
                gen_batch = denormalize_for_fid(gen_batch)
                fid_sde.update(prepare_for_fid(gen_batch), real=False)
                sde_samples_this_gamma.append(gen_batch.cpu().numpy())
        score = fid_sde.compute().item()
        sde_results.append((gamma, score, np.concatenate(sde_samples_this_gamma, axis=0)))
        print(f"SDE (gamma={gamma}) FID: {score:.4f}")

    # 6. Save 10 samples from each configuration
    if output_dir is None:
        output_dir = f"/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260323-bps-karras-cifar10/stats/fid_results_{n_samples}_samples_seed_{seed}_nsteps_{nsteps}_epoch_{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "real_samples.npy"), real_samples_concat[:10])
    np.save(os.path.join(output_dir, "gen_ode_samples.npy"), np.concatenate(ode_samples_all, axis=0)[:10])
    with open(os.path.join(output_dir, "fid_scores.txt"), "w") as f:
        f.write(f"FID Score (ODE): {fid_score_ode:.4f}\n")
        for gamma, score, samples in sde_results:
            np.save(os.path.join(output_dir, f"gen_sde_samples_gamma_{gamma}.npy"), samples[:10])
            f.write(f"FID Score (SDE, γ={gamma}): {score:.4f}\n")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main(
        n_samples=10000,
        device_id=6,
        batch_size=500,
        gamma_list=[0.05, 0.2, 0.5, 1.0, 2.0, 5.0],
        seed=42,
        nsteps=256,
        network_url="/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260323-bps-karras-cifar10/checkpoints/network-snapshot-005000.pkl",
        epoch=5
    )

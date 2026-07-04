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
from diffsci.models.karras.integrators import EulerIntegrator, KarrasIntegrator


def load_nvidia_edm_cifar10(network_url_or_path, device):
    """Load NVIDIA EDM CIFAR-10 unconditional net and wrap for DiffSci."""
    import dnnlib
    with dnnlib.util.open_url(network_url_or_path) as f:
        net = pickle.load(f)["ema"]
    return net.to(device)


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
    S_schurn_list=(10, 20, 40, 70, 100),
    seed=42,
    nsteps=256,
    image_size=32,
    network_url="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
    output_dir=None,
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

    # 5. SDE loop (per S_schurn)
    sde_results = []
    for S_schurn in S_schurn_list:
        print(f"Processing SDE (S_schurn={S_schurn})...")
        sde_integrator = KarrasIntegrator(s_schurn=S_schurn)
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
        sde_results.append((S_schurn, score, np.concatenate(sde_samples_this_gamma, axis=0)))
        print(f"SDE (S_schurn={S_schurn}) FID: {score:.4f}")

    # 6. Save 10 samples from each configuration
    if output_dir is None:
        output_dir = f"/home/ubuntu/repos/DiffSci/notebooks/exploratory/bps/cifar10_fid_results-seed={seed}-karrassampler-nsteps={nsteps}-nsamples={n_samples}"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "real_samples.npy"), real_samples_concat[:10])
    np.save(os.path.join(output_dir, "gen_ode_samples.npy"), np.concatenate(ode_samples_all, axis=0)[:10])
    with open(os.path.join(output_dir, "fid_scores.txt"), "w") as f:
        f.write(f"FID Score (ODE): {fid_score_ode:.4f}\n")
        for S_schurn, score, samples in sde_results:
            np.save(os.path.join(output_dir, f"gen_sde_samples_S_schurn_{S_schurn}.npy"), samples[:10])
            f.write(f"FID Score (SDE, S_schurn={S_schurn}): {score:.4f}\n")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main(
        n_samples=50000,
        device_id=6,
        batch_size=500,
        S_schurn_list=[10, 20, 40, 70, 100],
        seed=42,
        nsteps=1000,
        # network_url='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl'
    )

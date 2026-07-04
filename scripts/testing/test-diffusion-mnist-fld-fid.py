import argparse
import sys
from pathlib import Path

import diffsci.models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance
from diffsci.models.karras.integrators import EulerIntegrator, EulerMaruyamaIntegrator

REPO_ROOT = Path(__file__).resolve().parents[2]
FLD_REPO = REPO_ROOT / "external_repos" / "fld"
# DEFAULT_CHECKPOINTS = [
#     REPO_ROOT
#     / "savedmodels/experimental/20260428-bps-mnist-128ch-20samples/checkpoints"
#     / "model-epoch=199-val_loss=0.129291.ckpt",
#     REPO_ROOT
#     / "savedmodels/experimental/20260428-bps-mnist-128ch-20samples/checkpoints"
#     / "model-epoch=399-val_loss=0.159884.ckpt",
#     REPO_ROOT
#     / "savedmodels/experimental/20260428-bps-mnist-128ch-20samples/checkpoints"
#     / "model-epoch=499-val_loss=0.185757.ckpt",
# ]
DEFAULT_CHECKPOINTS = [
    REPO_ROOT
    / "/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260611-bps-mnist-128ch-120samples/checkpoints/model-epoch=099-val_loss=0.076399.ckpt",
    REPO_ROOT
    / "/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260611-bps-mnist-128ch-120samples/checkpoints/model-epoch=199-val_loss=0.092190.ckpt",
    REPO_ROOT
    / "/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260611-bps-mnist-128ch-120samples/checkpoints/model-epoch=499-val_loss=0.155855.ckpt",
]
GAMMA_LIST = [0.01, 0.1, 0.3, 0.5, 1, 2, 3, 5, 8]
IMAGE_SIZE = 28


class MNISTFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc_features = nn.Linear(256, feature_dim)
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc_features(x))
        if return_features:
            return features
        return self.classifier(features)


def extract_epoch_from_checkpoint(path):
    token = "epoch="
    path = str(path)
    if token in path:
        return int(path.split(token, 1)[1].split("-", 1)[0])
    return -1


def get_mnist_features(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            features = model(images, return_features=True)
            all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


def get_generated_features(feature_model, generated_images_np, device, batch_size=256):
    generated_tensor = torch.from_numpy(generated_images_np).float()
    feature_model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, generated_tensor.shape[0], batch_size):
            batch = generated_tensor[i : i + batch_size].to(device)
            features = feature_model(batch, return_features=True)
            all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


def prepare_for_fid(images):
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return (images.clamp(0, 1) * 255).to(dtype=torch.uint8)


def train_feature_extractor(device, data_root, seed, classifier_epochs=2, batch_size=128):
    torch.manual_seed(seed)
    np.random.seed(seed)

    classifier_model = MNISTFeatureExtractor(feature_dim=128).to(device)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    print(f"Training MNIST feature extractor for {classifier_epochs} epochs...")
    optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    classifier_model.train()
    for epoch in range(classifier_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch + 1} Complete | Loss: {total_loss / len(train_loader):.4f}"
        )
    print("Extractor ready!")
    return classifier_model


def load_diffusion_modules(checkpoint_paths, model_channels, device):
    modelconfig = diffsci.models.PUNetGConfig(model_channels=model_channels)
    model = diffsci.models.PUNetG(modelconfig)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()

    modules_for_eval = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        module = diffsci.models.KarrasModule.load_from_checkpoint(
            str(checkpoint_path),
            model=model,
            config=moduleconfig,
            conditional=False,
            map_location=device,
        )
        module = module.to(device)
        module.eval()
        modules_for_eval.append(
            {
                "epoch": extract_epoch_from_checkpoint(checkpoint_path),
                "path": str(checkpoint_path),
                "module": module,
            }
        )
    modules_for_eval.sort(key=lambda d: d["epoch"])
    return modules_for_eval


def generate_samples(modules_for_eval, n_samples, nsteps, device):
    ode_integrator = EulerIntegrator()
    sde_integrator = EulerMaruyamaIntegrator()
    all_generated_curves = []

    for spec in modules_for_eval:
        current_module = spec["module"]
        epoch = spec["epoch"]
        print(f"\nGenerating samples for epoch {epoch}...")

        with torch.no_grad():
            generated_samples_ode = current_module.sample(
                nsamples=n_samples,
                shape=[1, IMAGE_SIZE, IMAGE_SIZE],
                nsteps=nsteps,
                integrator=ode_integrator,
            )
        generated_samples_ode = generated_samples_ode.detach().cpu().numpy()
        print(f"  ODE shape: {generated_samples_ode.shape}")

        sde_list = []
        for gamma in GAMMA_LIST:
            current_module.config.noisescheduler.langevin_const = gamma
            with torch.no_grad():
                generated_samples_sde = current_module.sample(
                    nsamples=n_samples,
                    shape=[1, IMAGE_SIZE, IMAGE_SIZE],
                    nsteps=nsteps,
                    integrator=sde_integrator,
                )
            generated_samples_sde = generated_samples_sde.detach().cpu().numpy()
            sde_list.append((gamma, generated_samples_sde))
            print(f"  SDE gamma={gamma}: shape={generated_samples_sde.shape}")

        all_generated_curves.append(
            {
                "epoch": epoch,
                "path": spec["path"],
                "ode_samples": generated_samples_ode,
                "sde_list": sde_list,
            }
        )
    return all_generated_curves


def compute_fld_curves(
    classifier_model, all_generated_curves, device, data_root, fld_metric
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    print("Extracting train and test features...")
    train_features = get_mnist_features(classifier_model, train_set, device)
    test_features = get_mnist_features(classifier_model, test_set, device)

    std_test = test_features.std(dim=0)
    valid_dims = std_test > 1e-8
    num_dropped = (~valid_dims).sum().item()
    print(
        f"Feature dims kept: {valid_dims.sum().item()} / {valid_dims.numel()} "
        f"(dropped {num_dropped})"
    )
    if valid_dims.sum().item() == 0:
        raise RuntimeError("All feature dimensions have near-zero std; FLD cannot be computed.")

    train_features = train_features[:, valid_dims]
    test_features = test_features[:, valid_dims]
    print(
        f"NaNs in train/test features: "
        f"{torch.isnan(train_features).any().item()} / {torch.isnan(test_features).any().item()}"
    )

    all_fld_curves = []
    for curve in all_generated_curves:
        epoch = curve["epoch"]
        print(f"\nComputing FLD for epoch {epoch}...")

        ode_features = get_generated_features(
            classifier_model, curve["ode_samples"], device
        )
        ode_features = ode_features[:, valid_dims]
        if torch.isnan(ode_features).any() or torch.isinf(ode_features).any():
            ode_fld = float("nan")
        else:
            ode_fld = float(
                fld_metric.compute_metric(train_features, test_features, ode_features)
            )

        gammas = []
        sde_flds = []
        for gamma, generated_samples_sde in curve["sde_list"]:
            gen_features = get_generated_features(
                classifier_model, generated_samples_sde, device
            )
            gen_features = gen_features[:, valid_dims]
            if torch.isnan(gen_features).any() or torch.isinf(gen_features).any():
                fld_value = float("nan")
                print(f"  gamma={gamma}: NaN/Inf in generated features")
            else:
                fld_value = float(
                    fld_metric.compute_metric(
                        train_features, test_features, gen_features
                    )
                )
                print(f"  gamma={gamma}: FLD={fld_value:.3f}")
            gammas.append(float(gamma))
            sde_flds.append(fld_value)

        all_fld_curves.append(
            {
                "epoch": epoch,
                "path": curve["path"],
                "ode_fld": ode_fld,
                "gammas": np.array(gammas, dtype=float),
                "sde_flds": np.array(sde_flds, dtype=float),
            }
        )

    all_fld_curves.sort(key=lambda d: d["epoch"])
    return all_fld_curves


def compute_fid_curves(all_generated_curves, n_samples, data_root, device):
    fid_transform = transforms.Compose([transforms.ToTensor()])
    fid_test_set = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=fid_transform
    )
    fid_loader = DataLoader(fid_test_set, batch_size=256, shuffle=False)

    real_tensors = []
    for images, _ in fid_loader:
        real_tensors.append(images)
    real_tensors = torch.cat(real_tensors, dim=0)[:n_samples]
    real_samples_fid = prepare_for_fid(real_tensors).to(device)
    print(f"Real FID reference shape: {tuple(real_samples_fid.shape)}")

    all_fid_curves = []
    for curve in all_generated_curves:
        epoch = curve["epoch"]
        print(f"\nComputing FID for epoch {epoch}...")

        gen_ode_fid = prepare_for_fid(curve["ode_samples"]).to(device)
        fid_ode = FrechetInceptionDistance(normalize=False).to(device)
        fid_ode.update(real_samples_fid, real=True)
        fid_ode.update(gen_ode_fid, real=False)
        ode_fid = float(fid_ode.compute().item())
        print(f"  ODE FID={ode_fid:.3f}")

        gammas = []
        sde_fids = []
        for gamma, generated_samples_sde in curve["sde_list"]:
            gen_sde_fid = prepare_for_fid(generated_samples_sde).to(device)
            fid_sde = FrechetInceptionDistance(normalize=False).to(device)
            fid_sde.update(real_samples_fid, real=True)
            fid_sde.update(gen_sde_fid, real=False)
            fid_value = float(fid_sde.compute().item())
            print(f"  gamma={gamma}: FID={fid_value:.3f}")
            gammas.append(float(gamma))
            sde_fids.append(fid_value)

        all_fid_curves.append(
            {
                "epoch": epoch,
                "path": curve["path"],
                "ode_fid": ode_fid,
                "gammas": np.array(gammas, dtype=float),
                "sde_fids": np.array(sde_fids, dtype=float),
            }
        )

    all_fid_curves.sort(key=lambda d: d["epoch"])
    return all_fid_curves


def write_fld_scores(path, all_fld_curves):
    with open(path, "w") as f:
        for curve in all_fld_curves:
            f.write(f"Epoch {curve['epoch']}\n")
            f.write(f"Checkpoint: {curve['path']}\n")
            f.write(f"FLD Score (ODE): {curve['ode_fld']:.6f}\n")
            for gamma, fld_value in zip(curve["gammas"], curve["sde_flds"]):
                f.write(f"FLD Score (SDE, gamma={gamma}): {fld_value:.6f}\n")
            f.write("\n")


def write_fid_scores(path, all_fid_curves):
    with open(path, "w") as f:
        for curve in all_fid_curves:
            f.write(f"Epoch {curve['epoch']}\n")
            f.write(f"Checkpoint: {curve['path']}\n")
            f.write(f"FID Score (ODE): {curve['ode_fid']:.6f}\n")
            for gamma, fid_value in zip(curve["gammas"], curve["sde_fids"]):
                f.write(f"FID Score (SDE, gamma={gamma}): {fid_value:.6f}\n")
            f.write("\n")


def main(
    n_samples,
    checkpoint_paths=None,
    model_channels=128,
    nsteps=500,
    seed=42,
    device_id=0,
    data_root=None,
    output_dir=None,
    classifier_epochs=5,
):
    if checkpoint_paths is None:
        checkpoint_paths = DEFAULT_CHECKPOINTS
    checkpoint_paths = [Path(p) for p in checkpoint_paths]

    if data_root is None:
        data_root = str(REPO_ROOT / "data")
    if output_dir is None:
        output_dir = (
            Path(__file__).parent
            / f"mnist_fld_fid_results_{n_samples}_samples_seed_{seed}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(f"cuda:{device_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Running on {device} with {n_samples} samples")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if str(FLD_REPO) not in sys.path:
        sys.path.append(str(FLD_REPO))
    from fld.metrics.FLD import FLD

    classifier_model = train_feature_extractor(
        device=device,
        data_root=data_root,
        seed=seed,
        classifier_epochs=classifier_epochs,
    )
    modules_for_eval = load_diffusion_modules(
        checkpoint_paths=checkpoint_paths,
        model_channels=model_channels,
        device=device,
    )
    all_generated_curves = generate_samples(
        modules_for_eval=modules_for_eval,
        n_samples=n_samples,
        nsteps=nsteps,
        device=device,
    )

    fld_metric = FLD()
    all_fld_curves = compute_fld_curves(
        classifier_model=classifier_model,
        all_generated_curves=all_generated_curves,
        device=device,
        data_root=data_root,
        fld_metric=fld_metric,
    )
    all_fid_curves = compute_fid_curves(
        all_generated_curves=all_generated_curves,
        n_samples=n_samples,
        data_root=data_root,
        device=device,
    )

    fld_path = output_dir / "fld_scores.txt"
    fid_path = output_dir / "fid_scores.txt"
    write_fld_scores(fld_path, all_fld_curves)
    write_fid_scores(fid_path, all_fid_curves)

    print(f"\nSaved FLD scores to {fld_path}")
    print(f"Saved FID scores to {fid_path}")
    return all_fld_curves, all_fid_curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MNIST diffusion FLD and FID curves across epochs and gamma values."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of generated and real reference samples per configuration.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        default=None,
        help="Diffusion checkpoint paths (default: notebook epochs 199/399/499).",
    )
    parser.add_argument("--model-channels", type=int, default=128)
    parser.add_argument("--nsteps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--data-root",
        default=None,
        help="MNIST data directory (default: <repo>/data).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: scripts/testing/mnist_fld_fid_results_<n>_samples_seed_<seed>).",
    )
    parser.add_argument("--classifier-epochs", type=int, default=5)
    args = parser.parse_args()

    main(
        n_samples=args.n_samples,
        checkpoint_paths=args.checkpoint_paths,
        model_channels=args.model_channels,
        nsteps=args.nsteps,
        seed=args.seed,
        device_id=args.device_id,
        data_root=args.data_root,
        output_dir=args.output_dir,
        classifier_epochs=args.classifier_epochs,
    )

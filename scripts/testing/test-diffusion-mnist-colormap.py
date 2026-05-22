import os
import itertools
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchmetrics.image import FrechetInceptionDistance
from diffsci.models.karras.integrators import EulerIntegrator, EulerMaruyamaIntegrator
import diffsci.models
from tqdm import tqdm


def custom_spacing(min_val, max_val, N, alpha):
    """Power-law spaced points between ``min_val`` and ``max_val``.

    Mirrors the helper used in
    ``notebooks/exploratory/bps/046-bps-entropy_paper-investigating_scores.ipynb``.
    """
    min_alpha = min_val ** alpha
    max_alpha = max_val ** alpha
    lin_space = torch.linspace(min_alpha, max_alpha, N)
    return lin_space ** (1 / alpha)


def main(checkpoint_path,
         model_channels,
         n_samples=10000,
         device_id=7,
         batch_size=1000,
         gamma=1.0,
         seed=42,
         only_validation=False,
         model_name='model',
         nsteps=200,
         max_scale=80,
         initial_time=1.0,
         tmin=1e-3,
         ngrid=10,
         alpha=0.1,
         interval_grid=None):
    # 0. Setup Device
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} with {n_samples} samples (Batch size: {batch_size})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Real Data (Validation or All)
    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    if not only_validation:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = torch.utils.data.ConcatDataset([val_dataset, train_dataset])

    real_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Setup Model
    modelconfig = diffsci.models.PUNetGConfig(model_channels=model_channels)
    model = diffsci.models.PUNetG(modelconfig)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()

    module = diffsci.models.KarrasModule.load_from_checkpoint(checkpoint_path, model=model, config=moduleconfig, conditional=False)
    module = module.to(device)
    module.eval()

    print(module.config.noisescheduler.maximum_scale)
    module.config.noisescheduler.maximum_scale = max_scale
    print(module.config.noisescheduler.maximum_scale)

    # Helper: Prepare for FID
    def prepare_for_fid(images_tensor):
        if images_tensor.shape[1] == 1:
            images_tensor = images_tensor.repeat(1, 3, 1, 1)
        return (images_tensor.clamp(0, 1) * 255).to(dtype=torch.uint8)

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

    num_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            curr_batch = min(batch_size, n_samples - (i * batch_size))
            gen_batch = module.sample(
                nsamples=curr_batch,
                shape=[1, 28, 28],
                nsteps=nsteps,
                integrator=ode_integrator,
            )
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
                    shape=[1, 28, 28],
                    nsteps=nsteps,
                    integrator=sde_integrator,
                )
                fid_sde.update(prepare_for_fid(gen_batch), real=False)

        score = fid_sde.compute().item()
        fid_values[idx] = score
        print(f"  -> SDE FID = {score:.4f}")

    fid_grid = fid_values.reshape(ngrid, ngrid)

    # --- STEP 6: Save ---
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_dir)
    output_dir = os.path.join(
        checkpoint_dir,
        f"stats/fid_colormap_{n_samples}_samples_seed_{seed}_{model_name}_g={gamma}")
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
    }
    save_path = os.path.join(output_dir, filename)
    torch.save(save_obj, save_path)
    print(f"Saved FID grid to {save_path}")

    with open(os.path.join(output_dir, "fid_scores.txt"), "w") as f:
        f.write(f"FID Score (ODE): {fid_score_ode}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"initial_time: {initial_time}, tmin: {tmin}, ngrid: {ngrid}, alpha: {alpha}\n")
        f.write("FID grid (ngrid x ngrid, rows=smin, cols=smax, NaN where smin >= smax):\n")
        f.write(f"{fid_grid}\n")

    return fid_grid, fid_score_ode


if __name__ == "__main__":
    main(checkpoint_path="/home/ubuntu/repos/DiffSci/savedmodels/production/20260318-bps-mnist-128ch/checkpoints/model-epoch=039-val_loss=0.042329.ckpt",
         model_channels=128,
         n_samples=10000,
         nsteps=500,
         device_id=6,
         batch_size=500,
         gamma=5.0,
         seed=45,
         max_scale=80,
         initial_time=1.0,
         tmin=1e-3,
         ngrid=10,
         alpha=0.1,
         model_name='nsteps=500-epoch=39-manual_grid',
         interval_grid=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])

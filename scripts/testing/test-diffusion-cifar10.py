import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchmetrics.image import FrechetInceptionDistance
from diffsci.models.karras.integrators import EulerIntegrator, EulerMaruyamaIntegrator
import diffsci.models
from tqdm import tqdm  # Recommended for progress bars


def main(checkpoint_path,
         model_channels,
         n_samples=10000,
         device_id=7,
         batch_size=1000,
         gamma_list=[0.2, 0.5, 1.0, 1.5, 2.0, 3.0],
         nsteps=256,
         seed=42,
         output_dir="output",
         only_validation=False,
         preconditioner="edm"):
    # 0. Setup Device
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} with {n_samples} samples (Batch size: {batch_size})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Real Data (Validation or All)
    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if not only_validation:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = torch.utils.data.ConcatDataset([val_dataset, train_dataset])

    real_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Setup Model
    modelconfig = diffsci.models.PUNetGConfig(model_channels=model_channels, input_channels=3, output_channels=3)
    model = diffsci.models.PUNetG(modelconfig)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
    assert preconditioner in ["edm", "vp", "ve"]
    if preconditioner == "edm":
        moduleconfig.preconditioner = diffsci.models.karras.preconditioners.EDMPreconditioner(sigma_data=0.5)
    elif preconditioner == "vp":
        moduleconfig.preconditioner = diffsci.models.karras.preconditioners.VPPreconditioner(scheduler=moduleconfig.noisescheduler, M=1000)
    elif preconditioner == "ve":
        moduleconfig.preconditioner = diffsci.models.karras.preconditioners.VEPreconditioner()
    else:
        raise ValueError(f"Invalid preconditioner: {preconditioner}")

    module = diffsci.models.KarrasModule.load_from_checkpoint(
        checkpoint_path, model=model, config=moduleconfig, conditional=False
    )
    module = module.to(device)
    module.eval()

    # Helper: Prepare for FID
    def prepare_for_fid(images_tensor):
        if images_tensor.shape[1] == 1:
            images_tensor = images_tensor.repeat(1, 3, 1, 1)
        return (images_tensor.clamp(0, 1) * 255).to(dtype=torch.uint8)

    # Initialize Metrics
    fid_ode = FrechetInceptionDistance(normalize=False).to(device)

    # --- STEP 3: Process REAL Images in Batches ---
    print("Processing real images...")
    real_samples_all = []  # Keep light CPU copy for saving later

    for batch, _ in tqdm(real_loader):
        batch = batch.to(device)
        # Update FID stats incrementally
        fid_ode.update(prepare_for_fid(batch), real=True)
        # Save to list (move to CPU to save GPU RAM)
        real_samples_all.append(batch.cpu().numpy())

    real_samples_concat = np.concatenate(real_samples_all, axis=0)

    # --- STEP 4: Generate & Process ODE Samples in Batches ---
    print("Generating ODE samples...")
    ode_integrator = EulerIntegrator()
    ode_samples_all = []

    # Calculate how many batches needed
    num_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            # Calculate current batch size (handle last batch)
            curr_batch = min(batch_size, n_samples - (i * batch_size))

            gen_batch = module.sample(
                nsamples=curr_batch,
                shape=[3, 32, 32],
                nsteps=nsteps,
                integrator=ode_integrator
            )

            # Update FID immediately
            fid_ode.update(prepare_for_fid(gen_batch), real=False)

            # Save for disk (CPU)
            ode_samples_all.append(gen_batch.cpu().numpy())

    fid_score_ode = fid_ode.compute().item()
    print(f"ODE FID: {fid_score_ode}")

    # --- STEP 5: SDE Loop (Gamma) ---
    sde_integrator = EulerMaruyamaIntegrator()
    sde_results = []

    for gamma in gamma_list:
        print(f"Processing SDE (gamma={gamma})...")
        module.config.noisescheduler.langevin_const = gamma

        # New FID instance for this run
        fid_sde = FrechetInceptionDistance(normalize=False).to(device)

        # Re-add real stats (unfortunately we must re-feed or deepcopy the metric state)
        # To avoid reloading data, we can just loop over our saved numpy array
        # converting back to tensor batches.
        for i in range(0, len(real_samples_concat), batch_size):
            batch_real_np = real_samples_concat[i:i + batch_size]
            batch_real_torch = torch.from_numpy(batch_real_np).to(device)
            fid_sde.update(prepare_for_fid(batch_real_torch), real=True)

        sde_samples_this_gamma = []

        with torch.no_grad():
            for i in tqdm(range(num_batches), leave=False):
                curr_batch = min(batch_size, n_samples - (i * batch_size))

                gen_batch = module.sample(
                    nsamples=curr_batch,
                    shape=[3, 32, 32],
                    nsteps=nsteps,
                    integrator=sde_integrator
                )

                fid_sde.update(prepare_for_fid(gen_batch), real=False)
                sde_samples_this_gamma.append(gen_batch.cpu().numpy())

        score = fid_sde.compute().item()
        sde_results.append((gamma, score, np.concatenate(sde_samples_this_gamma, axis=0)))
        print(f"SDE (gamma={gamma}) FID: {score}")

    # --- STEP 6: Save ---
    # Create output directory based on checkpoint name and parameters
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_dir)
    output_dir = os.path.join(checkpoint_dir, f"stats/fid_results_{n_samples}_samples_seed_{seed}_nsteps={nsteps}")
    os.makedirs(output_dir, exist_ok=True)

    # Save 10 samples (as a single tensor) from each configuration
    np.save(os.path.join(output_dir, "real_samples.npy"), real_samples_concat[:10])
    np.save(os.path.join(output_dir, "gen_ode_samples.npy"), np.concatenate(ode_samples_all, axis=0)[:10])

    with open(os.path.join(output_dir, "fid_scores.txt"), "w") as f:
        f.write(f"FID Score (ODE): {fid_score_ode}\n")
        for gamma, score, samples in sde_results:
            np.save(os.path.join(output_dir, f"gen_sde_samples_gamma_{gamma}.npy"), samples[:10])
            f.write(f"FID Score (SDE, gamma={gamma}): {score}\n")


if __name__ == "__main__":
    main(
        checkpoint_path="/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260304-bps-cifar10-128ch-vp2/checkpoints/model-epoch=085-val_loss=0.008652.ckpt",
        model_channels=128,
        n_samples=50000,
        device_id=7,
        batch_size=500,
        gamma_list=[0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
        seed=45,
        nsteps=256,
        preconditioner="vp"
    )

"""
test-time-profile-correlation.py

For every checkpoint in a checkpoint folder, computes:
  1. all_entropies: KL divergences (both H(~p||p) and H(p||~p)) of SDE-generated
     samples vs. the reference data distribution, across a range of gamma values.
  2. all_errors: score L² error and DSM loss along a forward SDE trajectory.

Results are saved to an output/ directory alongside this script.

Default checkpoint folder:
  /home/ubuntu/repos/DiffSci/savedmodels/experimental/
  mixt_gauss2_silu-width=128-bs=16-precond=edm-diffsci-train=4000-nlm=default2/checkpoints

Model / dataset / diffusion parameters match notebook:
  054-bps-entropy_paper-investigating_scores-80.ipynb
"""

import argparse
import math
import os
import pathlib
import re

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy

import diffsci.data
import diffsci.models
from diffsci.models import KarrasModule, KarrasModuleConfig, EDMPreconditioner


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture  (identical to notebook cells #68-#72)
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Improved1DMLP(nn.Module):
    def __init__(self, data_dim=1, time_embed_dim=64, hidden_dim=128,
                 num_layers=3, residual=True):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()
        self.residual = residual

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        t_emb = self.time_mlp(self.time_embed(t))
        h = self.input_proj(x)
        for layer in self.layers:
            h = h + t_emb
            h = h + layer(self.act(h)) if self.residual else layer(self.act(h))
        return self.output_proj(h)


class CustomEDMPreconditioner(EDMPreconditioner):
    """EDMPreconditioner with an optional 'scale' noise conditioning mode."""

    def __init__(self, sigma_data: float = 0.5, mode: str = "scale"):
        super().__init__()
        self.register_buffer("sigma_data", torch.tensor(sigma_data))
        self.mode = mode

    def noise_conditioner(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.mode == "default":
            return 0.5 * torch.log(sigma)
        elif self.mode == "scale":
            return 0.25 * torch.log(sigma)
        elif self.mode == "floor":
            return 0.5 * torch.log(sigma + 1e-2)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions  (ported from notebook helper cells)
# ─────────────────────────────────────────────────────────────────────────────

def custom_spacing(min_val: float, max_val: float, N: int, alpha: float = 0.5) -> torch.Tensor:
    """Non-uniform spacing biased toward smaller values (alpha < 1)."""
    min_a = min_val ** alpha
    max_a = max_val ** alpha
    lin = torch.linspace(min_a, max_a, N)
    return lin ** (1.0 / alpha)


def approx_entropy1(samples: torch.Tensor, samples_ref: torch.Tensor,
                    epsilon: float = 1e-12, nbins: int = 100) -> float:
    """KL divergence KL(samples || samples_ref) via histogram approximation.

    To reduce sensitivity to the choice of nbins, the estimate is averaged
    over a range of bin counts (nbins-20, ..., nbins), matching the
    approx_entropy helper used in the mixtures notebooks.
    """
    s = samples.detach().cpu().numpy().ravel()
    r = samples_ref.detach().cpu().numpy().ravel()
    lo = float(min(s.min(), r.min()))
    hi = float(max(s.max(), r.max()))
    kls = []
    for nb in range(nbins - 20, nbins):
        if nb % 4 == 0:
            continue
        bins = np.linspace(lo, hi, nb)
        p = np.histogram(s, bins=bins, density=True)[0] + epsilon
        q = np.histogram(r, bins=bins, density=True)[0] + epsilon
        kls.append(entropy(p, q))
    return float(np.mean(kls))


def generate_multiple_gamma_nosave(
    initial_step: int,
    gamma_range: list,
    nsteps: int,
    nsamples: int,
    dataset,
    ngamma: int = 50,
    interval=None,
    noisescheduler=None,
    module=None,
    use_exact_prior: bool = False,
    alpha: float = 0.5,
):
    """
    Like generate_multiple_gamma from the notebook but without writing files.

    Returns:
        ode_final  : Tensor[nsamples, 1]  – ODE samples at t=0
        sde_finals : list of Tensor[nsamples, 1] – SDE samples for each gamma
        gamma_values : Tensor[ngamma]
    """
    if noisescheduler is None:
        noisescheduler = diffsci.models.EDMScheduler()

    noisescheduler._integrator = diffsci.models.EulerIntegrator()
    noisescheduler.langevin_interval = interval
    if module is not None:
        module.config.noisescheduler = noisescheduler

    time = noisescheduler.create_steps(nsteps + 1)
    initial_time = time[initial_step].item()
    print(f"    initial_time = {initial_time:.4f}")

    dataset.num_samples = nsamples
    samples = dataset.sample()
    sigma = noisescheduler.scheduler_fns.noise_fn(initial_time)
    prior = torch.randn_like(samples) * sigma
    exact_prior_samples = noisescheduler.apply_noise(samples, nsteps, initial_step)
    if use_exact_prior:
        prior = exact_prior_samples

    assert len(gamma_range) == 2
    gamma_values = custom_spacing(gamma_range[0], gamma_range[-1], ngamma, alpha)
    print(f"    gamma range: [{gamma_values[0].item():.4f}, {gamma_values[-1].item():.4f}]")

    if module is not None:
        prior = prior.to(module.device)
        with torch.no_grad():
            ode_history = module.propagate_partial_toward_sample(
                prior,
                initial_step=initial_step,
                nsteps=nsteps,
                record_history=True,
                integrator=diffsci.models.EulerIntegrator(),
            ).cpu()

        sde_histories = []
        for g in gamma_values:
            noisescheduler.langevin_const = g
            with torch.no_grad():
                sde_h = module.propagate_partial_toward_sample(
                    prior,
                    initial_step=initial_step,
                    nsteps=nsteps,
                    record_history=True,
                    integrator=diffsci.models.EulerMaruyamaIntegrator(),
                ).cpu()
            sde_histories.append(sde_h)
    else:
        ode_history = noisescheduler.propagate_partial(
            prior, dataset.gradlogprob, nsteps,
            initial_step=initial_step, final_step=nsteps,
            record_history=True, stochastic=False,
        )
        sde_histories = []
        for g in gamma_values:
            noisescheduler.langevin_const = g
            sde_h = noisescheduler.propagate_partial(
                prior, dataset.gradlogprob, nsteps,
                initial_step=initial_step, final_step=nsteps,
                record_history=True, stochastic=True,
            )
            sde_histories.append(sde_h)

    ode_final = ode_history[-1]                          # [nsamples, 1]
    sde_finals = [h[-1] for h in sde_histories]          # list of [nsamples, 1]
    return ode_final, sde_finals, gamma_values


def compute_entropies_for_checkpoint(
    sde_samples: list,
    diffused_data: torch.Tensor,
    gamma_values: torch.Tensor,
    nbins: int = 80,
):
    """
    Compute H(~p || p) and H(p || ~p) for each SDE sample set.

    Returns:
        gamma_values      : Tensor[ngamma]
        sde_entropies     : list[float]   KL(~p || p)
        inv_sde_entropies : list[float]   KL(p || ~p)
    """
    sde_entropies = []
    inv_sde_entropies = []
    for sde_sample in sde_samples:
        sde_entropies.append(approx_entropy1(sde_sample, diffused_data, nbins=nbins))
        inv_sde_entropies.append(approx_entropy1(diffused_data, sde_sample, nbins=nbins))
    return gamma_values, sde_entropies, inv_sde_entropies


def error_norm_single_step(
    module,
    step: int,
    sde_history: torch.Tensor,
    initial_step: int,
    time: torch.Tensor,
    scheduler,
    dataset,
    datasize: int = 2000,
):
    """L² score error and DSM loss at a single diffusion time step."""
    x = sde_history[step - initial_step, :datasize].cpu().unsqueeze(-1)
    t = time[step]
    sigma_scalar = scheduler.scheduler_fns.noise_fn(t)
    sigma = sigma_scalar * torch.ones(x.shape[0])
    s = scheduler.scheduler_fns.scaling_fn(t)

    x_ = x.to(module.device)
    sigma_ = sigma.to(x_)

    with torch.no_grad():
        score = module.get_score(x_ / s, sigma_)
        analytic_score = dataset.gradlogprob(x / s, sigma).to(score)
        score_error = torch.mean((score - analytic_score) ** 2).cpu().item()

        x_0 = sde_history[-1, :datasize].cpu()
        denoiser, _ = module.get_denoiser(x_ / s, sigma_)
        denoiser = denoiser.detach().cpu()
        dsm_loss = (torch.mean((denoiser - x_0) ** 2) / sigma_scalar ** 4).item()

    return score_error, dsm_loss


def compute_error_norm(
    module,
    initial_step: int,
    sde_history: torch.Tensor,
    time: torch.Tensor,
    scheduler,
    dataset,
    step_final: int,
    datasize: int = 2000,
):
    """
    Compute score L² errors and DSM losses over steps [initial_step, step_final).

    sde_history : Tensor[nsteps-initial_step+1, nsamples]
                  Index 0 = state at t=T, index -1 = state at t=0.
    """
    norm_values = []
    dsm_losses = []
    for step in range(initial_step, step_final):
        se, dl = error_norm_single_step(
            module, step, sde_history, initial_step, time, scheduler, dataset, datasize
        )
        norm_values.append(se)
        dsm_losses.append(dl)
    return norm_values, dsm_losses


def epoch_from_filename(filename: str):
    """Extract epoch int from 'sample-epoch=47-valid_loss=...' style names."""
    m = re.search(r"epoch=(\d+)", filename)
    return int(m.group(1)) if m else filename


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} (physical GPU {args.device_id})")

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
    print(f"Found {len(ckpt_files)} checkpoint(s):")
    for f in ckpt_files:
        print(f"  {f.name}")

    output_dir = pathlib.Path(__file__).parent.parent / "stats" / args.output_name
    output_dir.mkdir(exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    points = torch.tensor([-1.0, 0.1]).unsqueeze(-1)
    weights = torch.tensor([0.1, 0.9])
    scale_data = torch.tensor([0.2, 0.1])
    nsamples = args.nsamples

    dataset = diffsci.data.MixtureOfGaussiansDataset(
        num_samples=nsamples, means=points, weights=weights, scale=scale_data
    )
    dataset.num_samples = nsamples
    gaussian_samples2 = dataset.sample()

    # ── Model architecture template ────────────────────────────────────────
    width = 128
    mode = "default"
    model = Improved1DMLP(
        data_dim=1, time_embed_dim=width // 2,
        hidden_dim=width, num_layers=3, residual=False,
    )
    config = KarrasModuleConfig.from_edm()
    config.preconditioner = CustomEDMPreconditioner(mode=mode)

    # ── Scheduler & diffusion parameters ──────────────────────────────────
    nsteps = args.nsteps
    scheduler = diffsci.models.EDMScheduler()   # sigma_max=80 (default)

    initial_time_ = torch.tensor(float(args.initial_time))
    initial_step = scheduler.step_from_time(initial_time_, nsteps)
    initial_step_int = int(initial_step.item())
    time = scheduler.create_steps(nsteps + 1)
    initial_time = time[initial_step_int].item()
    print(f"Scheduler: EDMScheduler(sigma_max={scheduler.sigma_max.item():.1f})")
    print(f"initial_step = {initial_step_int},  initial_time = {initial_time:.4f}")

    # ── Diffused reference data (used as entropy reference) ────────────────
    tmin = time[-2]
    sigma_min = scheduler.scheduler_fns.noise_fn(tmin)
    scale_min = scheduler.scheduler_fns.scaling_fn(tmin)
    noise = torch.randn_like(gaussian_samples2) * sigma_min * scale_min
    diffused_data = (scale_min * gaussian_samples2 + noise).squeeze(-1)  # [nsamples]

    # ── Forward SDE history (computed once, shared across all checkpoints) ─
    print("\nGenerating forward SDE history (shared across checkpoints)…")
    scheduler.langevin_const = 1
    forw_sde_history_raw = scheduler.propagate_forward(
        gaussian_samples2, dataset.gradlogprob, nsteps,
        record_history=True, stochastic=True,
    ).cpu()
    # Take the first (nsteps - initial_step + 1) steps and flip so that:
    #   forward_sde_history[0]                    -> state at t = T  (high noise)
    #   forward_sde_history[step - initial_step]  -> state at time[step]
    #   forward_sde_history[-1]                   -> state at t ~ 0 (data)
    n_history = nsteps - initial_step_int + 1
    forward_sde_history = (
        forw_sde_history_raw[:n_history].squeeze(-1).flip(0)
    )   # shape: [n_history, nsamples]
    del forw_sde_history_raw
    print(f"forward_sde_history shape: {forward_sde_history.shape}")

    # ── Per-checkpoint computation ─────────────────────────────────────────
    all_entropies = {}   # epoch -> (gamma_values, sde_entropies, inv_sde_entropies)
    all_errors = {}      # epoch -> (error_values, dsm_losses)

    gamma_range = [args.gamma_min, args.gamma_max]

    for ckpt_path in ckpt_files:
        epoch = epoch_from_filename(ckpt_path.name)
        print(f"\n{'─'*60}")
        print(f"Checkpoint: {ckpt_path.name}  (epoch {epoch})")

        module = KarrasModule.load_from_checkpoint(
            model=model, config=config,
            checkpoint_path=str(ckpt_path),
        )
        module = module.to(device)
        module.eval()

        # 1) Generate multiple-gamma ODE/SDE samples (no file saving) ──────
        print("  [1/2] Generating multiple-gamma samples…")
        ode_samples, sde_samples, gamma_values = generate_multiple_gamma_nosave(
            initial_step=initial_step_int,
            gamma_range=gamma_range,
            nsteps=nsteps,
            nsamples=nsamples,
            dataset=dataset,
            ngamma=args.ngamma,
            noisescheduler=scheduler,
            module=module,
            use_exact_prior=False,
            alpha=0.5,
        )

        # 2) Compute KL divergences across gamma values ────────────────────
        gv, ent, inv_ent = compute_entropies_for_checkpoint(
            sde_samples, diffused_data, gamma_values, nbins=args.nbins,
        )
        all_entropies[epoch] = (gv, ent, inv_ent)
        best_idx = int(np.argmin(ent))
        print(f"    min KL(~p||p) = {min(ent):.6f}  at gamma = {gv[best_idx].item():.4f}")

        # 3) Compute score errors along forward SDE trajectory ─────────────
        print("  [2/2] Computing score error norms…")
        error_values, dsm_losses = compute_error_norm(
            module=module,
            initial_step=initial_step_int,
            sde_history=forward_sde_history,
            time=time,
            scheduler=scheduler,
            dataset=dataset,
            step_final=nsteps,
            datasize=args.datasize,
        )
        all_errors[epoch] = (error_values, dsm_losses)
        print(f"    mean score error = {np.mean(error_values):.6f}")

    # ── Save results ────────────────────────────────────────────────────────
    torch.save(all_entropies, output_dir / "all_entropies.pt")
    torch.save(all_errors, output_dir / "all_errors.pt")

    # Human-readable numpy dict
    np_results = {
        "time": time.numpy(),
        "initial_step": initial_step_int,
        "initial_time": initial_time,
        "gamma_range": gamma_range,
        "ngamma": args.ngamma,
        "entropies": {
            ep: {
                "gamma_values": gv.numpy(),
                "sde_entropies": np.array(ent),
                "inv_sde_entropies": np.array(inv_ent),
            }
            for ep, (gv, ent, inv_ent) in all_entropies.items()
        },
        "errors": {
            ep: {
                "error_values": np.array(ev),
                "dsm_losses": np.array(dl),
            }
            for ep, (ev, dl) in all_errors.items()
        },
    }
    np.save(output_dir / "results.npy", np_results, allow_pickle=True)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir}")
    print(f"  all_entropies.pt  – dict: epoch -> (gamma_values, sde_entropies, inv_sde_entropies)")
    print(f"  all_errors.pt     – dict: epoch -> (error_values, dsm_losses)")
    print(f"  results.npy       – combined numpy-friendly dict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute entropy and score-error profiles for all checkpoints in a folder."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/home/ubuntu/repos/DiffSci/savedmodels/production/mixt_gauss2_silu-width=128-bs=16-precond=edm-diffsci-train=4000-nlm=default8/checkpoints",
        help="Directory containing .ckpt files to iterate over.",
    )
    parser.add_argument("--nsteps",       type=int,   default=500,   help="Number of diffusion steps.")
    parser.add_argument("--initial_time", type=float, default=80.0,  help="Initial (max) diffusion time.")
    parser.add_argument("--gamma_min",    type=float, default=0.01,  help="Minimum gamma value.")
    parser.add_argument("--gamma_max",    type=float, default=20.0,  help="Maximum gamma value.")
    parser.add_argument("--ngamma",       type=int,   default=50,    help="Number of gamma values.")
    parser.add_argument("--nbins",        type=int,   default=80,    help="Histogram bins for KL estimation.")
    parser.add_argument("--nsamples",     type=int,   default=100000, help="Number of samples for generation.")
    parser.add_argument("--datasize",     type=int,   default=2000, help="Subset size for error norm computation.")
    parser.add_argument("--output_name",  type=str,   default="output_default8", help="Output folder name.")
    parser.add_argument(
        "--device_id",
        type=int,
        default=6,
        help="CUDA device index to use (sets CUDA_VISIBLE_DEVICES).",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    main(args)

#!/usr/bin/env python
"""
Test script: profile spatial-parallel UNet forward pass vs single-GPU reference.

Single forward pass with detailed timing breakdown.

Launch with:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
        scripts/0004d-porosity-field-generator-TEST.py \
        --stone Bentheimer --generation-case 5 \
        --checkpoint path/to/ckpt --pixel-size 1280
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
from diffsci2.extra import punetg_converters
from diffsci2.extra.matern_gaussian_process import MaternFieldSampler, PeriodicMaternFieldSampler
from diffsci2.distributed import SpatialContext, convert_to_spatial_parallel, scatter_along_dim, gather_along_dim


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASEPATH = os.path.join(os.path.dirname(__file__), '..')
NOTEBOOKPATH = os.path.join(BASEPATH, 'notebooks', 'exploratory', 'dfn')

GPDATA_DIR = os.path.join(NOTEBOOKPATH, 'data', 'gpdata3c')
GPDATA_PATHS = {
    'Bentheimer': os.path.join(GPDATA_DIR, 'bentheimer', 'bentheimer_porosity_analysis.npz'),
    'Doddington': os.path.join(GPDATA_DIR, 'doddington', 'doddington_porosity_analysis.npz'),
    'Estaillades': os.path.join(GPDATA_DIR, 'estaillades', 'estaillades_porosity_analysis.npz'),
    'Ketton': os.path.join(GPDATA_DIR, 'ketton', 'ketton_porosity_analysis.npz'),
}

GPDATA_129_DIR = os.path.join(NOTEBOOKPATH, 'data', 'gpdata4-129')
GPDATA_129_PATHS = {
    'Bentheimer': os.path.join(GPDATA_129_DIR, 'bentheimer', 'bentheimer_porosity_analysis.npz'),
    'Doddington': os.path.join(GPDATA_129_DIR, 'doddington', 'doddington_porosity_analysis.npz'),
    'Estaillades': os.path.join(GPDATA_129_DIR, 'estaillades', 'estaillades_porosity_analysis.npz'),
    'Ketton': os.path.join(GPDATA_129_DIR, 'ketton', 'ketton_porosity_analysis.npz'),
}

ORIGINAL_CHECKPOINTS = {
    'Bentheimer': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'bentheimer_pcond.ckpt'),
    'Doddington': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'doddington_pcond.ckpt'),
    'Estaillades': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'estaillades_pcond.ckpt'),
    'Ketton': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'ketton_pcond.ckpt'),
}

LATENT_TO_PIXEL_FACTOR = 8


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Profile spatial-parallel UNet forward pass')
    parser.add_argument('--stone', type=str, required=True,
                        choices=['Bentheimer', 'Doddington', 'Estaillades', 'Ketton'])
    parser.add_argument('--generation-case', type=int, default=5, choices=[1, 2, 4, 5, 6, 7])
    parser.add_argument('--pixel-size', type=int, default=1280)
    parser.add_argument('--coarse-n', type=int, default=32)
    parser.add_argument('--periodic', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_cpu(stone, generation_case, checkpoint_path):
    if generation_case == 7:
        assert checkpoint_path is not None
        ckpt = checkpoint_path
        weights = _load_weights(ckpt)
        model = diffsci2.nets.PUNetG(
            _punetg_config(cond_dropout=0.0), conditional_embedding=None)
        model.load_state_dict(weights)
    elif generation_case in [1, 5]:
        assert checkpoint_path is not None
        ckpt = checkpoint_path
        weights = _load_weights(ckpt)
        embedder = diffsci2.nets.ScalarEmbedder(dembed=64, key='porosity')
        model = diffsci2.nets.PUNetG(
            _punetg_config(cond_dropout=0.1), conditional_embedding=embedder)
        model.load_state_dict(weights)
    else:
        ckpt = ORIGINAL_CHECKPOINTS[stone]
        weights = _load_weights(ckpt)
        embedder = diffsci2.nets.ScalarEmbedder(dembed=64, key='porosity')
        model = diffsci2.nets.PUNetG(
            _punetg_config(cond_dropout=0.1), conditional_embedding=embedder)
        model.load_state_dict(weights)
    return model, ckpt


def _load_weights(ckpt_path):
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    return {k[len("model."):]: v for k, v in loaded.items() if k.startswith("model.")}


def _punetg_config(cond_dropout=0.1):
    return diffsci2.nets.PUNetGConfig(
        input_channels=4, output_channels=4, dimension=3,
        model_channels=64, channel_expansion=[2, 4],
        number_resnet_downward_block=2, number_resnet_upward_block=2,
        number_resnet_attn_block=0, number_resnet_before_attn_block=3,
        number_resnet_after_attn_block=3, kernel_size=3, in_out_kernel_size=3,
        in_embedding=False, time_projection_scale=10.0, input_projection_scale=1.0,
        transition_scale_factor=2, transition_kernel_size=3,
        dropout=0.1, cond_dropout=cond_dropout,
        first_resblock_norm="GroupLN", second_resblock_norm="GroupRMS",
        affine_norm=True, convolution_type="default", num_groups=1,
        attn_residual=False, attn_type="default", bias=True)


# ---------------------------------------------------------------------------
# Porosity
# ---------------------------------------------------------------------------

def sample_porosity_field(stone, latent_size, coarse_n, gpdata_paths, periodic):
    import scipy.special
    gpdata = np.load(gpdata_paths[stone])
    L = latent_size
    if periodic:
        sampler = PeriodicMaternFieldSampler(
            mean_val=float(gpdata['mean_logit']),
            sigma_sq=float(gpdata['matern_sigma_sq']),
            nu=float(gpdata['matern_nu']),
            length_scale=float(gpdata['matern_length_scale']))
        axes = [np.linspace(0, L, L) for _ in range(3)]
        sampler.initialize_field_from_grid(*axes)
        field = sampler.sample_grid(1)[0]
    else:
        sampler = MaternFieldSampler(
            mean_val=float(gpdata['mean_logit']),
            sigma_sq=float(gpdata['matern_sigma_sq']),
            nu=float(gpdata['matern_nu']),
            length_scale=float(gpdata['matern_length_scale']))
        x_c = np.linspace(0.5, L - 0.5, coarse_n)
        sampler.initialize_field_from_grid(x_c, x_c, x_c)
        x_f = np.linspace(0.5, L - 0.5, L)
        field = sampler.sample_grid_interpolated(1, x_f, x_f, x_f)[0]
    return scipy.special.expit(field).astype(np.float32)


# ---------------------------------------------------------------------------
# Timed forward pass helpers
# ---------------------------------------------------------------------------

def timed_forward(model, x, t, y, device, label, rank):
    """Run forward with warmup + timed pass using CUDA events."""
    # Warmup (first call has kernel launch overhead)
    log(rank, f"  [{label}] Warmup forward pass...")
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.inference_mode():
        _ = model(x, t, y)
    torch.cuda.synchronize(device)
    t_warmup = time.time() - t0
    log(rank, f"  [{label}] Warmup: {t_warmup:.3f}s")

    # Timed pass with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(device)
    start_event.record()
    with torch.inference_mode():
        out = model(x, t, y)
    end_event.record()
    torch.cuda.synchronize(device)

    gpu_ms = start_event.elapsed_time(end_event)
    peak = torch.cuda.max_memory_allocated(device) / 1e9

    log(rank, f"  [{label}] Timed: {gpu_ms:.1f}ms  Peak GPU: {peak:.2f} GB")
    log(rank, f"  [{label}] Output: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

    return out, gpu_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    latent_size = args.pixel_size // LATENT_TO_PIXEL_FACTOR
    local_d = latent_size // world_size
    generation_case = args.generation_case
    needs_porosity = generation_case in [1, 4, 5, 6]

    log(rank, f"=== 0004d UNet Forward Pass PROFILE ===")
    log(rank, f"Stone: {args.stone}  Case: {generation_case}  "
              f"Pixel: {args.pixel_size}  Latent: {latent_size}")
    log(rank, f"GPUs: {world_size}  Local D: {local_d}  Periodic: {args.periodic}")

    assert latent_size % world_size == 0

    # --- Load model on CPU ---
    t0 = time.time()
    model_cpu, ckpt_used = load_model_cpu(
        args.stone, generation_case, args.checkpoint)
    if args.periodic:
        model_cpu = punetg_converters.convert_conv_to_circular(model_cpu, [0, 1, 2], True)
    log(rank, f"Model loaded in {time.time()-t0:.2f}s: {ckpt_used}")

    # --- Create inputs on rank 0, broadcast ---
    log(rank, "\nGenerating shared inputs...")
    t0 = time.time()
    if rank == 0:
        x_full = torch.randn(1, 4, latent_size, latent_size, latent_size, device=device)
    else:
        x_full = torch.empty(1, 4, latent_size, latent_size, latent_size, device=device)
    dist.broadcast(x_full, src=0)
    t_val = torch.tensor([0.5], device=device)

    porosity_full = None
    if needs_porosity:
        gpdata_paths = GPDATA_129_PATHS if generation_case in [5, 6] else GPDATA_PATHS
        if rank == 0:
            porosity_np = sample_porosity_field(
                args.stone, latent_size, args.coarse_n, gpdata_paths, args.periodic)
            porosity_full = torch.tensor(porosity_np, dtype=torch.float32, device=device)
            log(rank, f"  Porosity range: [{porosity_np.min():.4f}, {porosity_np.max():.4f}]")
        else:
            porosity_full = torch.empty(
                latent_size, latent_size, latent_size, dtype=torch.float32, device=device)
        dist.broadcast(porosity_full, src=0)
    log(rank, f"  Inputs ready in {time.time()-t0:.2f}s")

    dist.barrier()

    # =====================================================================
    # Phase 1: Single-GPU reference (rank 0 only)
    # =====================================================================
    out_ref = None
    t_ref_ms = None
    if rank == 0:
        log(rank, "\n--- Phase 1: Single-GPU forward ---")
        model_ref = copy.deepcopy(model_cpu).to(device).eval()

        if needs_porosity:
            y_ref = {'porosity': porosity_full.unsqueeze(0)}
        else:
            y_ref = None

        torch.cuda.reset_peak_memory_stats(device)
        out_ref, t_ref_ms = timed_forward(model_ref, x_full, t_val, y_ref, device, "1GPU", rank)

        del model_ref
        torch.cuda.empty_cache()

    dist.barrier()

    # =====================================================================
    # Phase 2: Spatial-parallel forward
    # =====================================================================
    log(rank, "\n--- Phase 2: Spatial-parallel forward ---")

    t0 = time.time()
    model_par = copy.deepcopy(model_cpu)
    del model_cpu

    ctx = SpatialContext(
        rank=rank, world_size=world_size,
        process_group=dist.group.WORLD, split_dim=2,
        periodic=args.periodic)

    model_par = convert_to_spatial_parallel(model_par, ctx, inplace=True)
    model_par.to(device).eval()
    log(rank, f"  Conversion + to(device): {time.time()-t0:.2f}s")

    # Scatter inputs
    t0 = time.time()
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)
    del x_full

    if needs_porosity:
        porosity_local = scatter_along_dim(porosity_full, dim=0, ctx=ctx)
        y_local = {'porosity': porosity_local.unsqueeze(0)}
        del porosity_full
    else:
        y_local = None
    log(rank, f"  Scatter: {time.time()-t0:.4f}s")

    torch.cuda.reset_peak_memory_stats(device)
    out_local, t_par_ms = timed_forward(model_par, x_local, t_val, y_local, device, f"PAR-r{rank}", rank)

    # Gather
    t0 = time.time()
    out_par_full = gather_along_dim(out_local, dim=2, ctx=ctx)
    torch.cuda.synchronize(device)
    log(rank, f"  Gather: {time.time()-t0:.4f}s")

    dist.barrier()

    # =====================================================================
    # Phase 3: Compare
    # =====================================================================
    if rank == 0:
        log(rank, "\n--- Comparison ---")
        diff = (out_ref - out_par_full).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        log(rank, f"  max_diff  = {max_diff:.6e}")
        log(rank, f"  mean_diff = {mean_diff:.6e}")
        for c in range(4):
            ch_diff = (out_ref[0, c] - out_par_full[0, c]).abs()
            log(rank, f"  Ch {c}: max={ch_diff.max().item():.6e}  mean={ch_diff.mean().item():.6e}")

        log(rank, f"\n  1-GPU:     {t_ref_ms:.1f}ms")
        log(rank, f"  Parallel:  {t_par_ms:.1f}ms")
        log(rank, f"  Speedup:   {t_ref_ms / t_par_ms:.2f}x")

        if max_diff < 0.05:
            log(rank, "  PASS")
        else:
            log(rank, f"  FAIL (max_diff = {max_diff:.4f})")

    dist.barrier()
    dist.destroy_process_group()
    log(rank, "\nDone!")


if __name__ == '__main__':
    main()

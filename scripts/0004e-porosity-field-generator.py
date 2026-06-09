#!/usr/bin/env python
"""
Two-stage porosity field generator with non-cubic volume support.

Decomposes generation into two independent stages:

  Stage 1 (--mode latent): Generate latent volumes.
    - Multi-GPU with torchrun for spatial parallelism
    - Single-GPU with plain python (for testing / small volumes)
    - Saves latent .pt files and config to output-dir/latents/

  Stage 2 (--mode decode): Decode saved latents to pixel-space volumes.
    - Single GPU only (no torchrun needed)
    - Reads config + latents from output-dir/, writes .npy to output-dir/data/

Supports non-cubic volumes via --volume-shape (e.g., "1280x1280x4352").

Supports the same generation cases as 0004c/0004d:
  Case 1: Field porosity with post-trained model
  Case 2: Null conditioning
  Case 3: Scalar porosity
  Case 4: Field porosity with original model
  Case 5: Field porosity with 129-trained model
  Case 6: Field porosity with original model (gpdata4-129)
  Case 7: Unconditional with provided checkpoint
  Case 8: Field porosity with 257-trained model

Usage:
    # Stage 1 (multi-GPU):
    torchrun --nproc_per_node=4 scripts/0004e-porosity-field-generator.py \\
        --mode latent \\
        --checkpoint /path/to/ckpt \\
        --stone Bentheimer \\
        --output-dir ./output/ \\
        --generation-case 5 \\
        --volume-shape 1280x1280x4352 \\
        --volume-samples 1 \\
        --save-porosity

    # Stage 1 (single-GPU, for testing):
    python scripts/0004e-porosity-field-generator.py \\
        --mode latent \\
        --checkpoint /path/to/ckpt \\
        --stone Bentheimer \\
        --output-dir ./output/ \\
        --generation-case 5 \\
        --volume-shape 256 \\
        --volume-samples 1 \\
        --device cuda:0

    # Stage 2 (always single-GPU):
    python scripts/0004e-porosity-field-generator.py \\
        --mode decode \\
        --output-dir ./output/ \\
        --device cuda:0
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

# Add the aux directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
from model_loaders import load_autoencoder
from diffsci2.extra import chunk_decode_2, punetg_converters
from diffsci2.extra.matern_gaussian_process import MaternFieldSampler


AVAILABLE_STONES = ['Bentheimer', 'Doddington', 'Estaillades', 'Ketton']

# Constants
LATENT_TO_PIXEL_FACTOR = 8  # pixel_size = latent_size * 8
MIN_LATENT_MULTIPLE = 16    # latent_size must be multiple of 16

# Base paths
BASEPATH = os.path.join(os.path.dirname(__file__), '..')
NOTEBOOKPATH = os.path.join(BASEPATH, 'notebooks', 'exploratory', 'dfn')

# Paths to GP analysis data (fitted in latent space with voxel_size=1.0)
GPDATA_DIR = os.path.join(NOTEBOOKPATH, 'data', 'gpdata3c')
GPDATA_PATHS = {
    'Bentheimer': os.path.join(GPDATA_DIR, 'bentheimer', 'bentheimer_porosity_analysis.npz'),
    'Doddington': os.path.join(GPDATA_DIR, 'doddington', 'doddington_porosity_analysis.npz'),
    'Estaillades': os.path.join(GPDATA_DIR, 'estaillades', 'estaillades_porosity_analysis.npz'),
    'Ketton': os.path.join(GPDATA_DIR, 'ketton', 'ketton_porosity_analysis.npz'),
}

# Paths to GP analysis data for 129 models (fitted on 129-resolution data)
GPDATA_129_DIR = os.path.join(NOTEBOOKPATH, 'data', 'gpdata4-129')
GPDATA_129_PATHS = {
    'Bentheimer': os.path.join(GPDATA_129_DIR, 'bentheimer', 'bentheimer_porosity_analysis.npz'),
    'Doddington': os.path.join(GPDATA_129_DIR, 'doddington', 'doddington_porosity_analysis.npz'),
    'Estaillades': os.path.join(GPDATA_129_DIR, 'estaillades', 'estaillades_porosity_analysis.npz'),
    'Ketton': os.path.join(GPDATA_129_DIR, 'ketton', 'ketton_porosity_analysis.npz'),
}

# Paths to GP analysis data for 257 models
GPDATA_257_DIR = os.path.join(NOTEBOOKPATH, 'data', 'gpdata4-257')
GPDATA_257_PATHS = {
    'Bentheimer': os.path.join(GPDATA_257_DIR, 'bentheimer', 'bentheimer_porosity_analysis.npz'),
    'Doddington': os.path.join(GPDATA_257_DIR, 'doddington', 'doddington_porosity_analysis.npz'),
    'Estaillades': os.path.join(GPDATA_257_DIR, 'estaillades', 'estaillades_porosity_analysis.npz'),
    'Ketton': os.path.join(GPDATA_257_DIR, 'ketton', 'ketton_porosity_analysis.npz'),
}

# Paths to original (scalar-trained) checkpoints for cases 2, 3, 4
ORIGINAL_CHECKPOINTS = {
    'Bentheimer': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'bentheimer_pcond.ckpt'),
    'Doddington': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'doddington_pcond.ckpt'),
    'Estaillades': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'estaillades_pcond.ckpt'),
    'Ketton': os.path.join(BASEPATH, 'savedmodels', 'pore', 'production', 'ketton_pcond.ckpt'),
}

# Paths to real porosity volumes (for sampling scalar values in case 3)
POROSITY_VOLUMES = {
    'Bentheimer': os.path.join(NOTEBOOKPATH, 'data', 'gpdata2', 'bentheimer', 'Bentheimer_1000c_3p0035um_porosity_field_full.npy'),
    'Doddington': os.path.join(NOTEBOOKPATH, 'data', 'gpdata2', 'doddington', 'Doddington_1000c_2p6929um_porosity_field_full.npy'),
    'Estaillades': os.path.join(NOTEBOOKPATH, 'data', 'gpdata2', 'estaillades', 'Estaillades_1000c_3p31136um_porosity_field_full.npy'),
    'Ketton': os.path.join(NOTEBOOKPATH, 'data', 'gpdata2', 'ketton', 'Ketton_1000c_3p00006um_porosity_field_full.npy'),
}

# Paths to real porosity volumes for 129 models
POROSITY_VOLUMES_129 = {
    'Bentheimer': os.path.join(GPDATA_129_DIR, 'bentheimer', 'bentheimer_porosity_field_full.npy'),
    'Doddington': os.path.join(GPDATA_129_DIR, 'doddington', 'doddington_porosity_field_full.npy'),
    'Estaillades': os.path.join(GPDATA_129_DIR, 'estaillades', 'estaillades_porosity_field_full.npy'),
    'Ketton': os.path.join(GPDATA_129_DIR, 'ketton', 'ketton_porosity_field_full.npy'),
}

# Paths to real porosity volumes for 257 models
POROSITY_VOLUMES_257 = {
    'Bentheimer': os.path.join(GPDATA_257_DIR, 'bentheimer', 'bentheimer_porosity_field_full.npy'),
    'Doddington': os.path.join(GPDATA_257_DIR, 'doddington', 'doddington_porosity_field_full.npy'),
    'Estaillades': os.path.join(GPDATA_257_DIR, 'estaillades', 'estaillades_porosity_field_full.npy'),
    'Ketton': os.path.join(GPDATA_257_DIR, 'ketton', 'ketton_porosity_field_full.npy'),
}


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Two-stage porosity field generator (latent generation + decode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--mode', type=str, required=True, choices=['latent', 'decode'],
        help='Stage to run: "latent" generates and saves latent volumes, '
             '"decode" decodes saved latents to pixel space'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory. Latent mode writes to output-dir/latents/, '
             'decode mode reads from there and writes to output-dir/data/'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='GPU device (for single-GPU latent mode and decode mode)'
    )

    # --- Latent generation args ---
    latent_group = parser.add_argument_group('Latent generation (--mode latent)')
    latent_group.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (required for cases 1, 5, 7, 8)'
    )
    latent_group.add_argument(
        '--stone', type=str, default=None, choices=AVAILABLE_STONES,
        help='Stone type for GP parameters'
    )
    latent_group.add_argument(
        '--generation-case', type=int, default=5,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help='Generation case (default: 5)'
    )
    latent_group.add_argument(
        '--volume-shape', type=str, default='256',
        help='Volume shape in pixels: "N" for cubic NxNxN, or "DxHxW" for non-cubic '
             '(e.g., "1280x1280x4352"). Each dimension must be a multiple of 128.'
    )
    latent_group.add_argument(
        '--volume-samples', type=int, default=1,
        help='Number of latent samples to generate (default: 1)'
    )
    latent_group.add_argument(
        '--nsteps', type=int, default=21,
        help='Number of sampling steps (default: 21)'
    )
    latent_group.add_argument(
        '--guidance', type=float, default=1.0,
        help='Classifier-free guidance scale (default: 1.0, no guidance)'
    )
    latent_group.add_argument(
        '--coarse-n', type=int, default=32,
        help='Coarse grid size for GP sampling (default: 32)'
    )
    latent_group.add_argument(
        '--save-porosity', action='store_true',
        help='Save the input porosity field alongside each latent'
    )
    latent_group.add_argument(
        '--periodic', action='store_true',
        help='Use periodic (circular) convolutions and periodic porosity conditioning'
    )

    # --- Decode args ---
    decode_group = parser.add_argument_group('Decoding (--mode decode)')
    decode_group.add_argument(
        '--no-binarize', action='store_true',
        help='Save raw float values instead of binarized (bool) volumes'
    )
    decode_group.add_argument(
        '--chunk-size', type=int, default=40,
        help='Chunk size for VAE decoding (default: 40)'
    )
    decode_group.add_argument(
        '--disk-offload', action='store_true',
        help='Use disk-backed mmap buffers for chunk decode (for very large volumes)'
    )
    decode_group.add_argument(
        '--disk-offload-dir', type=str, default=None,
        help='Directory for mmap temp files (default: /tmp/chunk_decode_mmap)'
    )

    return parser.parse_args()


# ============================================================================
# Utilities
# ============================================================================

def parse_shape(s):
    """Parse shape string: '256' -> (256,256,256), '1280x1280x4352' -> (1280,1280,4352)."""
    parts = s.replace('x', ',').split(',')
    parts = [int(p.strip()) for p in parts]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    elif len(parts) == 3:
        return tuple(parts)
    else:
        raise ValueError(f"Invalid shape '{s}'. Use 'N' for cubic or 'DxHxW' for non-cubic.")


def validate_shape(pixel_shape):
    """Validate pixel shape and return latent shape (D, H, W)."""
    latent_shape = []
    for i, ps in enumerate(pixel_shape):
        multiple = LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE  # 128
        if ps % multiple != 0:
            raise ValueError(
                f"Dimension {i} of pixel shape ({ps}) must be a multiple of {multiple}. "
                f"Got pixel_shape={pixel_shape}."
            )
        latent_shape.append(ps // LATENT_TO_PIXEL_FACTOR)
    return tuple(latent_shape)


def is_distributed():
    """Check if we're running under torchrun (distributed context)."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def log(rank, msg):
    """Print only from rank 0."""
    if rank == 0:
        print(msg, flush=True)


def get_case_description(case):
    descriptions = {
        1: "Field porosity with post-trained (field-trained) model",
        2: "Null conditioning (y=None)",
        3: "Scalar porosity (random from real data)",
        4: "Field porosity with original (scalar-trained) model",
        5: "Field porosity with 129-trained model (gpdata4-129)",
        6: "Field porosity with original model (gpdata4-129)",
        7: "Unconditional with provided checkpoint",
        8: "Field porosity with 257-trained model (gpdata4-257)",
    }
    return descriptions.get(case, "Unknown case")


def get_gpdata_paths_for_case(case):
    if case in [5, 6]:
        return GPDATA_129_PATHS
    elif case == 8:
        return GPDATA_257_PATHS
    return None


def resolve_checkpoint(args):
    """Resolve checkpoint path based on generation case."""
    case = args.generation_case
    if case in [1, 5, 7, 8]:
        if args.checkpoint is None:
            raise ValueError(f"--checkpoint is required for generation case {case}")
        return args.checkpoint
    else:
        return ORIGINAL_CHECKPOINTS[args.stone]


# ============================================================================
# Model building
# ============================================================================

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


def _load_weights_cpu(checkpoint_path):
    """Load model weights from checkpoint on CPU."""
    loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['state_dict']
    return {k[len("model."):]: v for k, v in loaded.items() if k.startswith("model.")}


def _build_flowmodel(weights, unconditional):
    """Build PUNetG flow model and load weights."""
    if unconditional:
        flowmodel = diffsci2.nets.PUNetG(
            _punetg_config(cond_dropout=0.0), conditional_embedding=None)
    else:
        embedder = diffsci2.nets.ScalarEmbedder(dembed=64, key='porosity')
        flowmodel = diffsci2.nets.PUNetG(
            _punetg_config(cond_dropout=0.1), conditional_embedding=embedder)
    flowmodel.load_state_dict(weights)
    return flowmodel


def _build_simodule(flowmodel):
    """Create SIModule (no autoencoder — we decode separately)."""
    config = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=0.002, sigma_max=80.0, sigma_data=0.5,
        initial_norm=20.0, loss_formulation='denoiser')
    return diffsci2.models.SIModule(
        config=config, model=flowmodel, autoencoder=None)


# ============================================================================
# GP porosity sampling
# ============================================================================

def load_gpdata(stone, gpdata_paths=None):
    if gpdata_paths is None:
        gpdata_paths = GPDATA_PATHS
    return np.load(gpdata_paths[stone])


def load_porosity_volume(stone, porosity_volumes=None):
    if porosity_volumes is None:
        porosity_volumes = POROSITY_VOLUMES
    volume = np.load(porosity_volumes[stone])
    volume = volume[128:-128, 128:-128, 128:-128]
    return volume


def sample_scalar_porosity(porosity_volume):
    flat = porosity_volume.flatten()
    return np.array([np.random.choice(flat)], dtype=np.float32)


def create_porosity_sampler(stone, coarse_n=16, gpdata_paths=None, periodic=False):
    gpdata = load_gpdata(stone, gpdata_paths=gpdata_paths)
    if periodic:
        from diffsci2.extra.matern_gaussian_process import PeriodicMaternFieldSampler
        sampler = PeriodicMaternFieldSampler(
            mean_val=float(gpdata['mean_logit']),
            sigma_sq=float(gpdata['matern_sigma_sq']),
            nu=float(gpdata['matern_nu']),
            length_scale=float(gpdata['matern_length_scale']))
    else:
        sampler = MaternFieldSampler(
            mean_val=float(gpdata['mean_logit']),
            sigma_sq=float(gpdata['matern_sigma_sq']),
            nu=float(gpdata['matern_nu']),
            length_scale=float(gpdata['matern_length_scale']))
    return sampler, gpdata


def sample_porosity_field(sampler, latent_shape, coarse_n=16, periodic=False):
    """Sample a porosity field matching the given latent shape (D, H, W)."""
    import scipy.special
    Ld, Lh, Lw = latent_shape
    if periodic:
        axes = [np.linspace(0, s, s) for s in latent_shape]
        sampler.initialize_field_from_grid(*axes)
        field = sampler.sample_grid(1)[0]
    else:
        x_coarse = np.linspace(0.5, Ld - 0.5, coarse_n)
        y_coarse = np.linspace(0.5, Lh - 0.5, coarse_n)
        z_coarse = np.linspace(0.5, Lw - 0.5, coarse_n)
        sampler.initialize_field_from_grid(x_coarse, y_coarse, z_coarse)
        x_fine = np.linspace(0.5, Ld - 0.5, Ld)
        y_fine = np.linspace(0.5, Lh - 0.5, Lh)
        z_fine = np.linspace(0.5, Lw - 0.5, Lw)
        field = sampler.sample_grid_interpolated(1, x_fine, y_fine, z_fine)[0]
    porosity = scipy.special.expit(field)
    return porosity.astype(np.float32)


# ============================================================================
# Conditioning setup (shared by both GPU paths)
# ============================================================================

def setup_conditioning(args, case, latent_shape):
    """Setup porosity sampler / volume for the generation case. Returns (sampler, porosity_volume)."""
    sampler = None
    porosity_volume = None
    if case in [1, 4, 5, 6, 8]:
        gpdata_paths = get_gpdata_paths_for_case(case)
        sampler, gpdata = create_porosity_sampler(
            args.stone, args.coarse_n, gpdata_paths, periodic=args.periodic)
        print(f"  Porosity sampler for {args.stone}: "
              f"mean_logit={gpdata['mean_logit']:.4f}, "
              f"sigma^2={gpdata['matern_sigma_sq']:.4f}, "
              f"nu={gpdata['matern_nu']:.4f}, "
              f"l={gpdata['matern_length_scale']:.4f}")
    elif case == 3:
        porosity_volume = load_porosity_volume(args.stone)
        print(f"  Porosity volume: shape={porosity_volume.shape}, "
              f"range=[{porosity_volume.min():.4f}, {porosity_volume.max():.4f}]")
    return sampler, porosity_volume


def sample_conditioning(case, sampler, porosity_volume, latent_shape, coarse_n, periodic):
    """Sample conditioning for one volume. Returns (y_dict_with_tensors, porosity_data_numpy)."""
    if case in [1, 4, 5, 6, 8]:
        porosity_field = sample_porosity_field(sampler, latent_shape, coarse_n, periodic)
        y = {'porosity': torch.tensor(porosity_field, dtype=torch.float32)}
        return y, porosity_field
    elif case == 3:
        scalar = sample_scalar_porosity(porosity_volume)
        y = {'porosity': torch.tensor(scalar, dtype=torch.float32)}
        return y, scalar
    else:  # case 2, 7
        return None, None


# ============================================================================
# Stage 1: Latent generation — single GPU
# ============================================================================

def run_latent_single_gpu(args):
    pixel_shape = parse_shape(args.volume_shape)
    latent_shape = validate_shape(pixel_shape)
    device = args.device
    case = args.generation_case
    checkpoint_path = resolve_checkpoint(args)

    print(f"\n{'='*60}")
    print(f"  Latent Generation — single GPU ({device})")
    print(f"{'='*60}")
    print(f"  Case {case}: {get_case_description(case)}")
    print(f"  Stone: {args.stone}")
    print(f"  Pixel shape:  {pixel_shape}")
    print(f"  Latent shape: {latent_shape}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Samples: {args.volume_samples}")

    # Create output dirs
    latent_dir = os.path.join(args.output_dir, 'latents')
    os.makedirs(latent_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    t_load = time.time()
    weights = _load_weights_cpu(checkpoint_path)
    unconditional = (case == 7)
    flowmodel = _build_flowmodel(weights, unconditional)
    del weights
    if args.periodic:
        print("  Converting to circular convolutions")
        flowmodel = punetg_converters.convert_conv_to_circular(flowmodel, [0, 1, 2], True)
    flowmodule = _build_simodule(flowmodel)
    flowmodule.to(device)
    model_load_time = time.time() - t_load
    print(f"  Model loaded in {model_load_time:.1f}s")

    # Setup conditioning
    sampler, porosity_volume = setup_conditioning(args, case, latent_shape)

    # Config to save
    config = {
        'stone': args.stone,
        'checkpoint': checkpoint_path,
        'generation_case': case,
        'case_description': get_case_description(case),
        'pixel_shape': list(pixel_shape),
        'latent_shape': list(latent_shape),
        'volume_samples': args.volume_samples,
        'nsteps': args.nsteps,
        'guidance': args.guidance,
        'coarse_n': args.coarse_n,
        'periodic': args.periodic,
        'binarize': not args.no_binarize,
        'model_load_time': model_load_time,
        'distributed': False,
        'samples': [],
    }

    # Generate latents
    for i in range(args.volume_samples):
        print(f"\n  [{i+1}/{args.volume_samples}] Generating latent...")
        t_start = time.time()

        y, porosity_data = sample_conditioning(
            case, sampler, porosity_volume, latent_shape, args.coarse_n, args.periodic)

        latent = flowmodule.sample(
            1, shape=[4, *latent_shape],
            y=y, nsteps=args.nsteps,
            is_latent_shape=True, return_latents=True,
            guidance=args.guidance)

        latent_cpu = latent.cpu()
        del latent
        torch.cuda.empty_cache()

        sample_time = time.time() - t_start

        # Save latent
        latent_path = os.path.join(latent_dir, f'{i}.latent.pt')
        torch.save(latent_cpu, latent_path)
        del latent_cpu
        print(f"    Saved {latent_path} ({sample_time:.1f}s)")

        sample_info = {
            'index': i,
            'latent_file': f'{i}.latent.pt',
            'sample_time': sample_time,
        }

        # Save porosity
        if args.save_porosity and porosity_data is not None:
            poro_file = f'{i}.scalarporosity.npy' if case == 3 else f'{i}.porosity.npy'
            np.save(os.path.join(latent_dir, poro_file), porosity_data)
            sample_info['porosity_file'] = poro_file
            print(f"    Saved porosity: {poro_file}")

        config['samples'].append(sample_info)

    # Save config
    config_path = os.path.join(args.output_dir, 'latent_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")
    print("Latent generation complete.")


# ============================================================================
# Stage 1: Latent generation — distributed (torchrun)
# ============================================================================

def auto_select_split_axis(latent_shape, world_size):
    """Select best spatial axis (0=D, 1=H, 2=W) to split for parallelism.

    Picks the largest dimension that is evenly divisible by world_size.
    """
    axis_names = ['D', 'H', 'W']
    valid = [(i, latent_shape[i]) for i in range(3) if latent_shape[i] % world_size == 0]
    if not valid:
        # Compute valid world sizes for each axis
        msg_parts = []
        for i in range(3):
            divisors = sorted(d for d in range(2, latent_shape[i] + 1)
                              if latent_shape[i] % d == 0 and d <= 32)
            msg_parts.append(f"  {axis_names[i]}={latent_shape[i]}: nproc_per_node in {divisors}")
        raise ValueError(
            f"No spatial axis of latent shape {latent_shape} is divisible by "
            f"world_size={world_size}.\n"
            f"Valid --nproc_per_node values per axis:\n" +
            "\n".join(msg_parts))
    # Pick axis with largest dimension (best load balance)
    best_axis = max(valid, key=lambda x: x[1])[0]
    return best_axis


def run_latent_distributed(args):
    import torch.distributed as dist
    from diffsci2.distributed import (
        SpatialContext, convert_to_spatial_parallel,
        scatter_along_dim, gather_along_dim,
    )

    # Initialize distributed
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    pixel_shape = parse_shape(args.volume_shape)
    latent_shape = validate_shape(pixel_shape)
    case = args.generation_case
    checkpoint_path = resolve_checkpoint(args)
    unconditional = (case == 7)

    # Auto-select split axis
    split_axis = auto_select_split_axis(latent_shape, world_size)
    split_dim_5d = split_axis + 2   # dim index in [B, C, D, H, W]
    split_dim_3d = split_axis       # dim index in [D, H, W]
    local_size = latent_shape[split_axis] // world_size
    axis_names = ['D', 'H', 'W']

    log(rank, f"\n{'='*60}")
    log(rank, f"  Latent Generation — {world_size} GPUs (spatial parallel)")
    log(rank, f"{'='*60}")
    log(rank, f"  Case {case}: {get_case_description(case)}")
    log(rank, f"  Stone: {args.stone}")
    log(rank, f"  Pixel shape:  {pixel_shape}")
    log(rank, f"  Latent shape: {latent_shape}")
    log(rank, f"  Split axis: {axis_names[split_axis]} (dim {split_dim_5d}), "
              f"local={local_size}, world_size={world_size}")
    log(rank, f"  Checkpoint: {checkpoint_path}")

    # Create spatial context
    ctx = SpatialContext(
        rank=rank, world_size=world_size,
        process_group=dist.group.WORLD,
        split_dim=split_dim_5d,
        periodic=args.periodic)

    # Create output dirs (rank 0)
    latent_dir = os.path.join(args.output_dir, 'latents')
    if rank == 0:
        os.makedirs(latent_dir, exist_ok=True)

    # Load model with spatial parallel
    log(rank, "\nLoading model...")
    t_load = time.time()
    weights = _load_weights_cpu(checkpoint_path)
    flowmodel = _build_flowmodel(weights, unconditional)
    del weights
    if args.periodic:
        log(rank, "  Converting to circular convolutions")
        flowmodel = punetg_converters.convert_conv_to_circular(flowmodel, [0, 1, 2], True)
    flowmodule = _build_simodule(flowmodel)
    flowmodule.to(device)
    flowmodule.model = convert_to_spatial_parallel(flowmodule.model, ctx, inplace=True)
    flowmodule.to(device)
    model_load_time = time.time() - t_load
    log(rank, f"  Model loaded in {model_load_time:.1f}s")

    # Setup conditioning (rank 0 only for samplers)
    sampler = None
    porosity_volume = None
    if rank == 0:
        sampler, porosity_volume = setup_conditioning(args, case, latent_shape)

    dist.barrier()

    # Config (rank 0 will populate)
    config = {
        'stone': args.stone,
        'checkpoint': checkpoint_path,
        'generation_case': case,
        'case_description': get_case_description(case),
        'pixel_shape': list(pixel_shape),
        'latent_shape': list(latent_shape),
        'volume_samples': args.volume_samples,
        'nsteps': args.nsteps,
        'guidance': args.guidance,
        'coarse_n': args.coarse_n,
        'periodic': args.periodic,
        'binarize': not args.no_binarize,
        'model_load_time': model_load_time,
        'distributed': True,
        'world_size': world_size,
        'split_axis': split_axis,
        'samples': [],
    }

    # Generate latents
    for i in range(args.volume_samples):
        log(rank, f"\n  [{i+1}/{args.volume_samples}] Generating latent...")
        t_start = time.time()

        # --- Prepare conditioning ---
        y_local = None
        porosity_data = None  # numpy, rank 0 only

        if case in [1, 4, 5, 6, 8]:
            # Field porosity: sample on rank 0, broadcast, scatter
            if rank == 0:
                porosity_field = sample_porosity_field(
                    sampler, latent_shape, args.coarse_n, args.periodic)
                porosity_data = porosity_field
                porosity_full = torch.tensor(porosity_field, dtype=torch.float32, device=device)
            else:
                porosity_full = torch.empty(*latent_shape, dtype=torch.float32, device=device)
            dist.broadcast(porosity_full, src=0)
            porosity_local = scatter_along_dim(porosity_full, dim=split_dim_3d, ctx=ctx)
            del porosity_full
            y_local = {'porosity': porosity_local}

        elif case == 3:
            # Scalar porosity: sample on rank 0, broadcast
            if rank == 0:
                scalar = sample_scalar_porosity(porosity_volume)
                porosity_data = scalar
                scalar_tensor = torch.tensor(scalar, dtype=torch.float32, device=device)
            else:
                scalar_tensor = torch.empty(1, dtype=torch.float32, device=device)
            dist.broadcast(scalar_tensor, src=0)
            y_local = {'porosity': scalar_tensor}

        # else: case 2, 7 — y_local stays None

        # --- Generate and scatter noise ---
        if rank == 0:
            noise_full = torch.randn(1, 4, *latent_shape, device=device)
        else:
            noise_full = torch.empty(1, 4, *latent_shape, device=device)
        dist.broadcast(noise_full, src=0)
        noise_local = scatter_along_dim(noise_full, dim=split_dim_5d, ctx=ctx)
        del noise_full

        # --- Build local shape ---
        local_latent = list(latent_shape)
        local_latent[split_axis] = local_size

        log(rank, f"    Sampling (local shape: [4, {local_latent[0]}, {local_latent[1]}, {local_latent[2]}])...")
        t_sample = time.time()

        x_local = flowmodule.sample(
            1, shape=[4, *local_latent],
            y=y_local, nsteps=args.nsteps,
            is_latent_shape=True, return_latents=True,
            guidance=args.guidance, orig_noise=noise_local)

        log(rank, f"    Sampling done in {time.time() - t_sample:.1f}s")

        # --- Gather on rank 0 ---
        log(rank, "    Gathering latent...")
        x_full = gather_along_dim(x_local, dim=split_dim_5d, ctx=ctx)
        del x_local
        torch.cuda.empty_cache()

        sample_time = time.time() - t_start

        if rank == 0:
            latent_cpu = x_full.cpu()
            del x_full
            torch.cuda.empty_cache()

            latent_path = os.path.join(latent_dir, f'{i}.latent.pt')
            torch.save(latent_cpu, latent_path)
            del latent_cpu
            log(rank, f"    Saved {latent_path} ({sample_time:.1f}s)")

            sample_info = {
                'index': i,
                'latent_file': f'{i}.latent.pt',
                'sample_time': sample_time,
            }

            if args.save_porosity and porosity_data is not None:
                poro_file = f'{i}.scalarporosity.npy' if case == 3 else f'{i}.porosity.npy'
                np.save(os.path.join(latent_dir, poro_file), porosity_data)
                sample_info['porosity_file'] = poro_file
                log(rank, f"    Saved porosity: {poro_file}")

            config['samples'].append(sample_info)
        else:
            del x_full
            torch.cuda.empty_cache()

        dist.barrier()

    # Cleanup distributed
    log(rank, "\nDistributed sampling complete.")
    del flowmodule
    torch.cuda.empty_cache()
    dist.destroy_process_group()

    # Save config (rank 0 only)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'latent_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")
        print("Latent generation complete.")


# ============================================================================
# Stage 2: Decode
# ============================================================================

def run_decode(args):
    config_path = os.path.join(args.output_dir, 'latent_config.json')
    with open(config_path) as f:
        config = json.load(f)

    latent_dir = os.path.join(args.output_dir, 'latents')
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    pixel_shape = tuple(config['pixel_shape'])
    latent_shape = tuple(config['latent_shape'])
    binarize = config.get('binarize', True)
    if args.no_binarize:
        binarize = False
    periodic = config.get('periodic', False)
    device = args.device

    print(f"\n{'='*60}")
    print(f"  Decoding Latents")
    print(f"{'='*60}")
    print(f"  Pixel shape:  {pixel_shape}")
    print(f"  Latent shape: {latent_shape}")
    print(f"  Binarize: {binarize}")
    print(f"  Periodic: {periodic}")
    print(f"  Device: {device}")
    print(f"  Chunk size: {args.chunk_size}")
    if args.disk_offload:
        print(f"  Disk offload: {args.disk_offload_dir or '/tmp/chunk_decode_mmap'}")
    print(f"  Samples to decode: {len(config['samples'])}")

    # Load VAE
    print("\nLoading autoencoder...")
    t_load = time.time()
    vaemodule = load_autoencoder()
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    periodicity = [True, True, True] if periodic else [False, False, False]
    chunk_size = args.chunk_size

    decode_timing = []

    for sample_info in config['samples']:
        idx = sample_info['index']
        latent_file = sample_info['latent_file']
        latent_path = os.path.join(latent_dir, latent_file)

        print(f"\n  [{idx+1}/{len(config['samples'])}] Decoding {latent_file}...")
        t_start = time.time()

        # Load latent from disk
        latent = torch.load(latent_path, map_location='cpu', weights_only=True)
        print(f"    Latent shape: {list(latent.shape)}")

        # Decode
        chunk_decode_2.prepare_decoder_for_cached_decode(vaemodule.decoder)
        vaemodule.decoder.to(device)
        latent = latent.to(device)

        x = chunk_decode_2.chunk_decode_3d(
            vaemodule.decoder,
            latent,
            [chunk_size, chunk_size, chunk_size],
            device=device,
            periodicity=periodicity,
            use_cached_norms=True,
            use_disk_offload=args.disk_offload,
            disk_offload_dir=args.disk_offload_dir,
        )
        x = x[0][0].cpu().numpy()

        del latent
        vaemodule.decoder.cpu()
        torch.cuda.empty_cache()

        decode_time = time.time() - t_start
        print(f"    Decoded in {decode_time:.1f}s, output shape: {x.shape}")

        if binarize:
            x = (x > x.mean()).astype(bool)

        output_path = os.path.join(data_dir, f'{idx}.npy')
        np.save(output_path, x)
        print(f"    Saved to {output_path} ({'bool' if binarize else 'float'})")
        del x

        decode_timing.append({
            'index': idx,
            'decode_time': decode_time,
            'sample_time': sample_info.get('sample_time', 0),
        })

        # Copy porosity file to data dir
        if 'porosity_file' in sample_info:
            src = os.path.join(latent_dir, sample_info['porosity_file'])
            dst = os.path.join(data_dir, sample_info['porosity_file'])
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"    Copied {sample_info['porosity_file']} to data/")

    # Save timing
    total_decode = sum(s['decode_time'] for s in decode_timing)
    total_sample = sum(s['sample_time'] for s in decode_timing)

    timing = dict(config)
    timing['decode_timing'] = decode_timing
    timing['total_sample_time'] = total_sample
    timing['total_decode_time'] = total_decode
    timing['total_time'] = total_sample + total_decode

    timing_path = os.path.join(args.output_dir, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Total sample time: {total_sample:.1f}s")
    print(f"  Total decode time: {total_decode:.1f}s")
    print(f"  Total time:        {total_sample + total_decode:.1f}s")
    print(f"  Timing saved to {timing_path}")
    print("Decoding complete.")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.mode == 'latent':
        if args.stone is None:
            raise ValueError("--stone is required for latent generation mode")
        if is_distributed():
            run_latent_distributed(args)
        else:
            run_latent_single_gpu(args)
    elif args.mode == 'decode':
        run_decode(args)


if __name__ == '__main__':
    main()

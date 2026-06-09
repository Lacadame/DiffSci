#!/usr/bin/env python
"""
Spatially-parallel multi-GPU porosity field generator.

Uses spatial parallelism to distribute the latent diffusion sampling across
N GPUs along the depth (D) axis. Each GPU holds full model weights but only
processes 1/N of the spatial volume. After generating the latent volume,
gathers it on the first GPU and runs chunk_decode there.

This enables generating volumes too large for a single GPU (e.g. 2304^3).

Supports the same generation cases as 0004c:
  Case 1 (default): Field porosity with post-trained model
  Case 2: Null conditioning
  Case 3: Scalar porosity
  Case 4: Field porosity with original model
  Case 5: Field porosity with 129-trained model
  Case 7: Unconditional with provided checkpoint

Launch with:
    torchrun --nproc_per_node=N scripts/0004d-porosity-field-generator.py \
        --checkpoint /path/to/model.ckpt \
        --stone Estaillades \
        --output-dir ./generated_data/ \
        --volume-sizes 2304 \
        --volume-samples 1 \
        --nsteps 21

    For specific GPUs:
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        scripts/0004d-porosity-field-generator.py ...
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

# Add the aux directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
from model_loaders import load_autoencoder
from diffsci2.extra import chunk_decode_2, punetg_converters
from diffsci2.extra.matern_gaussian_process import MaternFieldSampler, PeriodicMaternFieldSampler
from diffsci2.distributed import SpatialContext, convert_to_spatial_parallel, scatter_along_dim, gather_along_dim


AVAILABLE_STONES = ['Bentheimer', 'Doddington', 'Estaillades', 'Ketton']

# Constants
LATENT_TO_PIXEL_FACTOR = 8  # pixel_size = latent_size * 8
MIN_LATENT_MULTIPLE = 16    # latent_size must be multiple of 16
DILATION_FACTOR = 1         # Working in latent space (no dilation needed)

# Base path for data
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Spatially-parallel multi-GPU volume generator'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (post-trained for case 1, ignored for cases 2-4 which use original)'
    )
    parser.add_argument(
        '--stone', type=str, required=True, choices=AVAILABLE_STONES,
        help='Stone type for GP parameters and original checkpoint'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for generated volumes'
    )
    parser.add_argument(
        '--generation-case', type=int, default=1, choices=[1, 2, 3, 4, 5, 7],
        help='Generation case: 1=field+post-trained (default), 2=null, 3=scalar, 4=field+original, 5=field+129-trained, 7=unconditional+provided'
    )
    parser.add_argument(
        '--coarse-n', type=int, default=32,
        help='Coarse grid size for GP sampling (default: 32)'
    )
    parser.add_argument(
        '--nsteps', type=int, default=21,
        help='Number of sampling steps (default: 21)'
    )
    parser.add_argument(
        '--guidance', type=float, default=1.0,
        help='Classifier-free guidance scale (default: 1.0, no guidance)'
    )
    parser.add_argument(
        '--volume-sizes', type=str, default='2304',
        help='Comma-separated list of volume sizes to generate (must be multiples of 128)'
    )
    parser.add_argument(
        '--volume-samples', type=str, default='1',
        help='Comma-separated list of number of samples per size'
    )
    parser.add_argument(
        '--save-porosity', action='store_true',
        help='Save the input porosity field alongside each volume'
    )
    parser.add_argument(
        '--periodic', action='store_true',
        help='Generate periodic (circular) volumes with periodic porosity conditioning'
    )
    parser.add_argument(
        '--no-binarize', action='store_true',
        help='Save raw float values instead of binarized (bool) volumes'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=40,
        help='Chunk size for VAE decoding (default: 40)'
    )
    parser.add_argument(
        '--disk-offload', action='store_true',
        help='Use disk-backed mmap buffers for chunk decode stage buffers. '
             'Required for very large volumes (e.g. 2304^3) where intermediate '
             'stage buffers (~2.85 TB for 64ch) exceed available CPU RAM.'
    )
    parser.add_argument(
        '--disk-offload-dir', type=str, default=None,
        help='Directory for mmap temp files (default: /tmp/chunk_decode_mmap)'
    )
    return parser.parse_args()


def parse_int_list(s):
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(',')]


def validate_volume_size(pixel_size, world_size):
    """Validate that pixel size corresponds to valid latent size divisible by world_size."""
    if pixel_size % (LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE) != 0:
        raise ValueError(
            f"Volume size {pixel_size} must be a multiple of "
            f"{LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE} (128). "
            f"Valid examples: 256, 384, 512, 640, 768, 896, 1024, 1152, ..."
        )
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    if latent_size % world_size != 0:
        raise ValueError(
            f"Latent size {latent_size} (from pixel size {pixel_size}) must be "
            f"divisible by world_size {world_size}. "
            f"latent_size / world_size = {latent_size / world_size}"
        )
    return latent_size


def log(rank, msg):
    """Print only from rank 0."""
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def init_distributed():
    """Initialize torch.distributed and return rank, world_size, device."""
    # 30-min timeout is generous for Phase 1 (sampling + gather).
    # Phase 2 (decode) runs after destroy_process_group, so no NCCL timeout applies.
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return rank, world_size, device


# ---------------------------------------------------------------------------
# GP porosity sampling (run on rank 0 only, then broadcast)
# ---------------------------------------------------------------------------

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
    flat_volume = porosity_volume.flatten()
    random_value = np.random.choice(flat_volume)
    return np.array([random_value], dtype=np.float32)


def create_porosity_sampler(stone, coarse_n=16, gpdata_paths=None):
    gpdata = load_gpdata(stone, gpdata_paths=gpdata_paths)
    sampler = MaternFieldSampler(
        mean_val=float(gpdata['mean_logit']),
        sigma_sq=float(gpdata['matern_sigma_sq']),
        nu=float(gpdata['matern_nu']),
        length_scale=float(gpdata['matern_length_scale'])
    )
    return sampler, gpdata


def create_periodic_porosity_sampler(stone, coarse_n=16, gpdata_paths=None):
    gpdata = load_gpdata(stone, gpdata_paths=gpdata_paths)
    sampler = PeriodicMaternFieldSampler(
        mean_val=float(gpdata['mean_logit']),
        sigma_sq=float(gpdata['matern_sigma_sq']),
        nu=float(gpdata['matern_nu']),
        length_scale=float(gpdata['matern_length_scale'])
    )
    return sampler, gpdata


def sample_porosity_field(sampler, latent_shape, coarse_n=16):
    import scipy.special
    L_x, L_y, L_z = latent_shape
    pixel_x = L_x * DILATION_FACTOR
    pixel_y = L_y * DILATION_FACTOR
    pixel_z = L_z * DILATION_FACTOR
    x_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_x - 0.5) * DILATION_FACTOR, coarse_n)
    y_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_y - 0.5) * DILATION_FACTOR, coarse_n)
    z_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_z - 0.5) * DILATION_FACTOR, coarse_n)
    sampler.initialize_field_from_grid(x_coarse, y_coarse, z_coarse)
    x_fine = np.linspace(0.5 * DILATION_FACTOR, (L_x - 0.5) * DILATION_FACTOR, L_x)
    y_fine = np.linspace(0.5 * DILATION_FACTOR, (L_y - 0.5) * DILATION_FACTOR, L_y)
    z_fine = np.linspace(0.5 * DILATION_FACTOR, (L_z - 0.5) * DILATION_FACTOR, L_z)
    logit_field = sampler.sample_grid_interpolated(1, x_fine, y_fine, z_fine)[0]
    porosity_field = scipy.special.expit(logit_field)
    return porosity_field.astype(np.float32)


def sample_periodic_porosity_field(sampler, shape, coarse_n):
    import scipy.special
    axes = [np.linspace(0, s, s) for s in shape]
    sampler.initialize_field_from_grid(*axes)
    field = sampler.sample_grid(1)[0]
    porosity = scipy.special.expit(field)
    return porosity.astype(np.float32)


def sample_new_porosity_field(sampler, pixel_size, coarse_n, periodic=False):
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    if periodic:
        return sample_periodic_porosity_field(sampler, [latent_size, latent_size, latent_size], coarse_n)
    else:
        return sample_porosity_field(sampler, [latent_size, latent_size, latent_size], coarse_n)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_weights_cpu(checkpoint_path):
    """Load model weights from checkpoint with map_location='cpu'."""
    loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['state_dict']
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


def load_models_distributed(checkpoint_path, device, periodic, unconditional, ctx):
    """Load flow model and autoencoder, convert model to spatial-parallel.

    Returns flowmodule (with spatial-parallel model) and vaemodule (CPU, for rank 0 decoding).
    """
    # Load flow model on CPU (map_location='cpu' handles checkpoints saved on any GPU)
    weights = _load_weights_cpu(checkpoint_path)
    if unconditional:
        flowmodel = diffsci2.nets.PUNetG(
            _punetg_config(cond_dropout=0.0), conditional_embedding=None)
    else:
        embedder = diffsci2.nets.ScalarEmbedder(dembed=64, key='porosity')
        flowmodel = diffsci2.nets.PUNetG(
            _punetg_config(cond_dropout=0.1), conditional_embedding=embedder)
    flowmodel.load_state_dict(weights)

    # Convert to circular convolutions for periodic generation BEFORE spatial-parallel
    # (spatial-parallel converter knows how to handle CircularConv3d)
    if periodic:
        flowmodel = punetg_converters.convert_conv_to_circular(flowmodel, [0, 1, 2], True)

    # Create SIModule without autoencoder (we decode separately on rank 0)
    flowmoduleconfig = diffsci2.models.SIModuleConfig.from_edm_sigma_space(
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        initial_norm=20.0,
        loss_formulation='denoiser'
    )
    flowmodule = diffsci2.models.SIModule(
        config=flowmoduleconfig,
        model=flowmodel,
        autoencoder=None  # Not needed; we use return_latents=True
    )
    flowmodule.to(device)

    # Convert the inner model to spatial-parallel (in-place to save memory)
    flowmodule.model = convert_to_spatial_parallel(flowmodule.model, ctx, inplace=True)
    flowmodule.to(device)  # re-move after conversion (converter creates new layers on CPU)

    # Load autoencoder only on rank 0 (only needed for Phase 2 decoding)
    vaemodule = load_autoencoder() if ctx.rank == 0 else None

    return flowmodule, vaemodule


# ---------------------------------------------------------------------------
# Spatial-parallel latent generation (Phase 1: all GPUs)
# ---------------------------------------------------------------------------

def generate_latent_spatial_parallel(
    flowmodule, y_np, pixel_size, nsteps, device, ctx, guidance=1.0
):
    """Generate a latent volume using spatial-parallel diffusion sampling.

    All ranks participate. Returns the full gathered latent on rank 0 (CPU),
    None on other ranks.

    Parameters
    ----------
    flowmodule : SIModule
        Flow model with spatial-parallel inner model.
    y_np : dict or None
        Conditioning dict with numpy arrays. For field porosity: {'porosity': ndarray [D,H,W]}.
        For scalar porosity: {'porosity': ndarray [1]}. None for unconditional.
    pixel_size : int
        Target volume size in pixels.
    nsteps : int
        Number of diffusion steps.
    device : torch.device
        This rank's device.
    ctx : SpatialContext
        Distributed context.
    guidance : float
        Classifier-free guidance scale.

    Returns
    -------
    latent : torch.Tensor or None
        Full latent [1, 4, D, D, D] on CPU (rank 0 only), None on other ranks.
    """
    rank = ctx.rank
    world_size = ctx.world_size
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    local_d = latent_size // world_size

    # --- Generate and scatter noise ---
    if rank == 0:
        noise_full = torch.randn(1, 4, latent_size, latent_size, latent_size, device=device)
    else:
        noise_full = torch.empty(1, 4, latent_size, latent_size, latent_size, device=device)
    dist.broadcast(noise_full, src=0)
    noise_local = scatter_along_dim(noise_full, dim=2, ctx=ctx)  # [1, 4, local_d, H, W]
    del noise_full

    # --- Prepare and scatter conditioning ---
    y_local = None
    if y_np is not None and 'porosity' in y_np:
        porosity_np = y_np['porosity']

        if porosity_np.ndim == 3:
            # Field porosity [D, H, W] -- scatter along D
            if rank == 0:
                porosity_full = torch.tensor(porosity_np, dtype=torch.float32, device=device)
            else:
                porosity_full = torch.empty(
                    latent_size, latent_size, latent_size,
                    dtype=torch.float32, device=device
                )
            dist.broadcast(porosity_full, src=0)
            porosity_local = scatter_along_dim(porosity_full, dim=0, ctx=ctx)  # [local_d, H, W]
            del porosity_full
            y_local = {'porosity': porosity_local}
        else:
            # Scalar porosity [1] -- same on all ranks, no scattering needed
            if rank == 0:
                scalar_val = torch.tensor(porosity_np, dtype=torch.float32, device=device)
            else:
                scalar_val = torch.empty(1, dtype=torch.float32, device=device)
            dist.broadcast(scalar_val, src=0)
            y_local = {'porosity': scalar_val}

    # --- Run spatially-parallel sampling ---
    log(rank, f"  Starting distributed sampling (latent {latent_size}^3, "
              f"{world_size} GPUs, local D={local_d})...")
    t_sample_start = time.time()

    x_latent_local = flowmodule.sample(
        1,
        shape=[4, local_d, latent_size, latent_size],
        y=y_local,
        nsteps=nsteps,
        is_latent_shape=True,
        return_latents=True,
        guidance=guidance,
        orig_noise=noise_local,
    )  # [1, 4, local_d, H, W]

    t_sample_end = time.time()
    log(rank, f"  Sampling done in {t_sample_end - t_sample_start:.1f}s")

    # --- Gather latent on rank 0, move to CPU to free GPU ---
    log(rank, "  Gathering latent...")
    x_latent_full = gather_along_dim(x_latent_local, dim=2, ctx=ctx)  # [1, 4, D, H, W]
    del x_latent_local
    torch.cuda.empty_cache()

    if rank == 0:
        x_latent_cpu = x_latent_full.cpu()
        del x_latent_full
        torch.cuda.empty_cache()
        return x_latent_cpu
    else:
        del x_latent_full
        torch.cuda.empty_cache()
        return None


# ---------------------------------------------------------------------------
# Latent decoding (Phase 2: rank 0 only, single-GPU, no distributed ops)
# ---------------------------------------------------------------------------

def decode_latent(vaemodule, x_latent, device, periodic=False, chunk_size=40,
                   use_disk_offload=False, disk_offload_dir=None):
    """Decode a latent tensor to pixel space using chunk_decode on a single GPU.

    Parameters
    ----------
    vaemodule : autoencoder module
        VAE autoencoder.
    x_latent : torch.Tensor
        Latent tensor [1, 4, D, H, W] (can be on CPU; will be moved as needed).
    device : torch.device
        GPU device to use for decoding.
    periodic : bool
        Whether to use periodic boundary conditions.
    chunk_size : int
        Chunk size for VAE decoding.
    use_disk_offload : bool
        If True, use memory-mapped files for intermediate stage buffers instead
        of CPU RAM. Required for very large volumes where stage buffers exceed
        available RAM (e.g. 2304^3 with 64 channels = ~2.85 TB).
    disk_offload_dir : str or None
        Directory for mmap temp files. If None, uses system temp directory.

    Returns
    -------
    x : ndarray
        Decoded volume [D*8, H*8, W*8].
    """
    latent_shape = list(x_latent.shape)
    pixel_size = x_latent.shape[2] * LATENT_TO_PIXEL_FACTOR
    print(f"  Decoding latent {latent_shape} -> pixel {pixel_size}^3 ...", flush=True)
    if use_disk_offload:
        offload_dir = disk_offload_dir or "/tmp/chunk_decode_mmap"
        print(f"  Using disk-backed mmap buffers (dir: {offload_dir})", flush=True)
    t_decode_start = time.time()

    periodicity = [True, True, True] if periodic else [False, False, False]
    chunk_decode_2.prepare_decoder_for_cached_decode(vaemodule.decoder)
    vaemodule.decoder.to(device)

    # Move latent to GPU for decoding
    x_latent = x_latent.to(device)

    x = chunk_decode_2.chunk_decode_3d(
        vaemodule.decoder,
        x_latent,
        [chunk_size, chunk_size, chunk_size],
        device=device,
        periodicity=periodicity,
        use_cached_norms=True,
        use_disk_offload=use_disk_offload,
        disk_offload_dir=disk_offload_dir,
    )
    x = x[0][0].cpu().numpy()

    del x_latent
    vaemodule.decoder.cpu()
    torch.cuda.empty_cache()

    t_decode_end = time.time()
    print(f"  Decode done in {t_decode_end - t_decode_start:.1f}s", flush=True)
    return x


# ---------------------------------------------------------------------------
# Case-specific latent generation wrappers (Phase 1: all GPUs)
# ---------------------------------------------------------------------------

def generate_latent_case1(flowmodule, sampler, pixel_size, coarse_n,
                          nsteps, device, ctx, guidance=1.0, periodic=False):
    """Case 1/4/5: Field porosity. Returns (latent_or_None, porosity_field_or_None)."""
    porosity_field = None
    if ctx.rank == 0:
        porosity_field = sample_new_porosity_field(sampler, pixel_size, coarse_n, periodic)
    y_np = {'porosity': porosity_field} if ctx.rank == 0 else {'porosity': None}
    if ctx.rank != 0:
        latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
        y_np = {'porosity': np.zeros((latent_size, latent_size, latent_size), dtype=np.float32)}
    latent = generate_latent_spatial_parallel(
        flowmodule, y_np, pixel_size, nsteps, device, ctx, guidance
    )
    return latent, porosity_field


def generate_latent_case2(flowmodule, pixel_size, nsteps, device, ctx, guidance=1.0):
    """Case 2/7: No conditioning. Returns (latent_or_None, None)."""
    latent = generate_latent_spatial_parallel(
        flowmodule, None, pixel_size, nsteps, device, ctx, guidance
    )
    return latent, None


def generate_latent_case3(flowmodule, porosity_volume, pixel_size, nsteps,
                          device, ctx, guidance=1.0):
    """Case 3: Scalar porosity. Returns (latent_or_None, scalar_porosity_or_None)."""
    scalar_porosity = None
    if ctx.rank == 0:
        scalar_porosity = sample_scalar_porosity(porosity_volume)
    if ctx.rank != 0:
        scalar_porosity = np.zeros(1, dtype=np.float32)
    y_np = {'porosity': scalar_porosity}
    latent = generate_latent_spatial_parallel(
        flowmodule, y_np, pixel_size, nsteps, device, ctx, guidance
    )
    return latent, scalar_porosity if ctx.rank == 0 else None


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

def get_case_description(case):
    descriptions = {
        1: "Field porosity with post-trained (field-trained) model",
        2: "Null conditioning (y=None)",
        3: "Scalar porosity (random from real data)",
        4: "Field porosity with original (scalar-trained) model",
        5: "Field porosity with 129-trained model (gpdata4-129)",
        7: "Unconditional with provided checkpoint",
    }
    return descriptions.get(case, "Unknown case")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Initialize distributed
    rank, world_size, device = init_distributed()
    log(rank, f"Initialized {world_size} GPUs for spatial parallelism")

    # Create spatial context
    ctx = SpatialContext(
        rank=rank,
        world_size=world_size,
        process_group=dist.group.WORLD,
        split_dim=2,  # D dimension in [B, C, D, H, W]
        periodic=args.periodic
    )

    # Parse volume sizes and samples
    volume_sizes = parse_int_list(args.volume_sizes)
    volume_samples = parse_int_list(args.volume_samples)

    if len(volume_sizes) != len(volume_samples):
        raise ValueError(
            f"volume-sizes ({len(volume_sizes)}) and volume-samples ({len(volume_samples)}) "
            f"must have the same length"
        )

    # Validate all volume sizes
    for size in volume_sizes:
        validate_volume_size(size, world_size)

    # Determine which checkpoint to use based on case
    generation_case = args.generation_case
    if generation_case in [1, 5, 7]:
        checkpoint_path = args.checkpoint
        log(rank, f"Using PROVIDED checkpoint for case {generation_case}: {checkpoint_path}")
    else:
        checkpoint_path = ORIGINAL_CHECKPOINTS[args.stone]
        log(rank, f"Using ORIGINAL checkpoint for case {generation_case}: {checkpoint_path}")

    # Create output directories (rank 0 only)
    if rank == 0:
        data_dir = os.path.join(args.output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
    else:
        data_dir = os.path.join(args.output_dir, 'data')

    # Timing data (rank 0 only)
    timing = {
        'checkpoint': checkpoint_path,
        'stone': args.stone,
        'world_size': world_size,
        'nsteps': args.nsteps,
        'coarse_n': args.coarse_n,
        'guidance': args.guidance,
        'periodic': args.periodic,
        'binarize': not args.no_binarize,
        'generation_case': generation_case,
        'case_description': get_case_description(generation_case),
        'volume_sizes': volume_sizes,
        'volume_samples': volume_samples,
        'samples': []
    }

    # Load models
    log(rank, f"\n=== Generation Case {generation_case}: {get_case_description(generation_case)} ===")
    log(rank, f"Loading models from {checkpoint_path}...")
    if args.periodic:
        log(rank, "  Using periodic (circular) convolutions")

    t_start = time.time()
    flowmodule, vaemodule = load_models_distributed(
        checkpoint_path,
        device,
        periodic=args.periodic,
        unconditional=(generation_case == 7),
        ctx=ctx
    )
    timing['model_load_time'] = time.time() - t_start
    log(rank, f"  Model load time: {timing['model_load_time']:.2f}s")

    # Setup for each case (rank 0 only needs samplers)
    sampler = None
    gpdata = None
    porosity_volume = None

    if generation_case in [1, 4, 5]:
        gpdata_paths = GPDATA_129_PATHS if generation_case == 5 else None
        if rank == 0:
            log(rank, f"Creating porosity sampler for {args.stone}...")
            if gpdata_paths is not None:
                log(rank, f"  Using 129 GP data from gpdata4-129")
            if args.periodic:
                sampler, gpdata = create_periodic_porosity_sampler(
                    args.stone, args.coarse_n, gpdata_paths=gpdata_paths)
                log(rank, "  Using periodic porosity sampler")
            else:
                sampler, gpdata = create_porosity_sampler(
                    args.stone, args.coarse_n, gpdata_paths=gpdata_paths)
            log(rank, f"  Mean logit: {gpdata['mean_logit']:.4f}")
            log(rank, f"  Matern params: sigma^2={gpdata['matern_sigma_sq']:.4f}, "
                       f"nu={gpdata['matern_nu']:.4f}, l={gpdata['matern_length_scale']:.4f}")

    if generation_case == 3 and rank == 0:
        log(rank, f"Loading real porosity volume for {args.stone}...")
        porosity_volume = load_porosity_volume(args.stone)
        log(rank, f"  Volume shape: {porosity_volume.shape}, "
                   f"range: [{porosity_volume.min():.4f}, {porosity_volume.max():.4f}]")

    # Synchronize before generation
    dist.barrier()

    # ===================================================================
    # PHASE 1: Distributed latent generation (all GPUs)
    # ===================================================================
    # Generate all latents using spatial-parallel sampling, save to temp
    # files. After this phase, ranks 1..N-1 exit and free their GPUs.
    # ===================================================================

    latent_jobs = []  # rank 0 only: list of dicts with paths + metadata

    for pixel_size, n_samples in zip(volume_sizes, volume_samples):
        if n_samples <= 0:
            continue

        latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
        local_d = latent_size // world_size
        log(rank, f"\nGenerating {n_samples} x {pixel_size}^3 latents "
                   f"(latent: {latent_size}^3, local D={local_d}, guidance={args.guidance})...")

        for i in range(n_samples):
            log(rank, f"  [{i+1}/{n_samples}] Sampling latent for {pixel_size}_{i}...")
            t_start = time.time()

            if generation_case in [1, 5]:
                latent, porosity_data = generate_latent_case1(
                    flowmodule, sampler, pixel_size, args.coarse_n,
                    args.nsteps, device, ctx, args.guidance, args.periodic
                )
            elif generation_case in [2, 7]:
                latent, porosity_data = generate_latent_case2(
                    flowmodule, pixel_size, args.nsteps, device, ctx, args.guidance
                )
            elif generation_case == 3:
                latent, porosity_data = generate_latent_case3(
                    flowmodule, porosity_volume, pixel_size, args.nsteps,
                    device, ctx, args.guidance
                )
            else:  # case 4
                latent, porosity_data = generate_latent_case1(
                    flowmodule, sampler, pixel_size, args.coarse_n,
                    args.nsteps, device, ctx, args.guidance, args.periodic
                )

            sample_time = time.time() - t_start

            # Rank 0: save latent to temp file to keep memory bounded
            if rank == 0:
                latent_path = os.path.join(data_dir, f'{pixel_size}_{i}.latent.pt')
                torch.save(latent, latent_path)
                del latent
                log(rank, f"    Saved temp latent to {latent_path} ({sample_time:.1f}s)")

                latent_jobs.append({
                    'pixel_size': pixel_size,
                    'index': i,
                    'latent_path': latent_path,
                    'porosity_data': porosity_data,
                    'sample_time': sample_time,
                })

            # Synchronize between samples (lightweight -- sampling is done)
            dist.barrier()

    # ===================================================================
    # Teardown distributed: free ranks 1..N-1
    # ===================================================================
    log(rank, f"\nDistributed sampling complete. Releasing {world_size - 1} GPUs...")
    del flowmodule
    torch.cuda.empty_cache()
    dist.destroy_process_group()

    if rank != 0:
        return  # Exit cleanly, GPU freed

    # ===================================================================
    # PHASE 2: Decode + save (rank 0 only, single GPU, no time limit)
    # ===================================================================
    print(f"\n=== Phase 2: Decoding {len(latent_jobs)} latent(s) on GPU 0 ===", flush=True)

    for job in latent_jobs:
        pixel_size = job['pixel_size']
        i = job['index']
        t_start = time.time()

        print(f"\n  [{i+1}] Decoding {pixel_size}_{i}...", flush=True)

        # Load temp latent from disk
        latent = torch.load(job['latent_path'], map_location='cpu', weights_only=True)

        x = decode_latent(
            vaemodule, latent, device,
            periodic=args.periodic, chunk_size=args.chunk_size,
            use_disk_offload=args.disk_offload,
            disk_offload_dir=args.disk_offload_dir,
        )
        del latent

        decode_time = time.time() - t_start
        total_time = job['sample_time'] + decode_time
        timing['samples'].append({
            'size': pixel_size, 'index': i,
            'sample_time': job['sample_time'],
            'decode_time': decode_time,
            'time': total_time,
        })

        if not args.no_binarize:
            x = (x > x.mean()).astype(bool)

        output_path = os.path.join(data_dir, f'{pixel_size}_{i}.npy')
        np.save(output_path, x)
        print(f"    Saved to {output_path} (sample={job['sample_time']:.1f}s, "
              f"decode={decode_time:.1f}s, "
              f"{'float' if args.no_binarize else 'bool'})", flush=True)
        del x

        porosity_data = job['porosity_data']
        if args.save_porosity and porosity_data is not None:
            if generation_case == 3:
                porosity_path = os.path.join(data_dir, f'{pixel_size}_{i}.scalarporosity.npy')
            else:
                porosity_path = os.path.join(data_dir, f'{pixel_size}_{i}.porosity.npy')
            np.save(porosity_path, porosity_data)
            print(f"    Saved porosity to {porosity_path}", flush=True)

        # Clean up temp latent file
        os.remove(job['latent_path'])

    # Save timing data
    for size in volume_sizes:
        times = [s['time'] for s in timing['samples'] if s['size'] == size]
        if times:
            timing[f'mean_time_{size}'] = float(np.mean(times))
            timing[f'std_time_{size}'] = float(np.std(times))
            timing[f'total_time_{size}'] = float(np.sum(times))
            timing[f'count_{size}'] = len(times)

    timing['total_generation_time'] = sum(s['time'] for s in timing['samples'])

    timing_path = os.path.join(args.output_dir, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)
    print(f"\nTiming data saved to {timing_path}", flush=True)

    print("\n=== Timing Summary ===", flush=True)
    for size in volume_sizes:
        if f'mean_time_{size}' in timing:
            print(f"  {size}^3: {timing[f'count_{size}']} samples, "
                  f"mean={timing[f'mean_time_{size}']:.2f}s, "
                  f"std={timing[f'std_time_{size}']:.2f}s, "
                  f"total={timing[f'total_time_{size}']:.2f}s", flush=True)
    print(f"  Total generation time: {timing['total_generation_time']:.2f}s", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()

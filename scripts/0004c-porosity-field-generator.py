#!/usr/bin/env python
"""
Multi-scale volume generator with different conditioning cases.

Supports seven generation cases:
  Case 1 (default): Field porosity with post-trained model - post-trained checkpoint, GP-sampled field
  Case 2: Null conditioning - original (scalar-trained) checkpoint, no conditioning
  Case 3: Scalar porosity - original checkpoint, random scalar porosity from real data
  Case 4: Field porosity with original model - original checkpoint, GP-sampled field (gpdata3c)
  Case 5: Field porosity with 129-trained model - 129 checkpoint, GP field from gpdata4-129
  Case 6: Field porosity with original model - original checkpoint, GP field from gpdata4-129
  Case 7: Unconditional with provided checkpoint - provided checkpoint, no conditioning
  Case 8: Field porosity with 257-trained model - 257 checkpoint, GP field from gpdata4-257

Usage:
    python 0004c-porosity-field-generator.py \
        --checkpoint /path/to/model.ckpt \
        --stone Estaillades \
        --output-dir ./generated_data/ \
        --volume-sizes 256,512,1024 \
        --volume-samples 64,8,1 \
        --generation-case 2 \
        --device cuda:0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Add the aux directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
import diffsci2.nets
from model_loaders import load_flow_model, load_model_from_module, load_autoencoder
from diffsci2.extra import chunk_decode_2, punetg_converters
from diffsci2.extra.matern_gaussian_process import MaternFieldSampler, PeriodicMaternFieldSampler
from diffsci2.nets.enhanced_conditioning import wrap_model_with_enhanced_conditioning


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

# Paths to GP analysis data for 257 models
GPDATA_257_DIR = os.path.join(NOTEBOOKPATH, 'data', 'gpdata4-257')
GPDATA_257_PATHS = {
    'Bentheimer': os.path.join(GPDATA_257_DIR, 'bentheimer', 'bentheimer_porosity_analysis.npz'),
    'Doddington': os.path.join(GPDATA_257_DIR, 'doddington', 'doddington_porosity_analysis.npz'),
    'Estaillades': os.path.join(GPDATA_257_DIR, 'estaillades', 'estaillades_porosity_analysis.npz'),
    'Ketton': os.path.join(GPDATA_257_DIR, 'ketton', 'ketton_porosity_analysis.npz'),
}

# Paths to real porosity volumes for 257 models
POROSITY_VOLUMES_257 = {
    'Bentheimer': os.path.join(GPDATA_257_DIR, 'bentheimer', 'bentheimer_porosity_field_full.npy'),
    'Doddington': os.path.join(GPDATA_257_DIR, 'doddington', 'doddington_porosity_field_full.npy'),
    'Estaillades': os.path.join(GPDATA_257_DIR, 'estaillades', 'estaillades_porosity_field_full.npy'),
    'Ketton': os.path.join(GPDATA_257_DIR, 'ketton', 'ketton_porosity_field_full.npy'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate multi-scale volumes with different conditioning cases'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (required for cases 1, 5, 7; ignored for cases 2-4, 6 which use original)'
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
        '--device', type=str, default='cuda:0',
        help='Device for computation'
    )
    parser.add_argument(
        '--generation-case', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help='Generation case: 1=field+post-trained (default), 2=null, 3=scalar, 4=field+original, 5=field+129-trained, 6=field+original+gpdata4-129, 7=unconditional+provided, 8=field+257-trained'
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
        '--volume-sizes', type=str, default='256,512,1024',
        help='Comma-separated list of volume sizes to generate (must be multiples of 128)'
    )
    parser.add_argument(
        '--volume-samples', type=str, default='64,8,1',
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
        '--nfields', type=int, default=None,
        help='Number of porosity fields to sample (enables variance test mode)'
    )
    parser.add_argument(
        '--nsamples-per-field', type=int, default=None,
        help='Number of volume samples per porosity field (enables variance test mode)'
    )
    # Enhanced conditioning options (for case 1 only)
    parser.add_argument(
        '--enhanced', action='store_true',
        help='Use enhanced conditioning architecture (FiLM + multi-scale) - case 1 only'
    )
    parser.add_argument(
        '--condition-embed-dim', type=int, default=64,
        help='Embedding dimension for enhanced conditioning (must match training)'
    )
    return parser.parse_args()


def parse_int_list(s):
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(',')]


def validate_volume_size(pixel_size):
    """Validate that pixel size corresponds to valid latent size."""
    if pixel_size % (LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE) != 0:
        raise ValueError(
            f"Volume size {pixel_size} must be a multiple of "
            f"{LATENT_TO_PIXEL_FACTOR * MIN_LATENT_MULTIPLE} (128). "
            f"Valid examples: 256, 384, 512, 640, 768, 896, 1024, 1152, ..."
        )
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    return latent_size


def load_gpdata(stone: str, gpdata_paths=None):
    """Load the GP analysis data for a stone type."""
    if gpdata_paths is None:
        gpdata_paths = GPDATA_PATHS
    path = gpdata_paths[stone]
    return np.load(path)


def load_porosity_volume(stone: str, porosity_volumes=None):
    """Load the real porosity volume for a stone type (used for scalar sampling)."""
    if porosity_volumes is None:
        porosity_volumes = POROSITY_VOLUMES
    path = porosity_volumes[stone]
    volume = np.load(path)
    # Crop edges (as done in notebook)
    volume = volume[128:-128, 128:-128, 128:-128]
    return volume


def sample_scalar_porosity(porosity_volume):
    """Sample a random scalar porosity value from the real porosity volume."""
    flat_volume = porosity_volume.flatten()
    random_value = np.random.choice(flat_volume)
    return np.array([random_value], dtype=np.float32)


def create_porosity_sampler(stone: str, coarse_n: int = 16, gpdata_paths=None):
    """Create a MaternFieldSampler initialized for the given stone."""
    gpdata = load_gpdata(stone, gpdata_paths=gpdata_paths)

    sampler = MaternFieldSampler(
        mean_val=float(gpdata['mean_logit']),
        sigma_sq=float(gpdata['matern_sigma_sq']),
        nu=float(gpdata['matern_nu']),
        length_scale=float(gpdata['matern_length_scale'])
    )

    return sampler, gpdata


def create_periodic_porosity_sampler(stone: str, coarse_n: int = 16, gpdata_paths=None):
    """Create a periodic Matern GP sampler for porosity field."""
    gpdata = load_gpdata(stone, gpdata_paths=gpdata_paths)

    sampler = PeriodicMaternFieldSampler(
        mean_val=float(gpdata['mean_logit']),
        sigma_sq=float(gpdata['matern_sigma_sq']),
        nu=float(gpdata['matern_nu']),
        length_scale=float(gpdata['matern_length_scale'])
    )

    return sampler, gpdata


def sample_porosity_field(sampler, latent_shape, coarse_n: int = 16):
    """
    Sample a porosity field for conditioning the latent diffusion model.

    The field is sampled in latent space coordinates (DILATION_FACTOR=1).
    GP parameters are fitted directly in latent space, so no dilation is needed.
    """
    import scipy.special

    L_x, L_y, L_z = latent_shape

    # Pixel-space dimensions (dilated coordinates)
    pixel_x = L_x * DILATION_FACTOR
    pixel_y = L_y * DILATION_FACTOR
    pixel_z = L_z * DILATION_FACTOR

    # Coarse grid for GP sampling (in pixel coordinates)
    x_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_x - 0.5) * DILATION_FACTOR, coarse_n)
    y_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_y - 0.5) * DILATION_FACTOR, coarse_n)
    z_coarse = np.linspace(0.5 * DILATION_FACTOR, (L_z - 0.5) * DILATION_FACTOR, coarse_n)

    # Initialize coarse grid
    sampler.initialize_field_from_grid(x_coarse, y_coarse, z_coarse)

    # Target (fine) grid at latent resolution
    x_fine = np.linspace(0.5 * DILATION_FACTOR, (L_x - 0.5) * DILATION_FACTOR, L_x)
    y_fine = np.linspace(0.5 * DILATION_FACTOR, (L_y - 0.5) * DILATION_FACTOR, L_y)
    z_fine = np.linspace(0.5 * DILATION_FACTOR, (L_z - 0.5) * DILATION_FACTOR, L_z)

    # Sample and interpolate to fine grid
    logit_field = sampler.sample_grid_interpolated(1, x_fine, y_fine, z_fine)[0]

    # Sigmoid to convert logit to porosity
    porosity_field = scipy.special.expit(logit_field)

    return porosity_field.astype(np.float32)


def sample_periodic_porosity_field(sampler, shape, coarse_n):
    """Sample a periodic porosity field at the given shape."""
    import scipy.special

    # Create coordinate grid matching the shape
    axes = [np.linspace(0, s, s) for s in shape]
    sampler.initialize_field_from_grid(*axes)

    # Sample the field
    field = sampler.sample_grid(1)[0]

    # Convert from logit to porosity using sigmoid
    porosity = scipy.special.expit(field)

    return porosity.astype(np.float32)


def load_unconditional_flow_model(checkpoint_path):
    """Load an unconditional flow model (no conditional embedding)."""
    weights = load_model_from_module(checkpoint_path)
    flowmodelconfig = diffsci2.nets.PUNetGConfig(
        input_channels=4,
        output_channels=4,
        dimension=3,
        model_channels=64,
        channel_expansion=[2, 4],
        number_resnet_downward_block=2,
        number_resnet_upward_block=2,
        number_resnet_attn_block=0,
        number_resnet_before_attn_block=3,
        number_resnet_after_attn_block=3,
        kernel_size=3,
        in_out_kernel_size=3,
        in_embedding=False,
        time_projection_scale=10.0,
        input_projection_scale=1.0,
        transition_scale_factor=2,
        transition_kernel_size=3,
        dropout=0.1,
        cond_dropout=0.0,
        first_resblock_norm="GroupLN",
        second_resblock_norm="GroupRMS",
        affine_norm=True,
        convolution_type="default",
        num_groups=1,
        attn_residual=False,
        attn_type="default",
        bias=True,
    )
    flowmodel = diffsci2.nets.PUNetG(
        flowmodelconfig,
        conditional_embedding=None,
    )
    flowmodel.load_state_dict(weights)
    return flowmodel


def load_models(checkpoint_path, device, periodic=False, enhanced=False, condition_embed_dim=64, unconditional=False):
    """Load flow model and autoencoder.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for computation
        periodic: Whether to use circular convolutions
        enhanced: Whether to use enhanced conditioning wrapper
        condition_embed_dim: Embedding dimension for enhanced conditioning
        unconditional: Whether to load as unconditional model (no conditional embedding)
    """
    # Load base flow model
    if unconditional:
        flowmodel = load_unconditional_flow_model(checkpoint_path)
    else:
        flowmodel = load_flow_model(checkpoint_path, custom_checkpoint_path=True)
    vaemodule = load_autoencoder()

    # Wrap with enhanced conditioning if requested
    if enhanced:
        print("  Using enhanced conditioning architecture")
        # Create the wrapper (for inference, cond_drop_p=0 since we don't need dropout)
        flowmodel = wrap_model_with_enhanced_conditioning(
            model=flowmodel,
            condition_embed_dim=condition_embed_dim,
            use_film=True,
            use_multiscale=True,
            use_gradient=True,
            cond_drop_p=0.0,  # No dropout at inference
        )

        # Load enhanced weights from checkpoint
        # The checkpoint contains the full wrapper state
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Lightning saves with 'model.' prefix
            model_state = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            flowmodel.load_state_dict(model_state, strict=False)
            print(f"  Loaded enhanced conditioning weights")

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
        autoencoder=vaemodule
    )

    # Convert to circular convolutions for periodic generation
    if periodic:
        if enhanced:
            # For enhanced model, convert the base_model inside the wrapper
            flowmodule.model.base_model = punetg_converters.convert_conv_to_circular(
                flowmodule.model.base_model, [0, 1, 2], True
            )
        else:
            flowmodule.model = punetg_converters.convert_conv_to_circular(
                flowmodule.model, [0, 1, 2], True
            )

    flowmodule.to(device)
    return flowmodule, vaemodule


def sample_new_porosity_field(sampler, pixel_size, coarse_n, periodic=False):
    """Sample a new porosity field for the given pixel size."""
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR
    if periodic:
        return sample_periodic_porosity_field(sampler, [latent_size, latent_size, latent_size], coarse_n)
    else:
        return sample_porosity_field(sampler, [latent_size, latent_size, latent_size], coarse_n)


def generate_volume_from_conditioning(flowmodule, vaemodule, y, pixel_size, nsteps, device, guidance=1.0, periodic=False):
    """
    Generate a single volume with given conditioning.

    Parameters
    ----------
    flowmodule : SIModule
        Flow model module.
    vaemodule : VAEModule
        Autoencoder module.
    y : dict or None
        Conditioning dict with 'porosity' key, or None for unconditional.
    pixel_size : int
        Target volume size in pixels (must be multiple of 128).
    nsteps : int
        Number of diffusion steps.
    device : str
        Device for computation.
    guidance : float
        Classifier-free guidance scale.
    periodic : bool
        Whether to use periodic (circular) generation.

    Returns
    -------
    volume : ndarray
        Generated volume of shape (pixel_size, pixel_size, pixel_size).
    """
    latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR

    use_chunk_decode = (pixel_size > 256)
    periodicity = [True, True, True] if periodic else [False, False, False]

    if use_chunk_decode:
        x_latent = flowmodule.sample(
            1, shape=[4, latent_size, latent_size, latent_size],
            y=y, nsteps=nsteps,
            is_latent_shape=True, return_latents=True, guidance=guidance
        )

        chunk_decode_2.prepare_decoder_for_cached_decode(vaemodule.decoder)
        vaemodule.decoder.to(device)

        chunk_size = [40, 40, 40]

        x = chunk_decode_2.chunk_decode_3d(
            vaemodule.decoder,
            x_latent,
            chunk_size,
            device=device,
            periodicity=periodicity,
            use_cached_norms=True
        )
        x = x[0][0].cpu().numpy()
    else:
        x = flowmodule.sample(1, shape=[1, pixel_size, pixel_size, pixel_size], y=y, nsteps=nsteps, guidance=guidance)
        x = x[0][0].cpu().numpy()

    torch.cuda.empty_cache()
    return x


def generate_volume_case1(flowmodule, vaemodule, sampler, pixel_size, coarse_n, nsteps, device, guidance=1.0, periodic=False):
    """Case 1: Field porosity with post-trained model - GP-sampled field."""
    porosity_field = sample_new_porosity_field(sampler, pixel_size, coarse_n, periodic)
    porosity_tensor = torch.tensor(porosity_field, dtype=torch.float32)
    y = {'porosity': porosity_tensor}
    x = generate_volume_from_conditioning(flowmodule, vaemodule, y, pixel_size, nsteps, device, guidance, periodic)
    return x, porosity_field


def generate_volume_case2(flowmodule, vaemodule, pixel_size, nsteps, device, guidance=1.0, periodic=False):
    """Case 2: Null conditioning - no porosity input."""
    y = None
    x = generate_volume_from_conditioning(flowmodule, vaemodule, y, pixel_size, nsteps, device, guidance, periodic)
    return x, None


def generate_volume_case3(flowmodule, vaemodule, porosity_volume, pixel_size, nsteps, device, guidance=1.0, periodic=False):
    """Case 3: Scalar porosity - random value from real data."""
    scalar_porosity = sample_scalar_porosity(porosity_volume)
    porosity_tensor = torch.tensor(scalar_porosity, dtype=torch.float32)
    y = {'porosity': porosity_tensor}
    x = generate_volume_from_conditioning(flowmodule, vaemodule, y, pixel_size, nsteps, device, guidance, periodic)
    return x, scalar_porosity


def generate_volume_case4(flowmodule, vaemodule, sampler, pixel_size, coarse_n, nsteps, device, guidance=1.0, periodic=False):
    """Case 4: Field porosity with original model - GP-sampled field."""
    porosity_field = sample_new_porosity_field(sampler, pixel_size, coarse_n, periodic)
    porosity_tensor = torch.tensor(porosity_field, dtype=torch.float32)
    y = {'porosity': porosity_tensor}
    x = generate_volume_from_conditioning(flowmodule, vaemodule, y, pixel_size, nsteps, device, guidance, periodic)
    return x, porosity_field


def get_case_description(case):
    """Return description string for a generation case."""
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


def main():
    args = parse_args()

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
        validate_volume_size(size)

    # Determine which checkpoint to use based on case
    generation_case = args.generation_case
    if generation_case in [1, 5, 7, 8]:
        # Use provided checkpoint for case 1 (post-trained), case 5 (129-trained), case 7 (unconditional), case 8 (257-trained)
        if args.checkpoint is None:
            raise ValueError(f"--checkpoint is required for generation case {generation_case}")
        checkpoint_path = args.checkpoint
        print(f"Using PROVIDED checkpoint for case {generation_case}: {checkpoint_path}")
    else:
        # Use original (scalar-trained) checkpoint for cases 2, 3, 4
        checkpoint_path = ORIGINAL_CHECKPOINTS[args.stone]
        print(f"Using ORIGINAL checkpoint for case {generation_case}: {checkpoint_path}")

    # Create output directories
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Timing data
    timing = {
        'checkpoint': checkpoint_path,
        'stone': args.stone,
        'device': args.device,
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
    print(f"\n=== Generation Case {generation_case}: {get_case_description(generation_case)} ===")
    print(f"Loading models from {checkpoint_path}...")
    if args.periodic:
        print("  Using periodic (circular) convolutions")
    # Enhanced conditioning only makes sense for case 1
    use_enhanced = args.enhanced and generation_case == 1
    if use_enhanced:
        print("  Using enhanced conditioning (FiLM + multi-scale)")

    t_start = time.time()
    flowmodule, vaemodule = load_models(
        checkpoint_path,
        args.device,
        periodic=args.periodic,
        enhanced=use_enhanced,
        condition_embed_dim=args.condition_embed_dim,
        unconditional=(generation_case == 7)
    )
    timing['model_load_time'] = time.time() - t_start
    timing['enhanced'] = use_enhanced
    print(f"  Model load time: {timing['model_load_time']:.2f}s")

    # Setup for each case
    sampler = None
    gpdata = None
    porosity_volume = None

    if generation_case in [1, 4, 5, 6, 8]:
        # Need porosity sampler
        # Cases 5, 6 use GP data fitted on 129-resolution data; case 8 uses 257-resolution data
        if generation_case in [5, 6]:
            gpdata_paths = GPDATA_129_PATHS
        elif generation_case == 8:
            gpdata_paths = GPDATA_257_PATHS
        else:
            gpdata_paths = None
        print(f"Creating porosity sampler for {args.stone}...")
        if gpdata_paths is not None and generation_case == 8:
            print(f"  Using 257 GP data from gpdata4-257")
        elif gpdata_paths is not None:
            print(f"  Using 129 GP data from gpdata4-129")
        if args.periodic:
            sampler, gpdata = create_periodic_porosity_sampler(args.stone, args.coarse_n, gpdata_paths=gpdata_paths)
            print("  Using periodic porosity sampler")
        else:
            sampler, gpdata = create_porosity_sampler(args.stone, args.coarse_n, gpdata_paths=gpdata_paths)
        print(f"  Mean logit: {gpdata['mean_logit']:.4f}")
        print(f"  Matern params: sigma^2={gpdata['matern_sigma_sq']:.4f}, "
              f"nu={gpdata['matern_nu']:.4f}, l={gpdata['matern_length_scale']:.4f}")

    if generation_case == 3:
        # Need real porosity volume for scalar sampling
        print(f"Loading real porosity volume for {args.stone}...")
        porosity_volume = load_porosity_volume(args.stone)
        print(f"  Volume shape: {porosity_volume.shape}, range: [{porosity_volume.min():.4f}, {porosity_volume.max():.4f}]")

    # Determine mode: variance test or standard
    variance_test = args.nfields is not None and args.nsamples_per_field is not None

    # Generate volumes for each size
    for pixel_size, n_samples in zip(volume_sizes, volume_samples):
        if n_samples <= 0:
            continue

        latent_size = pixel_size // LATENT_TO_PIXEL_FACTOR

        if variance_test and generation_case in [1, 4, 5, 6, 8]:
            # Variance test mode (only for field-based cases)
            nfields = args.nfields
            nsamples_per_field = args.nsamples_per_field
            total = nfields * nsamples_per_field
            print(f"\nVariance test: {nfields} fields x {nsamples_per_field} samples/field = {total} total")
            print(f"  {pixel_size}^3 (latent: {latent_size}^3, guidance={args.guidance})")

            for f in range(nfields):
                # Sample one field for this batch
                porosity_field = sample_new_porosity_field(
                    sampler, pixel_size, args.coarse_n, periodic=args.periodic
                )
                print(f"\n  Field {f}: porosity range [{porosity_field.min():.4f}, {porosity_field.max():.4f}]")

                if args.save_porosity:
                    porosity_path = os.path.join(data_dir, f'{pixel_size}_field_{f}.porosity.npy')
                    np.save(porosity_path, porosity_field)
                    print(f"    Saved porosity to {porosity_path}")

                for s in range(nsamples_per_field):
                    print(f"    Generating {pixel_size}_{s}_{f}...")
                    t_start = time.time()

                    porosity_tensor = torch.tensor(porosity_field, dtype=torch.float32)
                    y = {'porosity': porosity_tensor}
                    x = generate_volume_from_conditioning(
                        flowmodule, vaemodule, y,
                        pixel_size, args.nsteps, args.device, args.guidance,
                        periodic=args.periodic
                    )

                    elapsed = time.time() - t_start
                    timing['samples'].append({'size': pixel_size, 'field': f, 'sample': s, 'time': elapsed})

                    if not args.no_binarize:
                        x = (x > x.mean()).astype(bool)

                    output_path = os.path.join(data_dir, f'{pixel_size}_{s}_{f}.npy')
                    np.save(output_path, x)
                    print(f"      Saved to {output_path} ({elapsed:.2f}s, {'float' if args.no_binarize else 'bool'})")

        else:
            # Standard mode: each sample independent
            print(f"\nGenerating {n_samples} x {pixel_size}^3 samples (latent: {latent_size}^3, guidance={args.guidance})...")

            for i in range(n_samples):
                print(f"  Generating {pixel_size}_{i}...")
                t_start = time.time()

                # Generate based on case
                if generation_case in [1, 5, 8]:
                    x, porosity_data = generate_volume_case1(
                        flowmodule, vaemodule, sampler, pixel_size, args.coarse_n, args.nsteps, args.device, args.guidance,
                        periodic=args.periodic
                    )
                elif generation_case in [2, 7]:
                    x, porosity_data = generate_volume_case2(
                        flowmodule, vaemodule, pixel_size, args.nsteps, args.device, args.guidance,
                        periodic=args.periodic
                    )
                elif generation_case == 3:
                    x, porosity_data = generate_volume_case3(
                        flowmodule, vaemodule, porosity_volume, pixel_size, args.nsteps, args.device, args.guidance,
                        periodic=args.periodic
                    )
                else:  # case 4, 6
                    x, porosity_data = generate_volume_case4(
                        flowmodule, vaemodule, sampler, pixel_size, args.coarse_n, args.nsteps, args.device, args.guidance,
                        periodic=args.periodic
                    )

                elapsed = time.time() - t_start
                timing['samples'].append({'size': pixel_size, 'index': i, 'time': elapsed})

                if not args.no_binarize:
                    x = (x > x.mean()).astype(bool)

                output_path = os.path.join(data_dir, f'{pixel_size}_{i}.npy')
                np.save(output_path, x)
                print(f"    Saved to {output_path} ({elapsed:.2f}s, {'float' if args.no_binarize else 'bool'})")

                if args.save_porosity and porosity_data is not None:
                    # Use different extension for scalar vs field porosity
                    if generation_case == 3:
                        porosity_path = os.path.join(data_dir, f'{pixel_size}_{i}.scalarporosity.npy')
                    else:
                        porosity_path = os.path.join(data_dir, f'{pixel_size}_{i}.porosity.npy')
                    np.save(porosity_path, porosity_data)
                    print(f"    Saved porosity to {porosity_path}")

    # Compute summary statistics per size
    for size in volume_sizes:
        times = [s['time'] for s in timing['samples'] if s['size'] == size]
        if times:
            timing[f'mean_time_{size}'] = float(np.mean(times))
            timing[f'std_time_{size}'] = float(np.std(times))
            timing[f'total_time_{size}'] = float(np.sum(times))
            timing[f'count_{size}'] = len(times)

    timing['total_generation_time'] = sum(s['time'] for s in timing['samples'])

    # Save timing data
    timing_path = os.path.join(args.output_dir, 'timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing, f, indent=2)
    print(f"\nTiming data saved to {timing_path}")

    # Print summary
    print("\n=== Timing Summary ===")
    for size in volume_sizes:
        if f'mean_time_{size}' in timing:
            print(f"  {size}^3: {timing[f'count_{size}']} samples, "
                  f"mean={timing[f'mean_time_{size}']:.2f}s, "
                  f"std={timing[f'std_time_{size}']:.2f}s, "
                  f"total={timing[f'total_time_{size}']:.2f}s")
    print(f"  Total generation time: {timing['total_generation_time']:.2f}s")

    print("\nDone!")


if __name__ == '__main__':
    main()

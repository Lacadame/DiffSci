#!/usr/bin/env python
"""
Porosity Field Estimator for 3D Binary Volumes.

This script calculates the mean porosity field from a 3D binary volume,
computes the logit-transformed two-point correlation function, and fits
a Matérn Gaussian Process model to capture the spatial correlation structure.

The output can be used to generate new porosity field realizations for
generative models.

Usage:
    # Basic usage (saves only analysis data)
    python scripts/0002-porosity-field-estimator.py \
        --data-path /path/to/volume.raw \
        --output-dir ./output

    # Save porosity field as well
    python scripts/0002-porosity-field-estimator.py \
        --data-path /path/to/volume.raw \
        --output-dir ./output \
        --save-field

    # Custom kernel size and GPU
    python scripts/0002-porosity-field-estimator.py \
        --data-path /path/to/volume.raw \
        --output-dir ./output \
        --kernel-size 128 \
        --device cuda:0
"""
import argparse
import os

import numpy as np
import scipy.signal
import scipy.ndimage
import torch

from diffsci2.extra import two_point_correlation, matern_gaussian_process
from scipy.optimize import curve_fit

# Stone paths
DATA_DIR = '/home/ubuntu/repos/PoreGen/saveddata/raw/imperial_college/'
VOLUME_PATHS = {
    'Bentheimer': DATA_DIR + 'Bentheimer_1000c_3p0035um.raw',
    'Doddington': DATA_DIR + 'Doddington_1000c_2p6929um.raw',
    'Estaillades': DATA_DIR + 'Estaillades_1000c_3p31136um.raw',
    'Ketton': DATA_DIR + 'Ketton_1000c_3p00006um.raw',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate porosity field and fit Matérn GP parameters'
    )
    parser.add_argument(
        '--stone', type=str, default=None,
        choices=['Bentheimer', 'Doddington', 'Estaillades', 'Ketton'],
        help='Stone type (determines data path)'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to .raw binary volume file'
    )
    parser.add_argument(
        '--volume-shape', type=int, nargs=3, default=[1000, 1000, 1000],
        help='Shape of the volume (D H W)'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to save output files'
    )
    parser.add_argument(
        '--output-prefix', type=str, default=None,
        help='Prefix for output files (default: derived from input filename)'
    )
    parser.add_argument(
        '--kernel-size', type=int, default=256,
        help='Size of the averaging kernel for porosity field calculation'
    )
    parser.add_argument(
        '--save-field', action='store_true',
        help='Save the porosity field volume (can be large)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for TPC calculation (e.g., cuda:0, cpu)'
    )
    parser.add_argument(
        '--tpc-bins', type=int, default=50,
        help='Number of bins for radial TPC calculation'
    )
    parser.add_argument(
        '--voxel-size', type=float, default=1.0,
        help='Physical voxel size for distance scaling'
    )
    parser.add_argument(
        '--use-full-porosity-field', action='store_true',
        help='Use full "same" convolution for porosity field (default: use cropped "valid")'
    )
    parser.add_argument(
        '--fixed-nu', type=float, default=None,
        help='Fix nu and only regress sigma_sq and length_scale (e.g., 1.5)'
    )
    parser.add_argument(
        '--latent-downsample', type=int, default=None,
        help='Downsample factor for latent space (e.g., 8)'
    )
    parser.add_argument(
        '--porosity-dir', type=str, default=None,
        help='Directory with pre-computed porosity fields (porosity_field.npy and porosity_field_full.npy)'
    )
    return parser.parse_args()


def create_averaging_kernel(kernel_size):
    """Create a 3D box averaging kernel."""
    return np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)


def calculate_porosity_field(data, kernel_size):
    """
    Calculate the local porosity field using FFT convolution (valid mode, cropped).

    Parameters
    ----------
    data : ndarray
        Binary volume (0=pore, 1=solid or vice versa)
    kernel_size : int
        Size of the averaging kernel

    Returns
    -------
    porosity_field : ndarray
        Local mean porosity field
    """
    kernel = create_averaging_kernel(kernel_size)
    # 1 - data converts from solid=1 to pore=1 representation
    porosity_field = scipy.signal.fftconvolve(
        1 - data.astype(np.float32),
        kernel,
        mode='valid'
    )
    return porosity_field


def calculate_porosity_field_full(data, kernel_size):
    """
    Calculate the local porosity field using FFT convolution (same mode, full size).

    Uses proper normalization that accounts for boundary effects: at each position,
    divides by the number of actual (non-padded) voxels that contributed.

    Parameters
    ----------
    data : ndarray
        Binary volume (0=pore, 1=solid or vice versa)
    kernel_size : int
        Size of the averaging kernel

    Returns
    -------
    porosity_field : ndarray
        Local mean porosity field (same shape as input)
    """
    # Use a sum kernel (not normalized)
    kernel = np.ones((kernel_size, kernel_size, kernel_size))

    # 1 - data converts from solid=1 to pore=1 representation
    pore_data = 1 - data.astype(np.float32)

    # Convolve to get sum of pore values
    pore_sum = scipy.signal.fftconvolve(pore_data, kernel, mode='same')

    # Convolve ones to get the count of contributing voxels at each position
    # This accounts for boundary effects
    norm_conv_x = scipy.signal.fftconvolve(
        np.ones((pore_data.shape[0])),
        np.ones((kernel_size)),
        mode='same'
    )
    norm_conv_y = scipy.signal.fftconvolve(
        np.ones((pore_data.shape[1])),
        np.ones((kernel_size)),
        mode='same'
    )
    norm_conv_z = scipy.signal.fftconvolve(
        np.ones((pore_data.shape[2])),
        np.ones((kernel_size)),
        mode='same'
    )

    normalization = (
        norm_conv_x[None, None, :] *
        norm_conv_y[None, :, None] *
        norm_conv_z[:, None, None]
    )

    # Divide to get mean porosity
    porosity_field = pore_sum / normalization

    return porosity_field


def average_volume(volume, kernel_size):
    """Average pool a volume using torch AvgPool3d."""
    import torch
    avgpool = torch.nn.AvgPool3d(kernel_size)
    volume = torch.from_numpy(volume).float()
    volume = volume.unsqueeze(0).unsqueeze(0)
    avg_volume = avgpool(volume)
    avg_volume = avg_volume.squeeze().numpy()
    return avg_volume


def calculate_logit_correlation(porosity_field, device, bins, voxel_size):
    """
    Calculate the two-point correlation function of the logit-transformed field.

    Parameters
    ----------
    porosity_field : ndarray
        Local porosity field
    device : str
        Device for computation
    bins : int
        Number of radial bins
    voxel_size : float
        Physical voxel size

    Returns
    -------
    dict with keys:
        - r: radial distances
        - correlation: TPC values
        - mean_logit: mean of logit-transformed field
        - variance_logit: variance of logit-transformed field (empirical σ²)
    """
    porosity_tensor = torch.tensor(porosity_field).float()

    # Logit transform
    porosity_logit = torch.logit(porosity_tensor)
    mean_logit = porosity_logit.mean().item()

    # Center the field
    centered_logit = porosity_logit - mean_logit

    # Compute empirical variance (this is our σ²)
    variance_logit = centered_logit.var().item()

    # Compute TPC
    tpc = two_point_correlation.tpcf_fft(
        centered_logit.to(device),
        bins=bins,
        voxel_size=voxel_size
    )

    return {
        'r': tpc.r,
        'correlation': tpc.correlation,
        'mean_logit': mean_logit,
        'variance_logit': variance_logit
    }


def fit_matern_to_correlation_fixed_nu(r, correlation, nu_fixed, sigma_sq_empirical):
    """
    Fit Matérn with fixed nu and fixed sigma_sq, only regress length_scale.

    Parameters
    ----------
    r : array
        Radial distances
    correlation : array
        Correlation values
    nu_fixed : float
        Fixed nu parameter
    sigma_sq_empirical : float
        Empirical variance (fixed σ²)
    """
    # Check if nu is a classical value (0.5, 1.5, 2.5)
    is_classical = np.any(np.abs(nu_fixed - np.array([0.5, 1.5, 2.5])) < 1e-10)

    # Select appropriate covariance function
    if np.abs(nu_fixed - 0.5) < 1e-10:
        def cov_func(r, length_scale):
            return matern_gaussian_process.matern_covariance_half(r, sigma_sq_empirical, length_scale)
    elif np.abs(nu_fixed - 1.5) < 1e-10:
        def cov_func(r, length_scale):
            return matern_gaussian_process.matern_covariance_three_half(r, sigma_sq_empirical, length_scale)
    elif np.abs(nu_fixed - 2.5) < 1e-10:
        def cov_func(r, length_scale):
            return matern_gaussian_process.matern_covariance_five_half(r, sigma_sq_empirical, length_scale)
    else:
        # General case
        def cov_func(r, length_scale):
            return matern_gaussian_process.matern_covariance(r, sigma_sq_empirical, nu_fixed, length_scale)

    # Exclude r=0
    valid = np.isfinite(correlation) & (r > 1e-10)
    r_clean = r[valid]
    c_clean = correlation[valid]

    # Initial guess for length_scale
    drop_idx = np.abs(c_clean - sigma_sq_empirical*0.36).argmin()
    p0_l = r_clean[drop_idx] if drop_idx < len(r_clean) else r_clean[-1]/2
    if p0_l == 0:
        p0_l = 1.0

    try:
        popt, pcov = curve_fit(
            cov_func, r_clean, c_clean,
            p0=[p0_l],
            bounds=([1e-6], [np.inf]),
            maxfev=5000
        )
        return {
            'sigma_sq': sigma_sq_empirical,
            'nu': nu_fixed,
            'length_scale': popt[0],
            'popt': popt,
            'pcov': pcov,
            'fit_success': True
        }
    except (RuntimeError, ValueError) as e:
        print(f"  Fitting failed: {e}")
        return {
            'sigma_sq': sigma_sq_empirical,
            'nu': nu_fixed,
            'length_scale': np.nan,
            'popt': None,
            'pcov': None,
            'fit_success': False
        }


def fit_matern_to_correlation(r, correlation, sigma_sq_empirical, fixed_nu=None):
    """
    Fit Matérn kernel parameters to correlation data.

    Parameters
    ----------
    r : array
        Radial distances
    correlation : array
        Correlation values
    sigma_sq_empirical : float
        Empirical variance (fixed σ²)
    fixed_nu : float or None
        If provided, fix nu and only fit length_scale
    """
    if fixed_nu is not None:
        return fit_matern_to_correlation_fixed_nu(r, correlation, fixed_nu, sigma_sq_empirical)

    # Free nu case - fit only nu and length_scale with fixed sigma_sq
    def matern_fixed_sigma(r, nu, length_scale):
        return matern_gaussian_process.matern_covariance(r, sigma_sq_empirical, nu, length_scale)

    # Exclude r=0
    valid = np.isfinite(correlation) & (r > 1e-10)
    r_clean = r[valid]
    c_clean = correlation[valid]

    # Initial guesses
    p0_nu = 1.5
    drop_idx = np.abs(c_clean - sigma_sq_empirical*0.36).argmin()
    p0_l = r_clean[drop_idx] if drop_idx < len(r_clean) else r_clean[-1]/2
    if p0_l == 0:
        p0_l = 1.0

    try:
        popt, pcov = curve_fit(
            matern_fixed_sigma,
            r_clean,
            c_clean,
            p0=[p0_nu, p0_l],
            bounds=([0.1, 1e-6], [30.0, np.inf]),
            maxfev=5000
        )
        return {
            'sigma_sq': sigma_sq_empirical,
            'nu': popt[0],
            'length_scale': popt[1],
            'popt': popt,
            'pcov': pcov,
            'fit_success': True
        }
    except (RuntimeError, ValueError) as e:
        print(f"  Fitting failed: {e}")
        return {
            'sigma_sq': sigma_sq_empirical,
            'nu': np.nan,
            'length_scale': np.nan,
            'popt': None,
            'pcov': None,
            'fit_success': False
        }


def main():
    args = parse_args()

    # Resolve data path from stone or explicit path
    if args.stone:
        data_path = VOLUME_PATHS[args.stone]
        stone_name = args.stone.lower()
    elif args.data_path:
        data_path = args.data_path
        stone_name = None
    else:
        raise ValueError("Must specify --stone or --data-path")

    # Determine output prefix
    if args.output_prefix is None:
        output_prefix = stone_name if stone_name else os.path.splitext(os.path.basename(data_path))[0]
    else:
        output_prefix = args.output_prefix

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading volume from {data_path}")
    data = np.fromfile(data_path, dtype=np.uint8).reshape(args.volume_shape)
    print(f"Volume shape: {data.shape}")

    # Check if we should load pre-computed porosity fields
    if args.porosity_dir:
        print(f"\nLoading pre-computed porosity fields from {args.porosity_dir}")
        # Find the porosity field files
        import glob
        full_files = glob.glob(os.path.join(args.porosity_dir, "*_porosity_field_full.npy"))
        field_files = glob.glob(os.path.join(args.porosity_dir, "*_porosity_field.npy"))
        # Filter out the _full files from field_files
        field_files = [f for f in field_files if not f.endswith("_field_full.npy")]

        if full_files:
            porosity_field_full = np.load(full_files[0])
            print(f"Loaded full field: {full_files[0]}")
            print(f"  Shape: {porosity_field_full.shape}")
        else:
            porosity_field_full = None

        if field_files:
            porosity_field = np.load(field_files[0])
            print(f"Loaded cropped field: {field_files[0]}")
            print(f"  Shape: {porosity_field.shape}")
            print(f"  Range: [{porosity_field.min():.4f}, {porosity_field.max():.4f}]")
            print(f"  Mean: {porosity_field.mean():.4f}")
        else:
            raise ValueError(f"No porosity field found in {args.porosity_dir}")
    else:
        # Calculate porosity field at full resolution
        print(f"\nCalculating porosity field with kernel size {args.kernel_size}...")
        if args.use_full_porosity_field:
            print("  Using full 'same' convolution with boundary normalization...")
            porosity_field_full = calculate_porosity_field_full(data, args.kernel_size)
            print(f"  Full porosity field shape: {porosity_field_full.shape}")
            print(f"  Full porosity range: [{porosity_field_full.min():.4f}, {porosity_field_full.max():.4f}]")
            print(f"  Full mean porosity: {porosity_field_full.mean():.4f}")

            # Crop to valid region for GP calculation
            half_k = args.kernel_size // 2
            porosity_field = porosity_field_full[
                half_k:-half_k,
                half_k:-half_k,
                half_k:-half_k
            ]
            print(f"  Cropped porosity field shape (for GP): {porosity_field.shape}")
        else:
            porosity_field_full = None
            porosity_field = calculate_porosity_field(data, args.kernel_size)

    # Now downsample the porosity field if latent mode
    if args.latent_downsample:
        print(f"\nLatent mode: downsampling porosity field by {args.latent_downsample}")
        print(f"  Before: {porosity_field.shape}")
        porosity_field = average_volume(porosity_field, args.latent_downsample)
        print(f"  After: {porosity_field.shape}")
        # In latent space, voxel_size = 1 (latent coordinates)
        # The length scale will be in latent units
        tpc_voxel_size = 1.0
        print(f"  Using voxel_size=1.0 for TPC (latent space coordinates)")
    else:
        tpc_voxel_size = args.voxel_size

    if not args.porosity_dir:
        print(f"\nPorosity field shape: {porosity_field.shape}")
        print(f"Porosity range: [{porosity_field.min():.4f}, {porosity_field.max():.4f}]")
        print(f"Mean porosity: {porosity_field.mean():.4f}")

    # Optionally save porosity field
    if args.save_field:
        if args.use_full_porosity_field:
            # Save the full field
            field_path = os.path.join(args.output_dir, f'{output_prefix}_porosity_field_full.npy')
            print(f"Saving full porosity field to {field_path}")
            np.save(field_path, porosity_field_full)
        # Always save the cropped/valid field
        field_path = os.path.join(args.output_dir, f'{output_prefix}_porosity_field.npy')
        print(f"Saving porosity field to {field_path}")
        np.save(field_path, porosity_field)

    # Calculate logit correlation
    print(f"Calculating logit-transformed TPC on {args.device}...")
    corr_data = calculate_logit_correlation(
        porosity_field,
        device=args.device,
        bins=args.tpc_bins,
        voxel_size=tpc_voxel_size
    )
    print(f"Mean logit: {corr_data['mean_logit']:.4f}")
    print(f"Empirical variance (σ²): {corr_data['variance_logit']:.4f}")

    # Fit Matérn parameters (with fixed empirical σ²)
    if args.fixed_nu:
        print(f"\nFitting Matérn with FIXED nu={args.fixed_nu} and FIXED σ²={corr_data['variance_logit']:.4f}...")
        print("  Only fitting length_scale")
    else:
        print(f"\nFitting Matérn with FIXED σ²={corr_data['variance_logit']:.4f}...")
        print("  Fitting nu and length_scale")

    matern_fit = fit_matern_to_correlation(
        corr_data['r'],
        corr_data['correlation'],
        sigma_sq_empirical=corr_data['variance_logit'],
        fixed_nu=args.fixed_nu
    )

    if matern_fit['fit_success']:
        print(f"\nMatérn fit successful:")
        print(f"  sigma^2 (amplitude): {matern_fit['sigma_sq']:.4f} (FIXED - empirical)")
        print(f"  nu (smoothness):     {matern_fit['nu']:.4f}" + (" (FIXED)" if args.fixed_nu else ""))
        print(f"  l (length scale):    {matern_fit['length_scale']:.4f}")
    else:
        print("Warning: Matérn fitting failed to converge")

    # Save analysis data to npz
    analysis_path = os.path.join(args.output_dir, f'{output_prefix}_porosity_analysis.npz')
    print(f"Saving analysis data to {analysis_path}")

    np.savez(
        analysis_path,
        # Correlation data
        r=corr_data['r'],
        correlation=corr_data['correlation'],
        mean_logit=corr_data['mean_logit'],
        variance_logit=corr_data['variance_logit'],
        # Matérn parameters
        matern_sigma_sq=matern_fit['sigma_sq'],
        matern_nu=matern_fit['nu'],
        matern_length_scale=matern_fit['length_scale'],
        matern_fit_success=matern_fit['fit_success'],
        # Metadata
        kernel_size=args.kernel_size,
        volume_shape=args.volume_shape,
        porosity_field_shape=porosity_field.shape,
        voxel_size=tpc_voxel_size,
        original_voxel_size=args.voxel_size,
        source_file=os.path.basename(data_path),
        use_full_porosity_field=args.use_full_porosity_field,
        fixed_nu=args.fixed_nu if args.fixed_nu else np.nan,
        latent_downsample=args.latent_downsample if args.latent_downsample else 1,
        method='logit'
    )

    print("Done!")


if __name__ == '__main__':
    main()

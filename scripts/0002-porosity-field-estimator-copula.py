#!/usr/bin/env python
"""
Porosity Field Estimator using Gaussian Copula approach.

Supports three warping methods for the marginal distribution:
- histogram: Non-parametric empirical CDF (default)
- logit: Logit-normal assumption
- beta: Beta distribution fit

All methods transform data to Gaussian Z-space, fit Matérn correlation,
and save parameters for inverse transform during sampling.

Usage:
    python scripts/0002-porosity-field-estimator-copula.py \
        --stone Bentheimer \
        --porosity-dir notebooks/exploratory/dfn/data/gpdata2/bentheimer \
        --output-dir notebooks/exploratory/dfn/data/gpdata3-copula/bentheimer \
        --latent-downsample 8 \
        --fixed-nu 1.5 \
        --warping-method beta
"""
import argparse
import os

import numpy as np
import scipy.signal
import scipy.stats
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
        description='Calculate porosity field and fit Matérn GP using Gaussian copula'
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
        help='Prefix for output files'
    )
    parser.add_argument(
        '--kernel-size', type=int, default=256,
        help='Size of the averaging kernel for porosity field calculation'
    )
    parser.add_argument(
        '--save-field', action='store_true',
        help='Save the porosity field volume'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for TPC calculation'
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
        help='Use full "same" convolution for porosity field'
    )
    parser.add_argument(
        '--fixed-nu', type=float, default=None,
        help='Fix nu and only fit length_scale (e.g., 1.5)'
    )
    parser.add_argument(
        '--latent-downsample', type=int, default=None,
        help='Downsample factor for latent space (e.g., 8)'
    )
    parser.add_argument(
        '--porosity-dir', type=str, default=None,
        help='Directory with pre-computed porosity fields'
    )
    parser.add_argument(
        '--n-quantiles', type=int, default=1000,
        help='Number of quantile points to save for CDF interpolation'
    )
    parser.add_argument(
        '--warping-method', type=str, default='histogram',
        choices=['histogram', 'logit', 'beta'],
        help='Warping method for marginal distribution (default: histogram)'
    )
    return parser.parse_args()


def create_averaging_kernel(kernel_size):
    """Create a 3D box averaging kernel."""
    return np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)


def calculate_porosity_field(data, kernel_size):
    """Calculate local porosity field using FFT convolution (valid mode)."""
    kernel = create_averaging_kernel(kernel_size)
    porosity_field = scipy.signal.fftconvolve(
        1 - data.astype(np.float32),
        kernel,
        mode='valid'
    )
    return porosity_field


def calculate_porosity_field_full(data, kernel_size):
    """Calculate local porosity field using FFT convolution (same mode)."""
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    pore_data = 1 - data.astype(np.float32)
    pore_sum = scipy.signal.fftconvolve(pore_data, kernel, mode='same')

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

    porosity_field = pore_sum / normalization
    return porosity_field


def average_volume(volume, kernel_size):
    """Average pool a volume using torch AvgPool3d."""
    avgpool = torch.nn.AvgPool3d(kernel_size)
    volume = torch.from_numpy(volume).float()
    volume = volume.unsqueeze(0).unsqueeze(0)
    avg_volume = avgpool(volume)
    avg_volume = avg_volume.squeeze().numpy()
    return avg_volume


def estimate_empirical_cdf(data_flat, n_quantiles=1000):
    """
    Estimate empirical CDF and return quantile points for interpolation.

    Returns
    -------
    quantile_probs : array of shape (n_quantiles,)
        Probability values from 0 to 1 (uniform grid)
    quantile_values : array of shape (n_quantiles,)
        Corresponding data values (the inverse CDF)
    """
    # Use uniform probability grid, avoiding exact 0 and 1
    # (to avoid infinite values when applying Φ⁻¹)
    eps = 1e-6
    quantile_probs = np.linspace(eps, 1 - eps, n_quantiles)

    # Compute quantiles (this is F̂⁻¹)
    quantile_values = np.quantile(data_flat, quantile_probs)

    return quantile_probs, quantile_values


def fit_beta_distribution(data_flat):
    """
    Fit a beta distribution to data in (0, 1).

    Returns
    -------
    a, b : float
        Beta distribution shape parameters.
    """
    # Clamp to (0, 1) strictly
    eps = 1e-6
    data_clamped = np.clip(data_flat, eps, 1 - eps)

    # Fit beta distribution using MLE
    a, b, loc, scale = scipy.stats.beta.fit(data_clamped, floc=0, fscale=1)

    return a, b


def apply_beta_transform(data, beta_a, beta_b):
    """
    Apply beta-based normal-score transform: Z = Φ⁻¹(F_beta(data))

    Parameters
    ----------
    data : ndarray
        The original data (porosity field in (0, 1))
    beta_a, beta_b : float
        Beta distribution shape parameters

    Returns
    -------
    Z : ndarray
        Transformed data with N(0,1) marginals
    """
    data_flat = data.ravel()

    # Clamp to avoid edge issues
    eps = 1e-6
    data_clamped = np.clip(data_flat, eps, 1 - eps)

    # Apply beta CDF
    u = scipy.stats.beta.cdf(data_clamped, beta_a, beta_b)

    # Clamp u to avoid infinite Z
    u = np.clip(u, eps, 1 - eps)

    # Apply Φ⁻¹
    Z = scipy.stats.norm.ppf(u)

    return Z.reshape(data.shape)


def apply_logit_transform(data):
    """
    Apply logit-based transform and return standardized Z.

    Parameters
    ----------
    data : ndarray
        The original data (porosity field in (0, 1))

    Returns
    -------
    Z : ndarray
        Standardized logit-transformed data
    mean_logit : float
        Mean of logit(data)
    variance_logit : float
        Variance of logit(data)
    """
    data_flat = data.ravel()

    # Clamp to avoid infinite logit
    eps = 1e-6
    data_clamped = np.clip(data_flat, eps, 1 - eps)

    # Logit transform
    logit_data = np.log(data_clamped / (1 - data_clamped))

    # Compute mean and variance
    mean_logit = logit_data.mean()
    variance_logit = logit_data.var()

    # Standardize
    Z = (logit_data - mean_logit) / np.sqrt(variance_logit)

    return Z.reshape(data.shape), mean_logit, variance_logit


def apply_normal_score_transform(data, quantile_probs, quantile_values):
    """
    Apply normal-score transform: Z = Φ⁻¹(F̂(data))

    Parameters
    ----------
    data : ndarray
        The original data (porosity field)
    quantile_probs : array
        Probability values from estimate_empirical_cdf
    quantile_values : array
        Quantile values from estimate_empirical_cdf

    Returns
    -------
    Z : ndarray
        Transformed data with N(0,1) marginals
    """
    data_flat = data.ravel()

    # Apply empirical CDF: F̂(data)
    # For each value, find where it falls in the quantile distribution
    # Using searchsorted + interpolation

    # Sort quantile_values for searchsorted
    sort_idx = np.argsort(quantile_values)
    sorted_values = quantile_values[sort_idx]
    sorted_probs = quantile_probs[sort_idx]

    # Interpolate to get F̂(data)
    # Clamp to avoid extrapolation issues
    u = np.interp(data_flat, sorted_values, sorted_probs)

    # Clamp to (eps, 1-eps) to avoid infinite Z values
    eps = 1e-6
    u = np.clip(u, eps, 1 - eps)

    # Apply Φ⁻¹ to get Z
    Z = scipy.stats.norm.ppf(u)

    return Z.reshape(data.shape)


def calculate_gaussian_correlation(Z_field, device, bins, voxel_size):
    """
    Calculate the two-point correlation function of the normal-scored field.

    Since Z has marginal N(0,1) by construction, we just compute the TPC directly.
    The TPC at r=0 should be approximately 1 (the variance).

    Returns
    -------
    dict with keys:
        - r: radial distances
        - correlation: TPC values (this is the correlation function, variance ~1)
    """
    Z_tensor = torch.tensor(Z_field).float()

    # Z should already have mean ~0, but center just in case
    mean_Z = Z_tensor.mean().item()
    centered_Z = Z_tensor - mean_Z

    # Compute TPC
    tpc = two_point_correlation.tpcf_fft(
        centered_Z.to(device),
        bins=bins,
        voxel_size=voxel_size
    )

    # Verify variance is approximately 1
    var_Z = centered_Z.var().item()

    return {
        'r': tpc.r,
        'correlation': tpc.correlation,
        'mean_Z': mean_Z,
        'variance_Z': var_Z
    }


def fit_matern_correlation(r, correlation, fixed_nu=None):
    """
    Fit Matérn correlation parameters.

    Since Z has variance 1 by construction, we fix σ² = 1 and only fit length_scale
    (and optionally nu).

    The correlation function is: C(r) = k(r) / k(0) = k(r) / σ²
    With σ² = 1, C(r) = k(r).
    """
    sigma_sq = 1.0  # Fixed by construction

    # Exclude r=0
    valid = np.isfinite(correlation) & (r > 1e-10)
    r_clean = r[valid]
    c_clean = correlation[valid]

    if fixed_nu is not None:
        # Only fit length_scale
        if np.abs(fixed_nu - 0.5) < 1e-10:
            def cov_func(r, length_scale):
                return matern_gaussian_process.matern_covariance_half(r, sigma_sq, length_scale)
        elif np.abs(fixed_nu - 1.5) < 1e-10:
            def cov_func(r, length_scale):
                return matern_gaussian_process.matern_covariance_three_half(r, sigma_sq, length_scale)
        elif np.abs(fixed_nu - 2.5) < 1e-10:
            def cov_func(r, length_scale):
                return matern_gaussian_process.matern_covariance_five_half(r, sigma_sq, length_scale)
        else:
            def cov_func(r, length_scale):
                return matern_gaussian_process.matern_covariance(r, sigma_sq, fixed_nu, length_scale)

        # Initial guess for length_scale
        drop_idx = np.abs(c_clean - sigma_sq * 0.36).argmin()
        p0_l = r_clean[drop_idx] if drop_idx < len(r_clean) else r_clean[-1] / 2
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
                'sigma_sq': sigma_sq,
                'nu': fixed_nu,
                'length_scale': popt[0],
                'fit_success': True
            }
        except (RuntimeError, ValueError) as e:
            print(f"  Fitting failed: {e}")
            return {
                'sigma_sq': sigma_sq,
                'nu': fixed_nu,
                'length_scale': np.nan,
                'fit_success': False
            }
    else:
        # Fit both nu and length_scale (σ² still fixed at 1)
        def cov_func(r, nu, length_scale):
            return matern_gaussian_process.matern_covariance(r, sigma_sq, nu, length_scale)

        p0_nu = 1.5
        drop_idx = np.abs(c_clean - sigma_sq * 0.36).argmin()
        p0_l = r_clean[drop_idx] if drop_idx < len(r_clean) else r_clean[-1] / 2
        if p0_l == 0:
            p0_l = 1.0

        try:
            popt, pcov = curve_fit(
                cov_func, r_clean, c_clean,
                p0=[p0_nu, p0_l],
                bounds=([0.1, 1e-6], [30.0, np.inf]),
                maxfev=5000
            )
            return {
                'sigma_sq': sigma_sq,
                'nu': popt[0],
                'length_scale': popt[1],
                'fit_success': True
            }
        except (RuntimeError, ValueError) as e:
            print(f"  Fitting failed: {e}")
            return {
                'sigma_sq': sigma_sq,
                'nu': np.nan,
                'length_scale': np.nan,
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
        import glob
        full_files = glob.glob(os.path.join(args.porosity_dir, "*_porosity_field_full.npy"))
        field_files = glob.glob(os.path.join(args.porosity_dir, "*_porosity_field.npy"))
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
        print(f"\nCalculating porosity field with kernel size {args.kernel_size}...")
        if args.use_full_porosity_field:
            print("  Using full 'same' convolution with boundary normalization...")
            porosity_field_full = calculate_porosity_field_full(data, args.kernel_size)
            half_k = args.kernel_size // 2
            porosity_field = porosity_field_full[half_k:-half_k, half_k:-half_k, half_k:-half_k]
        else:
            porosity_field_full = None
            porosity_field = calculate_porosity_field(data, args.kernel_size)

    # Downsample to latent space if requested
    if args.latent_downsample:
        print(f"\nLatent mode: downsampling porosity field by {args.latent_downsample}")
        print(f"  Before: {porosity_field.shape}")
        porosity_field = average_volume(porosity_field, args.latent_downsample)
        print(f"  After: {porosity_field.shape}")
        tpc_voxel_size = 1.0
        print(f"  Using voxel_size=1.0 for TPC (latent space coordinates)")
    else:
        tpc_voxel_size = args.voxel_size

    print(f"\nPorosity field stats:")
    print(f"  Shape: {porosity_field.shape}")
    print(f"  Range: [{porosity_field.min():.4f}, {porosity_field.max():.4f}]")
    print(f"  Mean: {porosity_field.mean():.4f}")
    print(f"  Std: {porosity_field.std():.4f}")

    # Initialize warping parameters
    data_flat = porosity_field.ravel()
    quantile_probs = None
    quantile_values = None
    beta_a = np.nan
    beta_b = np.nan
    mean_logit = 0.0
    variance_logit = 1.0

    # Step 1 & 2: Apply warping based on method
    print(f"\nWarping method: {args.warping_method}")

    if args.warping_method == 'histogram':
        # Non-parametric empirical CDF
        print(f"Estimating empirical CDF with {args.n_quantiles} quantile points...")
        quantile_probs, quantile_values = estimate_empirical_cdf(data_flat, args.n_quantiles)
        print(f"  Quantile range: [{quantile_values.min():.4f}, {quantile_values.max():.4f}]")

        print("Applying normal-score transform: Z = Φ⁻¹(F̂(ϕ))...")
        Z_field = apply_normal_score_transform(porosity_field, quantile_probs, quantile_values)

    elif args.warping_method == 'beta':
        # Fit beta distribution
        print("Fitting beta distribution to porosity field...")
        beta_a, beta_b = fit_beta_distribution(data_flat)
        print(f"  Beta parameters: a={beta_a:.4f}, b={beta_b:.4f}")

        print("Applying beta transform: Z = Φ⁻¹(F_beta(ϕ))...")
        Z_field = apply_beta_transform(porosity_field, beta_a, beta_b)

    elif args.warping_method == 'logit':
        # Logit-normal
        print("Applying logit transform...")
        Z_field, mean_logit, variance_logit = apply_logit_transform(porosity_field)
        print(f"  mean_logit: {mean_logit:.4f}")
        print(f"  variance_logit: {variance_logit:.4f}")

    print(f"  Z mean: {Z_field.mean():.4f} (should be ~0)")
    print(f"  Z std: {Z_field.std():.4f} (should be ~1)")
    print(f"  Z range: [{Z_field.min():.4f}, {Z_field.max():.4f}]")

    # Optionally save fields
    if args.save_field:
        field_path = os.path.join(args.output_dir, f'{output_prefix}_porosity_field.npy')
        print(f"Saving porosity field to {field_path}")
        np.save(field_path, porosity_field)

        z_path = os.path.join(args.output_dir, f'{output_prefix}_Z_field.npy')
        print(f"Saving Z field to {z_path}")
        np.save(z_path, Z_field)

    # Step 3: Calculate correlation in Z-space
    print(f"\nCalculating TPC of Z field on {args.device}...")
    corr_data = calculate_gaussian_correlation(
        Z_field,
        device=args.device,
        bins=args.tpc_bins,
        voxel_size=tpc_voxel_size
    )
    print(f"  Mean Z: {corr_data['mean_Z']:.4f}")
    print(f"  Variance Z: {corr_data['variance_Z']:.4f} (should be ~1)")

    # Step 4: Fit Matérn correlation (σ² = 1 fixed)
    if args.fixed_nu:
        print(f"\nFitting Matérn correlation with FIXED nu={args.fixed_nu} and FIXED σ²=1...")
        print("  Only fitting length_scale")
    else:
        print(f"\nFitting Matérn correlation with FIXED σ²=1...")
        print("  Fitting nu and length_scale")

    matern_fit = fit_matern_correlation(
        corr_data['r'],
        corr_data['correlation'],
        fixed_nu=args.fixed_nu
    )

    if matern_fit['fit_success']:
        print(f"\nMatérn fit successful:")
        print(f"  sigma^2: {matern_fit['sigma_sq']:.4f} (FIXED by normal-score transform)")
        print(f"  nu: {matern_fit['nu']:.4f}" + (" (FIXED)" if args.fixed_nu else ""))
        print(f"  length_scale: {matern_fit['length_scale']:.4f}")
    else:
        print("Warning: Matérn fitting failed to converge")

    # Save analysis data
    analysis_path = os.path.join(args.output_dir, f'{output_prefix}_porosity_analysis.npz')
    print(f"\nSaving analysis data to {analysis_path}")

    # Determine method name for saving
    method_name = args.warping_method  # 'histogram', 'logit', or 'beta'

    np.savez(
        analysis_path,
        # Warping parameters (method-dependent)
        # Histogram method
        quantile_probs=quantile_probs if quantile_probs is not None else np.array([]),
        quantile_values=quantile_values if quantile_values is not None else np.array([]),
        # Beta method
        beta_a=beta_a,
        beta_b=beta_b,
        # Logit method
        mean_logit=mean_logit,
        variance_logit=variance_logit,
        # Correlation data
        r=corr_data['r'],
        correlation=corr_data['correlation'],
        # Matérn parameters
        matern_sigma_sq=matern_fit['sigma_sq'],
        matern_nu=matern_fit['nu'],
        matern_length_scale=matern_fit['length_scale'],
        matern_fit_success=matern_fit['fit_success'],
        # Metadata
        method=method_name,
        kernel_size=args.kernel_size,
        volume_shape=args.volume_shape,
        porosity_field_shape=porosity_field.shape,
        voxel_size=tpc_voxel_size,
        original_voxel_size=args.voxel_size,
        source_file=os.path.basename(data_path),
        use_full_porosity_field=args.use_full_porosity_field,
        fixed_nu=args.fixed_nu if args.fixed_nu else np.nan,
        latent_downsample=args.latent_downsample if args.latent_downsample else 1,
        n_quantiles=args.n_quantiles,
        warping_method=args.warping_method,
        # Summary stats of original porosity
        porosity_mean=float(porosity_field.mean()),
        porosity_std=float(porosity_field.std()),
        porosity_min=float(porosity_field.min()),
        porosity_max=float(porosity_field.max()),
    )

    print("Done!")


if __name__ == '__main__':
    main()

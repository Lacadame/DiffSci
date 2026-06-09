import warnings

import numpy as np
from scipy.special import kv, gamma, expit
from scipy.stats import norm
from scipy.optimize import curve_fit, differential_evolution
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def warp_from_gpdata(data, gpdata):
    """
    Transform data from original space to Gaussian Z-space using learned warping.

    Parameters
    ----------
    data : ndarray
        Data in original space (e.g., porosity field with values in (0, 1)).
    gpdata : dict-like (npz file)
        GP data containing warping parameters. Must have key 'method':
        - 'histogram': uses quantile_probs, quantile_values (empirical CDF)
        - 'beta': uses beta_a, beta_b (beta distribution)
        - 'logit': uses mean_logit, variance_logit (logit-normal)

    Returns
    -------
    Z : ndarray
        Data transformed to Gaussian space (approximately N(0, 1) marginals).

    Notes
    -----
    - histogram: Z = Φ⁻¹(F̂(data)) where F̂ is the empirical CDF
    - beta: Z = Φ⁻¹(F_beta(data)) where F_beta is the beta CDF
    - logit: Z = (logit(data) - mean_logit) / sqrt(variance_logit)
    """
    from scipy.stats import beta as beta_dist

    data = np.asarray(data)
    original_shape = data.shape
    data_flat = data.ravel()

    method = str(gpdata.get('method', 'logit'))
    eps = 1e-6

    if method == 'histogram':
        # Histogram/empirical CDF method: Z = Φ⁻¹(F̂(data))
        quantile_probs = np.asarray(gpdata['quantile_probs'])
        quantile_values = np.asarray(gpdata['quantile_values'])

        # Sort for interpolation (empirical CDF)
        sort_idx = np.argsort(quantile_values)
        sorted_values = quantile_values[sort_idx]
        sorted_probs = quantile_probs[sort_idx]

        # Interpolate to get F̂(data)
        u = np.interp(data_flat, sorted_values, sorted_probs)
        u = np.clip(u, eps, 1 - eps)

        # Apply Φ⁻¹ to get Z
        Z_flat = norm.ppf(u)

    elif method == 'beta':
        # Beta distribution method: Z = Φ⁻¹(F_beta(data))
        beta_a = float(gpdata['beta_a'])
        beta_b = float(gpdata['beta_b'])

        # Clamp data
        data_clamped = np.clip(data_flat, eps, 1 - eps)

        # Apply beta CDF
        u = beta_dist.cdf(data_clamped, beta_a, beta_b)
        u = np.clip(u, eps, 1 - eps)

        # Apply Φ⁻¹ to get Z
        Z_flat = norm.ppf(u)

    else:
        # Logit method: Z = (logit(data) - mean) / sqrt(var)
        mean_logit = float(gpdata['mean_logit'])
        variance_logit = float(gpdata['variance_logit'])

        # Clamp data to (0, 1) to avoid infinite logit values
        data_clamped = np.clip(data_flat, eps, 1 - eps)

        # Logit transform
        logit_data = np.log(data_clamped / (1 - data_clamped))

        # Standardize
        Z_flat = (logit_data - mean_logit) / np.sqrt(variance_logit)

    return Z_flat.reshape(original_shape)


def unwarp_from_gpdata(Z, gpdata):
    """
    Transform data from Gaussian Z-space back to original space using learned warping.

    Parameters
    ----------
    Z : ndarray
        Data in Gaussian space (approximately N(0, 1) marginals).
    gpdata : dict-like (npz file)
        GP data containing warping parameters. Must have key 'method':
        - 'histogram': uses quantile_probs, quantile_values
        - 'beta': uses beta_a, beta_b
        - 'logit': uses mean_logit, variance_logit

    Returns
    -------
    data : ndarray
        Data transformed back to original space (porosity in [0, 1]).

    Notes
    -----
    - histogram: data = F̂⁻¹(Φ(Z)) where F̂⁻¹ is the quantile function
    - beta: data = F_beta⁻¹(Φ(Z)) where F_beta⁻¹ is beta quantile function
    - logit: data = sigmoid(Z * sqrt(variance_logit) + mean_logit)
    """
    from scipy.stats import beta as beta_dist

    Z = np.asarray(Z)
    original_shape = Z.shape
    Z_flat = Z.ravel()

    method = str(gpdata.get('method', 'logit'))
    eps = 1e-6

    if method == 'histogram':
        # Histogram method: data = F̂⁻¹(Φ(Z))
        quantile_probs = np.asarray(gpdata['quantile_probs'])
        quantile_values = np.asarray(gpdata['quantile_values'])

        # Apply Φ(Z) to get u in (0, 1)
        u = norm.cdf(Z_flat)
        u = np.clip(u, eps, 1 - eps)

        # Interpolate quantile function F̂⁻¹(u)
        data_flat = np.interp(u, quantile_probs, quantile_values)

    elif method == 'beta':
        # Beta method: data = F_beta⁻¹(Φ(Z))
        beta_a = float(gpdata['beta_a'])
        beta_b = float(gpdata['beta_b'])

        # Apply Φ(Z) to get u in (0, 1)
        u = norm.cdf(Z_flat)
        u = np.clip(u, eps, 1 - eps)

        # Apply beta quantile function (ppf) - support is [0, 1]
        data_flat = beta_dist.ppf(u, beta_a, beta_b)

    else:
        # Logit method: data = sigmoid(Z * sqrt(var) + mean)
        mean_logit = float(gpdata['mean_logit'])
        variance_logit = float(gpdata['variance_logit'])

        # Un-standardize
        logit_data = Z_flat * np.sqrt(variance_logit) + mean_logit

        # Inverse logit (sigmoid)
        data_flat = expit(logit_data)

    return data_flat.reshape(original_shape)


def smooth_periodic(field, sigma, mode='wrap'):
    """
    Apply Gaussian smoothing to a field with periodic boundary conditions.

    Parameters
    ----------
    field : ndarray
        Input field of any dimension. Can be a single sample (D, H, W) or
        batched samples (n_samples, D, H, W).
    sigma : float or sequence of floats
        Standard deviation of the Gaussian kernel. If a single float, the same
        sigma is used for all spatial dimensions. If a sequence, specifies sigma
        for each spatial dimension.
    mode : str, optional
        Boundary mode. Default is 'wrap' for periodic boundaries.
        Other options: 'reflect', 'constant', 'nearest', 'mirror'.

    Returns
    -------
    smoothed : ndarray
        Smoothed field with same shape as input.

    Examples
    --------
    >>> # Smooth a single 3D field with sigma=2
    >>> smoothed = smooth_periodic(field, sigma=2.0)

    >>> # Smooth batched samples
    >>> samples = sampler.sample_grid(10)  # (10, 64, 64, 64)
    >>> smoothed = smooth_periodic(samples, sigma=1.5)
    """
    field = np.asarray(field)

    if field.ndim == 0:
        return field

    # Check if batched (heuristic: if 4D, assume first dim is batch for 3D fields)
    # For safety, we apply filter to each sample if it looks batched
    # Actually, gaussian_filter handles this naturally - sigma applies per axis
    # We just need to make sure sigma doesn't apply to batch dimension

    if np.isscalar(sigma):
        # Apply same sigma to all dimensions
        return gaussian_filter(field, sigma=sigma, mode=mode)
    else:
        # sigma is a sequence - use as-is
        return gaussian_filter(field, sigma=sigma, mode=mode)


def matern_covariance_half(r, sigma_sq, length_scale):
    """
    Matérn 1/2 (Exponential) covariance: C(r) = σ² exp(-r/ℓ)

    This is the roughest Matérn kernel (not differentiable).
    """
    r = np.asarray(r)
    return sigma_sq * np.exp(-r / length_scale)


def matern_covariance_three_half(r, sigma_sq, length_scale):
    """
    Matérn 3/2 covariance: C(r) = σ² (1 + √3 r/ℓ) exp(-√3 r/ℓ)

    Once differentiable kernel.
    """
    r = np.asarray(r)
    sqrt3_r_over_l = np.sqrt(3) * r / length_scale
    return sigma_sq * (1 + sqrt3_r_over_l) * np.exp(-sqrt3_r_over_l)


def matern_covariance_five_half(r, sigma_sq, length_scale):
    """
    Matérn 5/2 covariance: C(r) = σ² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(-√5 r/ℓ)

    Twice differentiable kernel.
    """
    r = np.asarray(r)
    sqrt5_r_over_l = np.sqrt(5) * r / length_scale
    return sigma_sq * (1 + sqrt5_r_over_l + (5 * r**2) / (3 * length_scale**2)) * np.exp(-sqrt5_r_over_l)


def matern_covariance(r, sigma_sq, nu, length_scale):
    """
    Matérn Covariance Function.

    Parameters
    ----------
    r : array_like
        Radial distances.
    sigma_sq : float
        Amplitude (Variance) - often called theta or sigma^2.
    nu : float
        Smoothness parameter.
    length_scale : float
        Length scale parameter (l).

    Returns
    -------
    C(r) : array_like
        Covariance values.
    """
    # Use analytical formulas for classical cases
    if np.abs(nu - 0.5) < 1e-10:
        return matern_covariance_half(r, sigma_sq, length_scale)
    elif np.abs(nu - 1.5) < 1e-10:
        return matern_covariance_three_half(r, sigma_sq, length_scale)
    elif np.abs(nu - 2.5) < 1e-10:
        return matern_covariance_five_half(r, sigma_sq, length_scale)

    # General case using Bessel functions
    r = np.asarray(r)

    # 1. Handle the r=0 singularity safely
    # We create a mask for r > 0.
    # At r=0, the limit of the Matern function is exactly sigma_sq.
    with np.errstate(divide='ignore', invalid='ignore'):
        # Stein's parameterization: sqrt(2*nu) * r / l
        scaled_r = (np.sqrt(2 * nu) * r) / length_scale

        # The Formula
        # factor = (2^(1-nu)) / Gamma(nu)
        factor = (2**(1.0 - nu)) / gamma(nu)

        # K_nu calculation
        # We only calculate where r > 0 to avoid RuntimeWarnings
        result = np.zeros_like(r, dtype=np.float64)

        # Compute only for non-zero distances
        mask = r > 1e-8
        if np.any(mask):
            args = scaled_r[mask]
            result[mask] = sigma_sq * factor * (args ** nu) * kv(nu, args)

    # Fill the r=0 (or very close to 0) values with the variance
    result[~mask] = sigma_sq

    return result


def fit_matern_parameters(r_data, corr_data):
    """
    Fits Matérn parameters (sigma^2, nu, l) to experimental data.
    """
    # 1. Clean Data
    # Remove NaNs or Infs if they exist, and exclude r=0
    valid = np.isfinite(corr_data) & (r_data > 1e-10)
    r_clean = r_data[valid]
    c_clean = corr_data[valid]

    # 2. Initial Guesses (p0)
    # sigma_sq: The max value of the correlation (usually at r=0)
    p0_sigma = np.max(c_clean)

    # length_scale: Distance where correlation drops to ~36% (1/e) of max
    # A rough heuristic to help the optimizer
    drop_idx = np.abs(c_clean - p0_sigma*0.36).argmin()
    p0_l = r_clean[drop_idx] if drop_idx < len(r_clean) else r_clean[-1]/2
    if p0_l == 0: p0_l = 1.0

    # nu: Start with 1.5 (intermediate smoothness)
    p0_nu = 1.5

    p0 = [p0_sigma, p0_nu, p0_l]

    # 3. Bounds
    # All parameters must be > 0.
    # nu upper bound: High nu (e.g. > 10) becomes indistinguishable from Gaussian
    # so we cap it loosely to prevent numerical instability in Bessel functions.
    lower_bounds = [1e-6, 0.1, 1e-6]
    upper_bounds = [np.inf, 30.0, np.inf]

    # 4. Optimization
    try:
        popt, pcov = curve_fit(
            matern_covariance,
            r_clean,
            c_clean,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=2000
        )
    except RuntimeError:
        print("Optimization failed to converge.")
        return None, None

    return popt, pcov


def fit_matern_classical(r_data, corr_data, nu):
    """
    Fit Matérn parameters for classical nu values (1/2, 3/2, 5/2).

    Uses analytical formulas and robust optimization for 2D parameter space
    (sigma_sq, length_scale).

    Parameters
    ----------
    r_data : array_like
        Radial distances
    corr_data : array_like
        Correlation values
    nu : float
        Must be one of 0.5, 1.5, or 2.5

    Returns
    -------
    popt : array
        [sigma_sq, nu, length_scale]
    pcov : array
        Covariance matrix (extended to 3x3 with nu variance = 0)
    """
    if not np.any(np.abs(nu - np.array([0.5, 1.5, 2.5])) < 1e-10):
        raise ValueError(f"nu must be 0.5, 1.5, or 2.5, got {nu}")

    # Select the appropriate covariance function
    if np.abs(nu - 0.5) < 1e-10:
        cov_func = matern_covariance_half
    elif np.abs(nu - 1.5) < 1e-10:
        cov_func = matern_covariance_three_half
    else:  # nu == 2.5
        cov_func = matern_covariance_five_half

    # Clean data: remove NaNs/Infs and exclude r=0
    valid = np.isfinite(corr_data) & (r_data > 1e-10)
    r_clean = r_data[valid]
    c_clean = corr_data[valid]

    # Initial guesses
    p0_sigma = np.max(c_clean)
    drop_idx = np.abs(c_clean - p0_sigma*0.36).argmin()
    p0_l = r_clean[drop_idx] if drop_idx < len(r_clean) else r_clean[-1]/2
    if p0_l == 0:
        p0_l = 1.0

    p0 = [p0_sigma, p0_l]

    # Bounds
    lower_bounds = [1e-6, 1e-6]
    upper_bounds = [np.inf, np.inf]

    # Use differential_evolution for global optimization (robust to local minima)
    def objective(params):
        """Sum of squared residuals."""
        sigma_sq, length_scale = params
        if sigma_sq <= 0 or length_scale <= 0:
            return 1e10
        pred = cov_func(r_clean, sigma_sq, length_scale)
        return np.sum((c_clean - pred)**2)

    # First try curve_fit (fast)
    try:
        popt_2d, pcov_2d = curve_fit(
            cov_func,
            r_clean,
            c_clean,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000
        )
        sigma_sq, length_scale = popt_2d
    except (RuntimeError, ValueError):
        # Fall back to differential evolution if curve_fit fails
        print("  curve_fit failed, using differential_evolution...")
        result = differential_evolution(
            objective,
            bounds=[(1e-6, p0_sigma*10), (1e-6, r_clean[-1])],
            seed=42,
            maxiter=1000,
            atol=1e-8,
            tol=1e-8
        )
        sigma_sq, length_scale = result.x
        # Estimate covariance from Hessian approximation
        pcov_2d = None

    # Return in format [sigma_sq, nu, length_scale] for compatibility
    popt = np.array([sigma_sq, nu, length_scale])

    # Extend covariance matrix to 3x3 (nu has zero variance since it's fixed)
    if pcov_2d is not None:
        pcov = np.zeros((3, 3))
        pcov[0, 0] = pcov_2d[0, 0]  # sigma_sq variance
        pcov[0, 2] = pcov_2d[0, 1]  # sigma_sq, length_scale covariance
        pcov[2, 0] = pcov_2d[1, 0]
        pcov[2, 2] = pcov_2d[1, 1]  # length_scale variance
    else:
        pcov = None

    return popt, pcov


class MaternFieldSampler:
    def __init__(
        self,
        mean_val,
        sigma_sq,
        nu,
        length_scale,
        jitter=1e-6
    ):
        """
        Initializes the Gaussian Process with a Matérn kernel.

        Parameters
        ----------
        mean_val : float
            The constant mean of the field (mu).
        sigma_sq : float
            Amplitude (Variance) - often called theta or sigma^2.
        nu : float
            Smoothness parameter.
        length_scale : float
            Length scale parameter (l).
        jitter : float
            Small value added to diagonal for numerical stability (white noise).
        """
        self.mean_val = mean_val
        self.sigma_sq = sigma_sq
        self.nu = nu
        self.length_scale = length_scale
        self.jitter = jitter

        # Field coordinates (set via initialize_field)
        self.X = None
        self.n_points = None
        self.K = None
        self.L = None
        self.grid_shape = None  # Shape when initialized from grid
        self.grid_axes = None   # Original axes for interpolation

    def initialize_field(self, X):
        """
        Initialize the field coordinates and pre-compute covariance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_points, dim)
            The spatial coordinates where the field is defined.
        """
        self.X = np.atleast_2d(X)
        self.n_points = self.X.shape[0]

        # Pre-compute the Covariance Matrix and Cholesky Decomposition
        self.K = self._build_covariance_matrix()

        # Add jitter to diagonal (K + epsilon*I) to ensure positive definiteness
        self.L = np.linalg.cholesky(self.K + np.eye(self.n_points) * self.jitter)

    def initialize_field_from_grid(self, *axes):
        """
        Initialize field coordinates from a meshgrid defined by 1D axes.

        Parameters
        ----------
        *axes : 1D arrays
            Coordinate arrays for each dimension.
            E.g., for 2D: initialize_field_from_grid(x, y)
            E.g., for 3D: initialize_field_from_grid(x, y, z)
        """
        # Store axes for later interpolation
        self.grid_axes = tuple(np.asarray(ax) for ax in axes)

        # Store grid shape: (len(x), len(y), ...) with transpose convention
        self.grid_shape = tuple(len(ax) for ax in axes)

        # Create meshgrid and flatten to (n_points, ndim)
        grids = np.meshgrid(*axes, indexing='ij')
        X = np.stack([g.ravel() for g in grids], axis=-1)

        self.initialize_field(X)

    def _matern_kernel(self, r):
        """Vectorized Matern function handling r=0 singularity"""
        # Use analytical formulas for classical cases
        if np.abs(self.nu - 0.5) < 1e-10:
            return matern_covariance_half(r, self.sigma_sq, self.length_scale)
        elif np.abs(self.nu - 1.5) < 1e-10:
            return matern_covariance_three_half(r, self.sigma_sq, self.length_scale)
        elif np.abs(self.nu - 2.5) < 1e-10:
            return matern_covariance_five_half(r, self.sigma_sq, self.length_scale)

        # General case with Bessel functions
        # 1. Prepare result array
        result = np.zeros_like(r, dtype=np.float64)

        # 2. Handle r > 0
        # We strictly mask 0 to avoid DivByZero or Inf in Bessel calculation
        mask = r > 1e-10
        if np.any(mask):
            r_valid = r[mask]
            # Stein's parameterization
            scaled_r = (np.sqrt(2 * self.nu) * r_valid) / self.length_scale
            factor = (2**(1.0 - self.nu)) / gamma(self.nu)

            result[mask] = self.sigma_sq * factor * (scaled_r ** self.nu) * kv(self.nu, scaled_r)

        # 3. Handle r = 0 (The limit is exactly sigma^2)
        result[~mask] = self.sigma_sq
        return result

    def _build_covariance_matrix(self):
        """Computes pairwise distance and applies kernel"""
        # cdist calculates Euclidean distance between all pairs
        # dists[i, j] = ||x_i - x_j||
        dists = cdist(self.X, self.X, metric='euclidean')
        return self._matern_kernel(dists)

    def sample(self, n_samples=1):
        """
        Generates samples from the GP.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_points)

        Raises
        ------
        RuntimeError
            If initialize_field() has not been called.
        """
        if self.L is None:
            raise RuntimeError(
                "Field not initialized. Call initialize_field(X) first."
            )

        # 1. Sample standard normal noise z ~ N(0, I)
        # Shape: (n_points, n_samples)
        z = np.random.normal(size=(self.n_points, n_samples))

        # 2. Apply Cholesky: y = mu + L * z
        # L is (n_points, n_points)
        y = self.mean_val + self.L @ z

        # 3. Transpose to return (n_samples, n_points)
        return y.T

    def sample_grid(self, n_samples=1):
        """
        Generates samples from the GP and reshapes to grid dimensions.

        Returns
        -------
        samples : ndarray of shape (n_samples, *grid_shape)
            E.g., for 2D grid of shape (Nx, Ny): returns (n_samples, Nx, Ny)
            E.g., for 3D grid of shape (Nx, Ny, Nz): returns (n_samples, Nx, Ny, Nz)

        Raises
        ------
        RuntimeError
            If initialize_field_from_grid() has not been called.
        """
        if self.grid_shape is None:
            raise RuntimeError(
                "Grid shape not set. Use initialize_field_from_grid() "
                "or set grid_shape manually before calling sample_grid()."
            )

        samples = self.sample(n_samples)
        return samples.reshape((n_samples,) + self.grid_shape)

    def sample_grid_interpolated(self, n_samples, *target_axes):
        """
        Sample from GP at coarse grid and interpolate to finer target grid.

        This is an efficient approximation: instead of computing the expensive
        Cholesky decomposition at the fine grid (O(n³)), we sample at the
        coarse grid and use linear interpolation.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        *target_axes : 1D arrays
            Coordinate arrays for the target (fine) grid.
            Must have the same number of dimensions as the initialized grid.

        Returns
        -------
        samples : ndarray of shape (n_samples, *target_shape)
            Interpolated samples on the fine grid.

        Raises
        ------
        RuntimeError
            If initialize_field_from_grid() has not been called.
        ValueError
            If number of target axes doesn't match grid dimensions.
        """
        if self.grid_axes is None:
            raise RuntimeError(
                "Grid axes not set. Use initialize_field_from_grid() first."
            )

        if len(target_axes) != len(self.grid_axes):
            raise ValueError(
                f"Expected {len(self.grid_axes)} axes, got {len(target_axes)}"
            )

        # Sample at coarse grid
        coarse_samples = self.sample_grid(n_samples)

        # Target grid shape
        target_shape = tuple(len(ax) for ax in target_axes)

        # Create target points for interpolation
        target_grids = np.meshgrid(*target_axes, indexing='ij')
        target_points = np.stack([g.ravel() for g in target_grids], axis=-1)

        # Interpolate each sample
        result = np.empty((n_samples,) + target_shape)
        for i in range(n_samples):
            interpolator = RegularGridInterpolator(
                self.grid_axes,
                coarse_samples[i],
                method='linear',
                bounds_error=False,
                fill_value=None  # Extrapolate
            )
            result[i] = interpolator(target_points).reshape(target_shape)

        return result


class PeriodicMaternFieldSampler:
    """
    Periodic Matérn Gaussian Process sampler using circulant embedding (FFT method).

    This sampler generates GP realizations on a regular grid with periodic boundary
    conditions. The key insight is that for a stationary GP on a periodic grid,
    the covariance matrix is circulant and can be diagonalized via FFT.

    Complexity: O(N^d log N) vs O(N^{3d}) for Cholesky on N^d grid.

    The field satisfies: f(x + L) = f(x) for all coordinates, where L is the period.

    Parameters
    ----------
    mean_val : float
        The constant mean of the field (mu).
    sigma_sq : float
        Amplitude (Variance) - often called theta or sigma^2.
    nu : float
        Smoothness parameter.
    length_scale : float
        Length scale parameter (l).
    jitter : float
        Small value for numerical stability (added to zero-frequency eigenvalue).

    Notes
    -----
    For best results, the domain size (period) should be significantly larger
    than the correlation length (L >> length_scale). If L is too small relative
    to length_scale, the circulant embedding may produce negative eigenvalues,
    which are thresholded to zero with a warning.

    References
    ----------
    Wood, A.T.A., & Chan, G. (1994). "Simulation of stationary Gaussian processes
    in [0,1]^d". Journal of Computational and Graphical Statistics, 3(4), 409-432.
    """

    def __init__(
        self,
        mean_val,
        sigma_sq,
        nu,
        length_scale,
        jitter=1e-6
    ):
        self.mean_val = mean_val
        self.sigma_sq = sigma_sq
        self.nu = nu
        self.length_scale = length_scale
        self.jitter = jitter

        # Grid parameters (set via initialize_field_from_grid)
        self.grid_shape = None
        self.grid_axes = None
        self.periods = None
        self.Lambda = None  # Eigenvalues (FFT of circulant first row)
        self._n_negative_eigenvalues = 0

    def initialize_periodic_grid(self, grid_shape, periods):
        """
        Initialize field for periodic sampling with explicit grid shape and periods.

        This is a convenience method when you don't need explicit coordinate axes.
        Coordinates are implicitly [0, period) in each dimension.

        Parameters
        ----------
        grid_shape : tuple of int
            Number of grid points in each dimension, e.g., (64, 64, 64).
        periods : tuple of float
            Period (domain size) in each dimension, e.g., (512.0, 512.0, 512.0).

        Example
        -------
        >>> sampler = PeriodicMaternFieldSampler(mean_val=0, sigma_sq=1, nu=1.5, length_scale=10)
        >>> sampler.initialize_periodic_grid((64, 64, 64), (256.0, 256.0, 256.0))
        >>> samples = sampler.sample_grid(5)
        """
        if len(grid_shape) != len(periods):
            raise ValueError(
                f"grid_shape and periods must have same length, "
                f"got {len(grid_shape)} and {len(periods)}"
            )

        # Create implicit axes: [0, period) with N points
        axes = []
        for N, L in zip(grid_shape, periods):
            ax = np.linspace(0, L - L/N, N)
            axes.append(ax)

        self.initialize_field_from_grid(*axes)

    def initialize_field_from_grid(self, *axes):
        """
        Initialize field for periodic sampling on a regular grid.

        The grid is assumed to be periodic, meaning the field wraps around.
        The period in each dimension is inferred from the axis extent:
        period_i = max(axis_i) - min(axis_i) + delta_i, where delta_i is the
        grid spacing. This ensures the last point connects to the first.

        Parameters
        ----------
        *axes : 1D arrays
            Coordinate arrays for each dimension.
            E.g., for 3D: initialize_field_from_grid(x, y, z)

        Notes
        -----
        The axes should be uniformly spaced for the FFT method to be exact.
        Non-uniform spacing will produce approximate results.
        """
        self.grid_axes = tuple(np.asarray(ax) for ax in axes)
        self.grid_shape = tuple(len(ax) for ax in self.grid_axes)
        ndim = len(self.grid_shape)

        # Infer periods from axes
        # Period = extent + one grid spacing (to close the periodic loop)
        self.periods = []
        for ax in self.grid_axes:
            if len(ax) > 1:
                delta = ax[1] - ax[0]  # Assuming uniform spacing
                period = (ax[-1] - ax[0]) + delta
            else:
                period = 1.0  # Default for single point
            self.periods.append(period)
        self.periods = tuple(self.periods)

        # Build periodic distance array
        # For each dimension, compute min(|i|, N-|i|) * (L/N)
        periodic_distances_1d = []
        for dim_idx, (N, L) in enumerate(zip(self.grid_shape, self.periods)):
            indices = np.arange(N)
            # Periodic distance: min(i, N-i) gives the shorter wrap-around distance
            periodic_dist = np.minimum(indices, N - indices) * (L / N)
            periodic_distances_1d.append(periodic_dist)

        # Create N-dimensional distance array via meshgrid
        dist_grids = np.meshgrid(*periodic_distances_1d, indexing='ij')

        # Euclidean distance from periodic components
        R_squared = sum(dg**2 for dg in dist_grids)
        R = np.sqrt(R_squared)

        # Evaluate Matérn covariance at all periodic distances
        C = self._matern_kernel(R)

        # Eigenvalues via N-dimensional FFT
        # For a real symmetric circulant matrix, eigenvalues are real
        self.Lambda = np.fft.fftn(C).real

        # Check for negative eigenvalues
        self._n_negative_eigenvalues = np.sum(self.Lambda < 0)
        if self._n_negative_eigenvalues > 0:
            min_eigenvalue = self.Lambda.min()
            total_eigenvalues = np.prod(self.grid_shape)
            warnings.warn(
                f"Circulant embedding produced {self._n_negative_eigenvalues} "
                f"negative eigenvalues (out of {total_eigenvalues}). "
                f"Min eigenvalue: {min_eigenvalue:.2e}. "
                f"Consider using a larger domain (period >> length_scale). "
                f"Negative eigenvalues will be set to 0.",
                RuntimeWarning
            )
            self.Lambda = np.maximum(self.Lambda, 0)

        # Add jitter to the zero-frequency component for numerical stability
        zero_freq_idx = tuple(0 for _ in range(ndim))
        self.Lambda[zero_freq_idx] += self.jitter

    def _matern_kernel(self, r):
        """Vectorized Matérn kernel handling r=0 singularity."""
        # Use analytical formulas for classical cases
        if np.abs(self.nu - 0.5) < 1e-10:
            return matern_covariance_half(r, self.sigma_sq, self.length_scale)
        elif np.abs(self.nu - 1.5) < 1e-10:
            return matern_covariance_three_half(r, self.sigma_sq, self.length_scale)
        elif np.abs(self.nu - 2.5) < 1e-10:
            return matern_covariance_five_half(r, self.sigma_sq, self.length_scale)

        # General case with Bessel functions
        r = np.asarray(r)
        result = np.zeros_like(r, dtype=np.float64)

        mask = r > 1e-10
        if np.any(mask):
            r_valid = r[mask]
            scaled_r = (np.sqrt(2 * self.nu) * r_valid) / self.length_scale
            factor = (2**(1.0 - self.nu)) / gamma(self.nu)
            result[mask] = self.sigma_sq * factor * (scaled_r ** self.nu) * kv(self.nu, scaled_r)

        result[~mask] = self.sigma_sq
        return result

    def sample(self, n_samples=1):
        """
        Generate samples from the periodic GP.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_points)
            Flattened samples. Use sample_grid() for shaped output.

        Raises
        ------
        RuntimeError
            If initialize_field_from_grid() has not been called.
        """
        if self.Lambda is None:
            raise RuntimeError(
                "Field not initialized. Call initialize_field_from_grid() first."
            )

        samples = self._sample_fft(n_samples)
        n_points = np.prod(self.grid_shape)
        return samples.reshape(n_samples, n_points)

    def sample_grid(self, n_samples=1, smooth_sigma=None):
        """
        Generate samples from the periodic GP with grid shape.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        smooth_sigma : float or None, optional
            If provided, apply Gaussian smoothing with this sigma (in grid units)
            to each sample using periodic boundary conditions. This can help
            reduce FFT artifacts (horizontal/vertical streaks). Typical values
            are 0.5 to 2.0 grid spacings.

        Returns
        -------
        samples : ndarray of shape (n_samples, *grid_shape)
            E.g., for 3D grid of shape (Nx, Ny, Nz): returns (n_samples, Nx, Ny, Nz)

        Raises
        ------
        RuntimeError
            If initialize_field_from_grid() has not been called.
        """
        if self.Lambda is None:
            raise RuntimeError(
                "Field not initialized. Call initialize_field_from_grid() first."
            )

        samples = self._sample_fft(n_samples)

        if smooth_sigma is not None and smooth_sigma > 0:
            # Apply Gaussian smoothing to each sample with periodic BC
            # sigma=0 means don't smooth the batch dimension
            ndim = len(self.grid_shape)
            sigma_per_axis = (0,) + (smooth_sigma,) * ndim  # (0, sigma, sigma, sigma)
            samples = gaussian_filter(samples, sigma=sigma_per_axis, mode='wrap')

        return samples

    def _sample_fft(self, n_samples):
        """
        Core FFT-based sampling algorithm.

        For a circulant covariance matrix K with eigenvalues λ (computed via FFT),
        we sample by:
        1. Generate complex Gaussian noise Z in Fourier space
        2. Scale by sqrt(λ): Y_hat = sqrt(λ) * Z
        3. Inverse FFT to get real-space sample: Y = IFFT(Y_hat)
        """
        samples = []
        sqrt_Lambda = np.sqrt(self.Lambda)

        for _ in range(n_samples):
            # Complex Gaussian in Fourier space
            # Z ~ CN(0, I) means Re(Z), Im(Z) ~ N(0, 1/2) independently
            Z_real = np.random.randn(*self.grid_shape)
            Z_imag = np.random.randn(*self.grid_shape)
            Z = (Z_real + 1j * Z_imag) / np.sqrt(2)

            # Scale by sqrt of eigenvalues
            Y_hat = sqrt_Lambda * Z

            # Inverse FFT and take real part
            # The imaginary part should be negligible for a proper circulant matrix
            Y = np.fft.ifftn(Y_hat).real

            # Normalize by sqrt(N) to get correct variance
            # (FFT normalization convention)
            Y = Y * np.sqrt(np.prod(self.grid_shape))

            samples.append(self.mean_val + Y)

        return np.array(samples)

    def sample_grid_interpolated(self, n_samples, *target_axes):
        """
        Sample from periodic GP and interpolate to a finer target grid.

        Uses periodic-aware interpolation: values wrap around at boundaries.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        *target_axes : 1D arrays
            Coordinate arrays for the target (fine) grid.
            Must have the same number of dimensions as the initialized grid.
            Target coordinates should be within the periodic domain.

        Returns
        -------
        samples : ndarray of shape (n_samples, *target_shape)
            Interpolated samples on the fine grid.

        Raises
        ------
        RuntimeError
            If initialize_field_from_grid() has not been called.
        ValueError
            If number of target axes doesn't match grid dimensions.
        """
        if self.grid_axes is None:
            raise RuntimeError(
                "Grid axes not set. Use initialize_field_from_grid() first."
            )

        if len(target_axes) != len(self.grid_axes):
            raise ValueError(
                f"Expected {len(self.grid_axes)} axes, got {len(target_axes)}"
            )

        # Sample at coarse grid
        coarse_samples = self.sample_grid(n_samples)

        # Target grid shape
        target_shape = tuple(len(ax) for ax in target_axes)

        # Create target points for interpolation
        target_grids = np.meshgrid(*target_axes, indexing='ij')
        target_points = np.stack([g.ravel() for g in target_grids], axis=-1)

        # For periodic interpolation, we extend the coarse grid by one point
        # in each dimension (wrapping around)
        extended_axes = []
        for ax, period in zip(self.grid_axes, self.periods):
            # Add one more point at the end that equals the first point + period
            extended_ax = np.concatenate([ax, [ax[0] + period]])
            extended_axes.append(extended_ax)

        # Interpolate each sample with periodic extension
        result = np.empty((n_samples,) + target_shape)
        for i in range(n_samples):
            # Extend the sample data by wrapping
            extended_data = self._extend_periodic(coarse_samples[i])

            interpolator = RegularGridInterpolator(
                tuple(extended_axes),
                extended_data,
                method='linear',
                bounds_error=False,
                fill_value=None
            )

            # Wrap target points into the periodic domain before interpolating
            wrapped_points = self._wrap_points(target_points)
            result[i] = interpolator(wrapped_points).reshape(target_shape)

        return result

    def _extend_periodic(self, data):
        """
        Extend data array by one point in each dimension via periodic wrapping.

        For a 3D array of shape (Nx, Ny, Nz), returns shape (Nx+1, Ny+1, Nz+1)
        where the extra slices wrap around to the beginning.
        """
        ndim = data.ndim
        result = data

        for dim in range(ndim):
            # Take the first slice along this dimension and append it
            first_slice = np.take(result, [0], axis=dim)
            result = np.concatenate([result, first_slice], axis=dim)

        return result

    def _wrap_points(self, points):
        """
        Wrap points into the periodic domain [min_ax, min_ax + period).
        """
        wrapped = points.copy()
        for dim, (ax, period) in enumerate(zip(self.grid_axes, self.periods)):
            min_val = ax[0]
            # Wrap to [min_val, min_val + period)
            wrapped[:, dim] = min_val + np.mod(wrapped[:, dim] - min_val, period)
        return wrapped

    @property
    def n_negative_eigenvalues(self):
        """Number of negative eigenvalues encountered (thresholded to 0)."""
        return self._n_negative_eigenvalues


class SpectralMaternFieldSampler:
    """
    Periodic Matérn GP sampler using direct spectral representation.

    This sampler generates GP realizations by directly sampling in Fourier space
    using the analytic spectral density of the Matérn kernel. Unlike the circulant
    embedding method (PeriodicMaternFieldSampler), this approach:

    1. Does NOT discretize the covariance function
    2. Directly uses the closed-form Matérn spectral density
    3. May have different (often better) artifact characteristics

    The Matérn spectral density in d dimensions is:

        S(ω) ∝ (2ν/ℓ² + 4π²|ω|²)^{-(ν + d/2)}

    Parameters
    ----------
    mean_val : float
        The constant mean of the field (mu).
    sigma_sq : float
        Amplitude (Variance) - the field variance at each point.
    nu : float
        Smoothness parameter. Common values: 0.5 (exponential), 1.5, 2.5.
    length_scale : float
        Correlation length scale parameter (ℓ).

    Notes
    -----
    The spectral density is normalized so that the total variance equals sigma_sq.

    References
    ----------
    Rasmussen, C.E., & Williams, C.K.I. (2006). Gaussian Processes for Machine
    Learning. MIT Press. Chapter 4.
    """

    def __init__(
        self,
        mean_val,
        sigma_sq,
        nu,
        length_scale
    ):
        self.mean_val = mean_val
        self.sigma_sq = sigma_sq
        self.nu = nu
        self.length_scale = length_scale

        # Grid parameters (set via initialize)
        self.grid_shape = None
        self.grid_axes = None
        self.periods = None
        self.sqrt_S = None  # Square root of spectral density (precomputed)

    def initialize_periodic_grid(self, grid_shape, periods):
        """
        Initialize for periodic sampling with explicit grid shape and periods.

        Parameters
        ----------
        grid_shape : tuple of int
            Number of grid points in each dimension, e.g., (64, 64, 64).
        periods : tuple of float
            Period (domain size) in each dimension, e.g., (512.0, 512.0, 512.0).
        """
        if len(grid_shape) != len(periods):
            raise ValueError(
                f"grid_shape and periods must have same length, "
                f"got {len(grid_shape)} and {len(periods)}"
            )

        self.grid_shape = tuple(grid_shape)
        self.periods = tuple(periods)
        ndim = len(grid_shape)

        # Create implicit axes
        self.grid_axes = tuple(
            np.linspace(0, L - L/N, N) for N, L in zip(grid_shape, periods)
        )

        # Build frequency grid
        # fftfreq returns frequencies in cycles per sample
        # We need to scale by N/L to get cycles per unit length
        freq_grids = []
        for N, L in zip(grid_shape, periods):
            # fftfreq(N, d=L/N) gives frequencies in cycles per unit length
            freq = np.fft.fftfreq(N, d=L/N)
            freq_grids.append(freq)

        # Create meshgrid of frequencies
        K_grids = np.meshgrid(*freq_grids, indexing='ij')

        # Compute |k|^2 (sum of squared frequencies)
        K_sq = sum(K**2 for K in K_grids)

        # Matérn spectral density (unnormalized)
        # S(ω) ∝ (2ν/ℓ² + 4π²|ω|²)^{-(ν + d/2)}
        alpha = 2 * self.nu / (self.length_scale ** 2)
        exponent = -(self.nu + ndim / 2)
        S_unnorm = (alpha + 4 * np.pi**2 * K_sq) ** exponent

        # Normalize so that total variance equals sigma_sq
        # The variance is (1/V) * sum(S) where V is the domain volume
        # After IFFT with our convention, we need: sum(S) = sigma_sq * N_total
        N_total = np.prod(grid_shape)
        S = S_unnorm * (self.sigma_sq * N_total / S_unnorm.sum())

        # Store sqrt for sampling
        self.sqrt_S = np.sqrt(S)

    def initialize_field_from_grid(self, *axes):
        """
        Initialize from coordinate axes.

        Parameters
        ----------
        *axes : 1D arrays
            Coordinate arrays for each dimension.
        """
        grid_shape = tuple(len(ax) for ax in axes)

        # Infer periods from axes
        periods = []
        for ax in axes:
            ax = np.asarray(ax)
            if len(ax) > 1:
                delta = ax[1] - ax[0]
                period = (ax[-1] - ax[0]) + delta
            else:
                period = 1.0
            periods.append(period)

        self.initialize_periodic_grid(grid_shape, tuple(periods))
        self.grid_axes = tuple(np.asarray(ax) for ax in axes)

    def sample(self, n_samples=1):
        """
        Generate samples from the periodic GP.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_points)
            Flattened samples.
        """
        if self.sqrt_S is None:
            raise RuntimeError(
                "Field not initialized. Call initialize_periodic_grid() first."
            )

        samples = self._sample_spectral(n_samples)
        n_points = np.prod(self.grid_shape)
        return samples.reshape(n_samples, n_points)

    def sample_grid(self, n_samples=1, smooth_sigma=None):
        """
        Generate samples from the periodic GP with grid shape.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        smooth_sigma : float or None, optional
            If provided, apply Gaussian smoothing (periodic BC) to reduce artifacts.

        Returns
        -------
        samples : ndarray of shape (n_samples, *grid_shape)
        """
        if self.sqrt_S is None:
            raise RuntimeError(
                "Field not initialized. Call initialize_periodic_grid() first."
            )

        samples = self._sample_spectral(n_samples)

        if smooth_sigma is not None and smooth_sigma > 0:
            ndim = len(self.grid_shape)
            sigma_per_axis = (0,) + (smooth_sigma,) * ndim
            samples = gaussian_filter(samples, sigma=sigma_per_axis, mode='wrap')

        return samples

    def _sample_spectral(self, n_samples):
        """
        Core spectral sampling algorithm.

        For each sample:
        1. Generate complex Gaussian noise Z in Fourier space
        2. Scale by sqrt(S): Y_hat = sqrt(S) * Z
        3. Inverse FFT to get real-space sample

        To ensure a real-valued output, we use the fact that for real fields,
        the Fourier transform has Hermitian symmetry. We sample a general
        complex field and take the real part, which effectively averages
        contributions from k and -k.
        """
        samples = []

        for _ in range(n_samples):
            # Complex Gaussian noise in Fourier space
            Z_real = np.random.randn(*self.grid_shape)
            Z_imag = np.random.randn(*self.grid_shape)
            Z = (Z_real + 1j * Z_imag) / np.sqrt(2)

            # Scale by sqrt of spectral density
            Y_hat = self.sqrt_S * Z

            # Inverse FFT and take real part
            # The factor sqrt(N) comes from FFT normalization
            Y = np.fft.ifftn(Y_hat).real * np.sqrt(np.prod(self.grid_shape))

            samples.append(self.mean_val + Y)

        return np.array(samples)

    def sample_grid_interpolated(self, n_samples, *target_axes):
        """
        Sample and interpolate to a finer target grid with periodic BC.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        *target_axes : 1D arrays
            Coordinate arrays for the target (fine) grid.

        Returns
        -------
        samples : ndarray of shape (n_samples, *target_shape)
        """
        if self.grid_axes is None:
            raise RuntimeError(
                "Grid axes not set. Use initialize_field_from_grid() first."
            )

        if len(target_axes) != len(self.grid_axes):
            raise ValueError(
                f"Expected {len(self.grid_axes)} axes, got {len(target_axes)}"
            )

        coarse_samples = self.sample_grid(n_samples)
        target_shape = tuple(len(ax) for ax in target_axes)
        target_grids = np.meshgrid(*target_axes, indexing='ij')
        target_points = np.stack([g.ravel() for g in target_grids], axis=-1)

        # Extend axes for periodic interpolation
        extended_axes = []
        for ax, period in zip(self.grid_axes, self.periods):
            extended_ax = np.concatenate([ax, [ax[0] + period]])
            extended_axes.append(extended_ax)

        result = np.empty((n_samples,) + target_shape)
        for i in range(n_samples):
            extended_data = self._extend_periodic(coarse_samples[i])

            interpolator = RegularGridInterpolator(
                tuple(extended_axes),
                extended_data,
                method='linear',
                bounds_error=False,
                fill_value=None
            )

            wrapped_points = self._wrap_points(target_points)
            result[i] = interpolator(wrapped_points).reshape(target_shape)

        return result

    def _extend_periodic(self, data):
        """Extend array by one point in each dim via periodic wrapping."""
        result = data
        for dim in range(data.ndim):
            first_slice = np.take(result, [0], axis=dim)
            result = np.concatenate([result, first_slice], axis=dim)
        return result

    def _wrap_points(self, points):
        """Wrap points into periodic domain."""
        wrapped = points.copy()
        for dim, (ax, period) in enumerate(zip(self.grid_axes, self.periods)):
            min_val = ax[0]
            wrapped[:, dim] = min_val + np.mod(wrapped[:, dim] - min_val, period)
        return wrapped


class RFFMaternFieldSampler:
    """
    Periodic Matérn GP sampler using Random Fourier Features (RFF).

    This sampler approximates GP samples by summing random sinusoids with
    frequencies drawn from the Matérn spectral density. Unlike FFT-based methods,
    this approach:

    1. Does NOT use FFT at all - no grid-aligned frequency artifacts
    2. Directly samples continuous frequencies from the spectral density
    3. Produces smooth, isotropic samples without directional artifacts

    The approximation:
        f(x) ≈ sqrt(2σ²/M) * Σ_{m=1}^{M} cos(2π ω_m · x + φ_m)

    where ω_m are drawn from the Matérn spectral density and φ_m are uniform phases.

    Parameters
    ----------
    mean_val : float
        The constant mean of the field.
    sigma_sq : float
        Amplitude (Variance).
    nu : float
        Smoothness parameter.
    length_scale : float
        Correlation length scale.
    n_features : int
        Number of random Fourier features. More features = better approximation
        but slower. Typically 500-2000 is sufficient for good quality.

    Notes
    -----
    This is an approximation that converges to the true GP as n_features → ∞.
    The error decreases as O(1/sqrt(n_features)).

    For periodic fields, we use frequencies that are integer multiples of 1/L,
    but sampled according to the spectral density weights.

    References
    ----------
    Rahimi, A., & Recht, B. (2007). "Random features for large-scale kernel
    machines". NIPS.
    """

    def __init__(
        self,
        mean_val,
        sigma_sq,
        nu,
        length_scale,
        n_features=1000
    ):
        self.mean_val = mean_val
        self.sigma_sq = sigma_sq
        self.nu = nu
        self.length_scale = length_scale
        self.n_features = n_features

        # Grid parameters
        self.grid_shape = None
        self.grid_axes = None
        self.periods = None

        # Precomputed frequencies and weights
        self._frequencies = None  # (n_features, ndim)
        self._weights = None      # (n_features,) - sqrt of spectral density

    def initialize_periodic_grid(self, grid_shape, periods):
        """
        Initialize for periodic sampling.

        Parameters
        ----------
        grid_shape : tuple of int
            Number of grid points in each dimension.
        periods : tuple of float
            Period (domain size) in each dimension.
        """
        self.grid_shape = tuple(grid_shape)
        self.periods = tuple(float(p) for p in periods)
        ndim = len(grid_shape)

        # Create coordinate axes
        self.grid_axes = tuple(
            np.linspace(0, L - L/N, N) for N, L in zip(grid_shape, periods)
        )

        # Sample frequencies from Matérn spectral density
        self._sample_frequencies(ndim)

    def initialize_field_from_grid(self, *axes):
        """Initialize from coordinate axes."""
        grid_shape = tuple(len(ax) for ax in axes)

        periods = []
        for ax in axes:
            ax = np.asarray(ax)
            if len(ax) > 1:
                delta = ax[1] - ax[0]
                period = (ax[-1] - ax[0]) + delta
            else:
                period = 1.0
            periods.append(period)

        self.initialize_periodic_grid(grid_shape, tuple(periods))
        self.grid_axes = tuple(np.asarray(ax) for ax in axes)

    def _sample_frequencies(self, ndim):
        """
        Sample frequencies from Matérn spectral density.

        The Matérn spectral density in d dimensions (for frequency f in cycles/length):
            S(f) ∝ (2ν/ℓ² + 4π²|f|²)^{-(ν + d/2)}
                 = (ν/(2π²ℓ²) + |f|²)^{-(ν + d/2)}  [factoring out 4π²]

        This is proportional to a scaled multivariate Student-t distribution.

        If t ~ multivariate-t(df) with df = 2ν, then f = t/(2πℓ) has the
        correct spectral density.
        """
        df = 2 * self.nu  # degrees of freedom for Student-t

        # Sample from multivariate Student-t(df)
        # Method: t = z / sqrt(χ²/df) where z ~ N(0, I), χ² ~ chi-squared(df)
        z = np.random.randn(self.n_features, ndim)
        chi2 = np.random.chisquare(df, size=self.n_features)
        t = z / np.sqrt(chi2 / df)[:, np.newaxis]

        # Scale to get frequencies: f = t / (2πℓ)
        # This gives the correct Matérn spectral density
        self._frequencies = t / (2 * np.pi * self.length_scale)

        # Weight for RFF: sqrt(2σ²/M) where M is number of features
        self._weights = np.sqrt(2 * self.sigma_sq / self.n_features)

    def sample(self, n_samples=1):
        """Generate samples (flattened)."""
        if self._frequencies is None:
            raise RuntimeError("Not initialized. Call initialize_periodic_grid() first.")

        samples = self._sample_rff(n_samples)
        n_points = np.prod(self.grid_shape)
        return samples.reshape(n_samples, n_points)

    def sample_grid(self, n_samples=1, smooth_sigma=None):
        """
        Generate samples with grid shape.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        smooth_sigma : float or None
            Optional Gaussian smoothing (usually not needed with RFF).

        Returns
        -------
        samples : ndarray of shape (n_samples, *grid_shape)
        """
        if self._frequencies is None:
            raise RuntimeError("Not initialized. Call initialize_periodic_grid() first.")

        samples = self._sample_rff(n_samples)

        if smooth_sigma is not None and smooth_sigma > 0:
            ndim = len(self.grid_shape)
            sigma_per_axis = (0,) + (smooth_sigma,) * ndim
            samples = gaussian_filter(samples, sigma=sigma_per_axis, mode='wrap')

        return samples

    def _sample_rff(self, n_samples):
        """
        Core RFF sampling.

        f(x) = sqrt(2σ²/M) * Σ_m cos(2π ω_m · x + φ_m)

        For efficiency, we compute this using matrix operations.
        """
        # Build coordinate array: (N_total, ndim)
        grids = np.meshgrid(*self.grid_axes, indexing='ij')
        X = np.stack([g.ravel() for g in grids], axis=-1)  # (N_total, ndim)
        N_total = X.shape[0]

        samples = []
        for _ in range(n_samples):
            # Random phases for this sample
            phases = np.random.uniform(0, 2 * np.pi, self.n_features)

            # Compute: cos(2π * X @ ω.T + φ)
            # X: (N_total, ndim), frequencies: (n_features, ndim)
            # X @ frequencies.T: (N_total, n_features)
            projection = 2 * np.pi * (X @ self._frequencies.T)  # (N_total, n_features)
            projection += phases[np.newaxis, :]  # Add phases

            # Sum of cosines
            cos_features = np.cos(projection)  # (N_total, n_features)
            f = self._weights * cos_features.sum(axis=1)  # (N_total,)

            samples.append(self.mean_val + f.reshape(self.grid_shape))

        return np.array(samples)

    def sample_grid_interpolated(self, n_samples, *target_axes):
        """Sample and interpolate to finer grid."""
        if self.grid_axes is None:
            raise RuntimeError("Grid not initialized.")

        if len(target_axes) != len(self.grid_axes):
            raise ValueError(f"Expected {len(self.grid_axes)} axes")

        coarse_samples = self.sample_grid(n_samples)
        target_shape = tuple(len(ax) for ax in target_axes)
        target_grids = np.meshgrid(*target_axes, indexing='ij')
        target_points = np.stack([g.ravel() for g in target_grids], axis=-1)

        extended_axes = [
            np.concatenate([ax, [ax[0] + period]])
            for ax, period in zip(self.grid_axes, self.periods)
        ]

        result = np.empty((n_samples,) + target_shape)
        for i in range(n_samples):
            extended_data = self._extend_periodic(coarse_samples[i])
            interpolator = RegularGridInterpolator(
                tuple(extended_axes), extended_data,
                method='linear', bounds_error=False, fill_value=None
            )
            wrapped_points = self._wrap_points(target_points)
            result[i] = interpolator(wrapped_points).reshape(target_shape)

        return result

    def _extend_periodic(self, data):
        result = data
        for dim in range(data.ndim):
            first_slice = np.take(result, [0], axis=dim)
            result = np.concatenate([result, first_slice], axis=dim)
        return result

    def _wrap_points(self, points):
        wrapped = points.copy()
        for dim, (ax, period) in enumerate(zip(self.grid_axes, self.periods)):
            min_val = ax[0]
            wrapped[:, dim] = min_val + np.mod(wrapped[:, dim] - min_val, period)
        return wrapped

    def resample_frequencies(self):
        """
        Resample the random frequencies.

        Call this if you want a different set of basis functions for the
        next batch of samples.
        """
        if self.grid_shape is not None:
            self._sample_frequencies(len(self.grid_shape))

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import torch
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma


class MaternFieldSampler:
    def __init__(self, X, mean_val, params, jitter=1e-6):
        """
        Initializes the Gaussian Process with a Matérn kernel.

        Parameters
        ----------
        X : ndarray of shape (n_points, dim)
            The spatial coordinates where the field is defined.
        mean_val : float
            The constant mean of the field (mu).
        params : dict or tuple
            (sigma_sq, nu, length_scale)
        jitter : float
            Small value added to diagonal for numerical stability (white noise).
        """
        self.X = np.atleast_2d(X)
        self.mean_val = mean_val
        self.n_points = self.X.shape[0]

        # Unpack parameters
        if isinstance(params, dict):
            self.sigma_sq = params['sigma_sq']
            self.nu = params['nu']
            self.length_scale = params['length_scale']
        else:
            self.sigma_sq, self.nu, self.length_scale = params

        # Pre-compute the Covariance Matrix and Cholesky Decomposition
        self.K = self._build_covariance_matrix()

        # Add jitter to diagonal (K + epsilon*I) to ensure positive definiteness
        self.L = np.linalg.cholesky(self.K + np.eye(self.n_points) * jitter)

    def _matern_kernel(self, r):
        """Vectorized Matern function handling r=0 singularity"""
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
        """
        # 1. Sample standard normal noise z ~ N(0, I)
        # Shape: (n_points, n_samples)
        z = np.random.normal(size=(self.n_points, n_samples))

        # 2. Apply Cholesky: y = mu + L * z
        # L is (n_points, n_points)
        y = self.mean_val + self.L @ z

        # 3. Transpose to return (n_samples, n_points)
        return y.T


def interpolate_array(arr, method='linear'):
    """
    Interpolate an array [p0, p2, p4, p6] to [p0, p1, p2, p3, p4, p5, p6]
    by treating input as points at x-coordinates [0, 2, 4, 6] and interpolating
    to x-coordinates [0, 1, 2, 3, 4, 5, 6].

    Parameters:
    -----------
    arr : array-like
        Input array of values at even x-coordinates [p0, p2, p4, p6]
    method : str, optional
        Interpolation method: 'linear' or 'spline' (default: 'linear')

    Returns:
    --------
    numpy.ndarray
        Interpolated array [p0, p1, p2, p3, p4, p5, p6]
    """

    arr = np.asarray(arr)

    # Original x-coordinates (even indices: 0, 2, 4, 6, ...)
    n = len(arr)
    x_original = np.arange(0, n * 2, 2)

    # Target x-coordinates (all integers: 0, 1, 2, 3, 4, 5, 6, ...)
    x_target = np.arange(0, (n - 1) * 2 + 1)

    if method == 'linear':
        # Linear interpolation
        interp_func = interp1d(x_original, arr, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        result = interp_func(x_target)

    elif method == 'spline':
        # Cubic spline interpolation (s=0 means exact fit through points)
        spline = UnivariateSpline(x_original, arr, s=0, k=min(3, n - 1))
        result = spline(x_target)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear' or 'spline'")

    return result


def array_map(f, x):
    shape = x.shape
    return np.array([f(xx) for xx in x.reshape(-1)]).reshape(shape)


def map_porosity_to_condition(porosity):
    return {'porosity': torch.tensor([porosity]).float()}


def make_vertical_porosity_map(
    input_array,
    grid_size=(2, 2),
    method='linear',
    as_condition=True
):
    interpolated_array = interpolate_array(input_array, method=method)
    grid = np.ones(grid_size)[..., None] * interpolated_array[None, None, :]
    if as_condition:
        grid = array_map(map_porosity_to_condition, grid)
    else:
        grid = grid.astype(np.float32)
    return grid


def get_grid_center(sizes, grid):
    xs = []
    for size, grid in zip(sizes, grid):
        x = np.linspace(0, size, grid + 1)
        centers = (x[:-1] + x[1:]) / 2
        xs.append(centers)
    return np.meshgrid(*xs)


def matern_grid_sample(sizes, grid, mean_val, params, nsamples=1, as_condition=False):
    grid_center = get_grid_center(sizes, grid)
    grid_center = np.array(grid_center).T.reshape(-1, len(sizes))
    gp = MaternFieldSampler(grid_center, mean_val, params)
    samples = gp.sample(nsamples)
    samples = np.exp(samples) / (1 + np.exp(samples))
    samples = samples.reshape(nsamples, *grid)
    if as_condition:
        samples = array_map(map_porosity_to_condition, samples)
    return samples

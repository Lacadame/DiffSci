import numpy as np
import scipy.fft
import torch
from types import SimpleNamespace


def tpcf_fft(im, bins=50, voxel_size=1):
    """
    Calculates E[psi(x) psi(x+r)] using FFT (Circular Correlation).
    Fast, but assumes periodicity. Valid for non-periodic data
    only if r << image size.
    """
    # --- Numpy Version ---
    if isinstance(im, np.ndarray):
        # 1. FFT Auto-correlation (Circular)
        # We do NOT pad, as requested.
        ft = scipy.fft.rfftn(im)
        psd = np.abs(ft)**2
        autocorr = scipy.fft.irfftn(psd, s=im.shape)
        autocorr = scipy.fft.fftshift(autocorr)

        # Normalize by total N (Biased estimator of Expectation)
        N = im.size
        autocorr = autocorr / N

        return _radial_profile_numpy(autocorr, bins, voxel_size)

    # --- PyTorch Version ---
    elif torch.is_tensor(im):
        device = im.device
        im_float = im.float()

        # 1. FFT Auto-correlation (Circular)
        ft = torch.fft.rfftn(im_float)
        psd = torch.abs(ft)**2
        autocorr = torch.fft.irfftn(psd, s=im_float.shape)
        autocorr = torch.fft.fftshift(autocorr)

        # Normalize
        N = im_float.numel()
        autocorr = autocorr / N

        return _radial_profile_torch(autocorr, bins, voxel_size)


def tpcf_dumb(im, bins=50, voxel_size=1, num_samples=1000):
    """
    Calculates E[psi(x) psi(x+r)] using Spatial shifts (Non-Periodic).
    Extremely slow. Used ONLY for ground-truth verification.

    It randomly samples pairs of points at distance r to estimate the value.
    """
    im = np.asarray(im)
    shape = im.shape
    center = np.array(shape) // 2
    r_max = np.min(center)

    if isinstance(bins, int):
        bin_edges = np.linspace(0, r_max, bins + 1)
    else:
        bin_edges = bins

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    radial_vals = []

    # Brute force: Pick a random direction and distance,
    # shift the array, mask valid areas, compute mean product.
    # To make it deterministic and comparable to FFT radial avg,
    # we strictly iterate over integer shifts (dy, dx).

    # We will accumulate sums and counts for every integer shift
    # then bin them at the end.

    # Coordinate grid for lags
    y_lags = np.arange(-r_max, r_max)
    x_lags = np.arange(-r_max, r_max)

    # To save time in "dumb" mode, we skip iterating every single pixel
    # and just iterate lags.

    r_values = []
    corr_values = []

    # We limit checking to a subset of lags to keep this function from freezing your PC
    # Check a grid of lags
    step = max(1, int(r_max / 30))

    for dy in range(-int(r_max), int(r_max), step):
        for dx in range(-int(r_max), int(r_max), step):
            dist = np.sqrt(dy**2 + dx**2)
            if dist == 0 or dist > r_max: continue

            # Valid Non-Periodic Correlation (Spatial Shift)
            # Extract overlapping regions

            # Slice logic
            sy_src = slice(max(0, -dy), min(shape[0], shape[0] - dy))
            sy_dst = slice(max(0, dy), min(shape[0], shape[0] + dy))

            sx_src = slice(max(0, -dx), min(shape[1], shape[1] - dx))
            sx_dst = slice(max(0, dx), min(shape[1], shape[1] + dx))

            # Product of overlapping areas
            region_1 = im[sy_src, sx_src]
            region_2 = im[sy_dst, sx_dst]

            if region_1.size > 0:
                # E[psi(x)psi(x+r)]
                val = np.mean(region_1 * region_2)
                r_values.append(dist)
                corr_values.append(val)

    # Bin the results
    r_values = np.array(r_values)
    corr_values = np.array(corr_values)

    binned_res = []
    for i in range(len(bin_edges)-1):
        mask = (r_values >= bin_edges[i]) & (r_values < bin_edges[i+1])
        if np.any(mask):
            binned_res.append(np.mean(corr_values[mask]))
        else:
            binned_res.append(0)

    res = SimpleNamespace()
    res.r = bin_centers * voxel_size
    res.correlation = np.array(binned_res)
    return res

# --- Helpers (Same as before) ---

def _radial_profile_numpy(autocorr, bins, voxel_size):
    shape = np.array(autocorr.shape)
    center = shape // 2
    coords = np.indices(shape)
    for i in range(len(shape)):
        coords[i] = coords[i] - center[i]
    dt = np.sqrt(np.sum(coords**2, axis=0))

    if isinstance(bins, int):
        r_max = np.min(center)
        bin_edges = np.linspace(0, r_max, bins + 1)
    else:
        bin_edges = np.array(bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    radial_values = []

    for i in range(len(bin_edges) - 1):
        mask = (dt >= bin_edges[i]) & (dt < bin_edges[i+1])
        if np.any(mask):
            radial_values.append(np.mean(autocorr[mask]))
        else:
            radial_values.append(0.0)

    results = SimpleNamespace()
    results.r = bin_centers * voxel_size
    results.correlation = np.array(radial_values)
    return results

def _radial_profile_torch(autocorr, bins, voxel_size):
    shape = torch.tensor(autocorr.shape, device=autocorr.device)
    center = shape // 2
    coords = torch.meshgrid(*[torch.arange(s, device=autocorr.device) for s in shape], indexing='ij')
    dt = torch.zeros_like(autocorr)
    for i, coord in enumerate(coords):
        dt += (coord - center[i])**2
    dt = torch.sqrt(dt)

    if isinstance(bins, int):
        r_max = torch.min(center).item()
        bin_edges = torch.linspace(0, r_max, bins + 1, device=autocorr.device)
    else:
        bin_edges = torch.tensor(bins, device=autocorr.device)

    bin_indices = torch.bucketize(dt, bin_edges)
    num_buckets = len(bin_edges) + 1
    bin_sums = torch.zeros(num_buckets, device=autocorr.device)
    bin_sums.scatter_add_(0, bin_indices.view(-1), autocorr.view(-1))
    bin_counts = torch.zeros(num_buckets, device=autocorr.device)
    ones = torch.ones_like(autocorr.view(-1))
    bin_counts.scatter_add_(0, bin_indices.view(-1), ones)

    bin_means = torch.zeros_like(bin_sums)
    mask = bin_counts > 0
    bin_means[mask] = bin_sums[mask] / bin_counts[mask]

    valid_means = bin_means[1:len(bin_edges)]
    results = SimpleNamespace()
    results.r = ((bin_edges[:-1] + bin_edges[1:]) / 2).cpu().numpy() * voxel_size
    results.correlation = valid_means.cpu().numpy()
    return results
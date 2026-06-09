"""
Quantitative metrics for periodicity of generated images and volumes.

For a tileable (periodic) image, pixels at one edge are immediate neighbors of
pixels at the opposite edge on the torus. The mean squared difference between
opposite borders therefore gives a simple, monotone measure of periodicity:
zero (or near-zero) for perfectly periodic images, large for images with a
visible seam.

These functions are used in the periodic-boundary 2D campaign to compare
training/inference padding combinations quantitatively, in addition to the
visual roll test.
"""

import numpy as np
import torch


def _as_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def periodicity_mse_2d(
    image,
    border_width: int = 1,
    reduce_axes: bool = True,
):
    """
    Mean-squared difference between opposite borders of a 2D image.

    Args:
        image: tensor or array of shape [..., H, W]. Extra leading dims
               (batch, channel) are averaged over.
        border_width: width of the strip used on each side (default 1 pixel).
                      For w > 1, averages MSE over a strip of that width.
        reduce_axes: if True, return scalar tensor averaging the H-axis and
                     W-axis components; if False, return dict {'h': ..., 'w': ...}.

    Returns:
        scalar tensor, or dict of scalar tensors.
    """
    image = _as_tensor(image).to(torch.float32)
    w = border_width

    top = image[..., :w, :]
    bottom = image[..., -w:, :]
    mse_h = ((top - bottom) ** 2).mean()

    left = image[..., :, :w]
    right = image[..., :, -w:]
    mse_w = ((left - right) ** 2).mean()

    if reduce_axes:
        return (mse_h + mse_w) / 2
    return {'h': mse_h, 'w': mse_w}


def periodicity_ratio_2d(
    image,
    border_width: int = 1,
    n_samples: int = 32,
    seed: int = 0,
):
    """
    Razão entre MSE de bordas opostas e MSE de pares adjacentes no interior.

    Mede o "excesso de seam" relativo à variação natural do meio. Ao contrário
    de periodicity_mse_2d, é insensível ao piso imposto pela arquitetura
    (e.g., gen_padding=circular força MSE de borda perto de zero por construção):
    aqui o numerador e o denominador sofrem o mesmo piso, e a razão revela a
    qualidade do modelo independentemente disso.

    Args:
        image: tensor ou array de shape [..., H, W].
        border_width: largura w da faixa nas bordas (e dos pares interiores).
        n_samples: número de pares interiores amostrados para o baseline.
        seed: semente para a amostragem de pares.

    Returns:
        dict com:
            border_h, border_w     -- MSE de bordas (idêntico a periodicity_mse_2d)
            adjacent_h, adjacent_w -- mediana de MSE de pares adjacentes interiores
            ratio_h, ratio_w       -- border / adjacent por eixo
            ratio_avg              -- média de ratio_h e ratio_w

    Interpretação:
        ratio ≈ 1  -> bordas tão similares quanto vizinhos interiores naturais
        ratio >> 1 -> seam ativo (bordas muito mais diferentes que vizinhança)
        ratio < 1  -> bordas mais similares que vizinhos naturais
                      (suspeita de over-periodização)
    """
    image = _as_tensor(image).to(torch.float32)
    w = border_width

    top = image[..., :w, :]
    bottom = image[..., -w:, :]
    border_h = ((top - bottom) ** 2).mean().item()

    left = image[..., :, :w]
    right = image[..., :, -w:]
    border_w = ((left - right) ** 2).mean().item()

    H, W = image.shape[-2], image.shape[-1]
    gen = torch.Generator().manual_seed(seed)

    h_mses = []
    if H >= 2 * w + 1:
        hi = H - 2 * w
        for _ in range(n_samples):
            idx = int(torch.randint(0, hi, (1,), generator=gen).item())
            a = image[..., idx:idx + w, :]
            b = image[..., idx + w:idx + 2 * w, :]
            h_mses.append(((a - b) ** 2).mean().item())

    w_mses = []
    if W >= 2 * w + 1:
        wi = W - 2 * w
        for _ in range(n_samples):
            idx = int(torch.randint(0, wi, (1,), generator=gen).item())
            a = image[..., :, idx:idx + w]
            b = image[..., :, idx + w:idx + 2 * w]
            w_mses.append(((a - b) ** 2).mean().item())

    adjacent_h = float(np.median(h_mses)) if h_mses else 0.0
    adjacent_w = float(np.median(w_mses)) if w_mses else 0.0

    eps = 1e-12
    ratio_h = border_h / (adjacent_h + eps)
    ratio_w = border_w / (adjacent_w + eps)

    return {
        'border_h': border_h,
        'border_w': border_w,
        'adjacent_h': adjacent_h,
        'adjacent_w': adjacent_w,
        'ratio_h': ratio_h,
        'ratio_w': ratio_w,
        'ratio_avg': 0.5 * (ratio_h + ratio_w),
    }


def periodicity_mse_2d_multiwidth(image, widths=(1, 4, 16)):
    """
    MSE de borda em várias larguras (avg sobre H e W).

    Útil para detectar a zona de transição: se MSE(w=1) ≈ MSE(w=4), o seam é
    apenas na borda externa; se MSE(w=4) ≈ MSE(w=16), há zona de transição
    extensa logo após a borda.

    Returns:
        dict {f'w{w}': mse_avg} para cada w em widths.
    """
    out = {}
    for w in widths:
        avg = periodicity_mse_2d(image, border_width=w, reduce_axes=True)
        out[f'w{w}'] = float(avg.item())
    return out


def periodicity_mse_3d(
    volume,
    border_width: int = 1,
    reduce_axes: bool = True,
):
    """
    Mean-squared difference between opposite faces of a 3D volume.

    Args:
        volume: tensor or array of shape [..., D, H, W]. Extra leading dims
                (batch, channel) are averaged over.
        border_width: width of the slab used on each side (default 1 voxel).
        reduce_axes: if True, return scalar tensor averaging the three
                     axis components; if False, return dict {'d': ..., 'h': ..., 'w': ...}.

    Returns:
        scalar tensor, or dict of scalar tensors.
    """
    volume = _as_tensor(volume).to(torch.float32)
    w = border_width

    front = volume[..., :w, :, :]
    back = volume[..., -w:, :, :]
    mse_d = ((front - back) ** 2).mean()

    top = volume[..., :, :w, :]
    bottom = volume[..., :, -w:, :]
    mse_h = ((top - bottom) ** 2).mean()

    left = volume[..., :, :, :w]
    right = volume[..., :, :, -w:]
    mse_w = ((left - right) ** 2).mean()

    if reduce_axes:
        return (mse_d + mse_h + mse_w) / 3
    return {'d': mse_d, 'h': mse_h, 'w': mse_w}

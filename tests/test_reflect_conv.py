"""
Unit tests for the reflect-padding line (Fase 1 do roteiro v0_2 de bordas periódicas).

Cobre:
  - ReflectConv2d / ReflectConv3d: shape de saída, modos de padding por dim.
  - convert_conv_to_reflect: cópia exata de pesos e contagem.
  - periodicity_mse_2d / 3d: ~0 em imagem periódica, grande em imagem com seam.
"""

import numpy as np
import torch
import torch.nn as nn

from diffsci2.nets.commonlayers import (
    CircularConv2d,
    CircularConv3d,
    ReflectConv2d,
    ReflectConv3d,
)
from diffsci2.extra.punetg_converters import (
    convert_conv_to_circular,
    convert_conv_to_reflect,
    count_conv_layers,
)
from diffsci2.extra.periodicity_metrics import (
    periodicity_mse_2d,
    periodicity_mse_3d,
    periodicity_ratio_2d,
    periodicity_mse_2d_multiwidth,
)


# ----- ReflectConv shape sanity ---------------------------------------------

def test_reflect_conv2d_shape():
    conv = ReflectConv2d(in_channels=3, out_channels=8, kernel_size=3)
    x = torch.randn(2, 3, 16, 16)
    y = conv(x)
    assert y.shape == (2, 8, 16, 16), f"got {y.shape}"


def test_reflect_conv2d_partial_dims():
    # reflect only on H, zero-pad on W
    conv = ReflectConv2d(in_channels=1, out_channels=1, kernel_size=3,
                         reflect_dims=[0])
    x = torch.randn(1, 1, 8, 8)
    y = conv(x)
    assert y.shape == (1, 1, 8, 8)


def test_reflect_conv3d_shape():
    conv = ReflectConv3d(in_channels=2, out_channels=4, kernel_size=3)
    x = torch.randn(1, 2, 8, 8, 8)
    y = conv(x)
    assert y.shape == (1, 4, 8, 8, 8), f"got {y.shape}"


def test_reflect_conv_odd_kernel_only():
    try:
        ReflectConv2d(1, 1, kernel_size=4)
    except AssertionError:
        return
    raise AssertionError("ReflectConv2d should reject even kernel size")


# ----- Converter: weight preservation ---------------------------------------

def test_convert_conv2d_to_reflect_preserves_weights():
    plain = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    reflect = convert_conv_to_reflect(plain, inplace=False)

    # Top-level convert returns wrapped module: build a wrapper around plain
    # for a direct comparison.
    class Wrapper(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

    wrapper = Wrapper(nn.Conv2d(3, 8, kernel_size=3, padding=1))
    converted = convert_conv_to_reflect(wrapper, inplace=False)
    assert isinstance(converted.c, ReflectConv2d)
    assert torch.equal(converted.c.conv.weight.data, wrapper.c.weight.data)
    if wrapper.c.bias is not None:
        assert torch.equal(converted.c.conv.bias.data, wrapper.c.bias.data)


def test_convert_conv3d_to_reflect_preserves_weights():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv3d(2, 4, kernel_size=3)

    w = Wrapper()
    converted = convert_conv_to_reflect(w, inplace=False)
    assert isinstance(converted.c, ReflectConv3d)
    assert torch.equal(converted.c.conv.weight.data, w.c.weight.data)


# ----- count_conv_layers ----------------------------------------------------

def test_count_conv_layers_recognises_reflect_and_circular():
    # NB: count_conv_layers walks module.modules() and therefore counts the
    # inner nn.Conv2d/3d of every Circular/Reflect wrapper as a plain Conv as
    # well. That is the pre-existing behavior of count_conv_layers and not
    # something this PR changed; the test asserts it explicitly so any future
    # refactor that "fixes" the leak is visible.
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv2d(1, 1, 3)
            self.b = CircularConv2d(1, 1, 3)
            self.c = ReflectConv2d(1, 1, 3)
            self.d = nn.Conv3d(1, 1, 3)
            self.e = CircularConv3d(1, 1, 3)
            self.f = ReflectConv3d(1, 1, 3)

    counts = count_conv_layers(Mixed())
    # 1 plain + 1 inner-of-Circular + 1 inner-of-Reflect = 3.
    assert counts['Conv2d'] == 3, counts
    assert counts['CircularConv2d'] == 1, counts
    assert counts['ReflectConv2d'] == 1, counts
    assert counts['Conv3d'] == 3, counts
    assert counts['CircularConv3d'] == 1, counts
    assert counts['ReflectConv3d'] == 1, counts


# ----- Periodicity metric sanity --------------------------------------------

def test_periodicity_mse_2d_zero_on_periodic_image():
    # Build an image with matching opposite borders: pick a sinusoid with
    # period (W-1) so that base[0] = sin(0) = 0 and base[W-1] = sin(2π) ≈ 0.
    # All rows are equal (broadcast), so top == bottom too.
    H, W = 32, 32
    x = np.arange(W, dtype=np.float32)
    base = np.sin(2 * np.pi * x / (W - 1))
    img = np.broadcast_to(base, (H, W)).copy()
    mse = periodicity_mse_2d(img).item()
    assert mse < 1e-6, f"expected ~0, got {mse}"


def test_periodicity_mse_2d_large_on_seamful_image():
    # Linear ramp: top row is 0, bottom row is 1. Far from periodic.
    H, W = 32, 32
    img = np.broadcast_to(np.linspace(0, 1, H, dtype=np.float32)[:, None],
                          (H, W)).copy()
    mse = periodicity_mse_2d(img, reduce_axes=False)
    assert mse['h'].item() > 0.5, mse
    # W-axis should still be ~0 since each column is constant.
    assert mse['w'].item() < 1e-6, mse


def test_periodicity_mse_3d_zero_on_periodic_volume():
    D, H, W = 8, 8, 8
    # Constant volume: all faces identical.
    vol = np.ones((D, H, W), dtype=np.float32) * 0.42
    mse = periodicity_mse_3d(vol).item()
    assert mse < 1e-6, f"expected ~0, got {mse}"


def test_periodicity_mse_3d_detects_seam_per_axis():
    D, H, W = 8, 8, 8
    # Ramp along D only.
    ramp = np.linspace(0, 1, D, dtype=np.float32)
    vol = np.broadcast_to(ramp[:, None, None], (D, H, W)).copy()
    mse = periodicity_mse_3d(vol, reduce_axes=False)
    assert mse['d'].item() > 0.5, mse
    assert mse['h'].item() < 1e-6, mse
    assert mse['w'].item() < 1e-6, mse


# ----- Métrica v2: razão borda/interior + multi-largura ----------------------

def test_periodicity_ratio_2d_seam_image_has_large_ratio():
    # Image with a sharp seam: top half all 0, bottom half all 1.
    H, W = 64, 64
    img = np.zeros((H, W), dtype=np.float32)
    img[H // 2:, :] = 1.0
    r = periodicity_ratio_2d(img, border_width=1, n_samples=16, seed=0)
    # Border h: row 0 (all 0) vs row H-1 (all 1) -> MSE = 1.
    assert r['border_h'] > 0.9, r
    # Adjacent interior pairs almost always within same half -> MSE ≈ 0,
    # so ratio_h >> 1.
    assert r['ratio_h'] > 10, r


def test_periodicity_ratio_2d_periodic_image_has_small_ratio():
    # Image whose border rows literally match: replicate row 0 onto row -1.
    H, W = 32, 32
    img = np.random.RandomState(0).rand(H, W).astype(np.float32)
    img[-1] = img[0]
    img[:, -1] = img[:, 0]
    r = periodicity_ratio_2d(img, border_width=1, n_samples=16, seed=0)
    # Border MSE = 0 by construction -> ratio ≈ 0 regardless of adjacent.
    assert r['border_h'] < 1e-6, r
    assert r['border_w'] < 1e-6, r
    assert r['ratio_avg'] < 1e-3, r


def test_periodicity_mse_2d_multiwidth_returns_expected_keys():
    img = np.random.RandomState(0).rand(32, 32).astype(np.float32)
    out = periodicity_mse_2d_multiwidth(img, widths=(1, 4, 8))
    assert set(out.keys()) == {'w1', 'w4', 'w8'}
    for v in out.values():
        assert isinstance(v, float)
        assert v >= 0


def test_periodicity_mse_2d_multiwidth_w1_matches_periodicity_mse_2d():
    img = np.random.RandomState(1).rand(32, 32).astype(np.float32)
    out = periodicity_mse_2d_multiwidth(img, widths=(1,))
    direct = float(periodicity_mse_2d(img, border_width=1).item())
    assert abs(out['w1'] - direct) < 1e-9


# ----- Smoke: plain → reflect conversion on a small model -------------------

def test_plain_to_reflect_on_tiny_model():
    """
    Smoke test: plain model -> reflect-converted model, end-to-end forward.

    Note (Fase 1): the chained scenario reflect-trained -> circular-at-inference
    (Parte 8 / hipótese principal do v0_2) is NOT covered here. The existing
    _replace_conv_recursive helpers do not unwrap CircularConv/ReflectConv
    wrappers before re-converting, so chaining produces nested wrappers (and
    double padding). This is a known limitation to discuss with Danilo before
    Parte 8 runs.
    """
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 4, 3)
            self.c2 = nn.Conv2d(4, 1, 3)

        def forward(self, x):
            return self.c2(self.c1(x))

    m = Tiny()
    m_reflect = convert_conv_to_reflect(m, inplace=False)
    counts = count_conv_layers(m_reflect)
    assert counts['ReflectConv2d'] == 2, counts

    x = torch.randn(1, 1, 16, 16)
    y = m_reflect(x)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_reflect_conv2d_shape()
    test_reflect_conv2d_partial_dims()
    test_reflect_conv3d_shape()
    test_reflect_conv_odd_kernel_only()
    test_convert_conv2d_to_reflect_preserves_weights()
    test_convert_conv3d_to_reflect_preserves_weights()
    test_count_conv_layers_recognises_reflect_and_circular()
    test_periodicity_mse_2d_zero_on_periodic_image()
    test_periodicity_mse_2d_large_on_seamful_image()
    test_periodicity_mse_3d_zero_on_periodic_volume()
    test_periodicity_mse_3d_detects_seam_per_axis()
    test_periodicity_ratio_2d_seam_image_has_large_ratio()
    test_periodicity_ratio_2d_periodic_image_has_small_ratio()
    test_periodicity_mse_2d_multiwidth_returns_expected_keys()
    test_periodicity_mse_2d_multiwidth_w1_matches_periodicity_mse_2d()
    test_plain_to_reflect_on_tiny_model()
    print("All reflect-line tests passed.")

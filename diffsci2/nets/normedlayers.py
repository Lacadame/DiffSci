"""Magnitude-preserving layers from Karras et al. 2024 (EDM2).

The original three Magnitude-Preserving Linear / Conv2d / Conv3d classes
plus `normalize` are EDM2 CONFIG-D / CONFIG-E learned-layer machinery
(weight-normalize-on-use + forced WN in training).

The additional pieces below — PixelNorm, Gain, mp_silu, mp_sum — are
the EDM2 CONFIG-G fixed-function and "needed-data-dependent" layers.
Together they form a self-contained EDM2 toolkit usable to retrofit
existing architectures (e.g. `VAENet` via the new
`VAENetConfig.norm_type='pixel'` / `output_gain_init` flags).
"""
import math

import torch
import torch.nn.functional as F


class MagnitudePreservingLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features,
                                                     in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.epsilon = 1e-4

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/math.sqrt(fan_in)
        return torch.nn.functional.linear(x, w, self.bias)


class MagnitudePreservingConv2d(torch.nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels,
                                                     in_channels,
                                                     kernel_size,
                                                     kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.epsilon = 1e-4

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/math.sqrt(fan_in)
        return torch.nn.functional.conv2d(x,
                                          w,
                                          self.bias,
                                          self.stride,
                                          self.padding)


class MagnitudePreservingConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels,
                        in_channels,
                        kernel_size,
                        kernel_size,
                        kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.epsilon = 1e-4

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/math.sqrt(fan_in)
        return torch.nn.functional.conv3d(x,
                                          w,
                                          self.bias,
                                          self.stride,
                                          self.padding)


def normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = math.sqrt(n.numel()/x.numel())
    return x / torch.add(eps, n, alpha=alpha)


# ----------------------------------------------------------------------------
# EDM2 CONFIG-G fixed-function and lightweight learned layers.
# ----------------------------------------------------------------------------

class PixelNorm(torch.nn.Module):
    """Per-pixel channel-RMS normalization (Karras et al. ProGAN / EDM2).

    For input ``x`` of shape ``[B, C, *spatial]``::

        y[b, c, p] = x[b, c, p] / sqrt(mean_c(x[b, c, p]**2) + eps)

    Spatial receptive field is 1 — this is the whole point: PixelNorm
    is a drop-in alternative to GroupNorm/InstanceNorm/LayerNorm that
    does **not** introduce cross-pixel coupling. Use it in
    architectures that need a truly finite spatial RF (e.g. tiled
    inference of large volumes without a per-tile calibration pass).

    No affine parameters: Karras notes that the learned scale/bias
    interact badly with weight-normalised convs. If you need per-channel
    offsets, get them from a conv bias.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `dim=1` is the channel axis. We DO NOT reduce over spatial dims.
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class Gain(torch.nn.Module):
    """A single learned scalar multiplier.

    Used at the output of magnitude-preserving networks so the model has
    explicit control over its output scale. Karras' score nets init this
    to 0 because they predict zero noise at step 0; VAE decoders should
    init to 1.0 because they predict ``x`` itself.
    """

    def __init__(self, init: float = 1.0):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * x

    def extra_repr(self) -> str:
        return f"gain={self.gain.item():.4f}"


# Karras CONFIG-G: E_{x~N(0,1)}[silu(x)^2]^(1/2) ≈ 0.596.
_SILU_RMS = 0.596


def mp_silu(x: torch.Tensor) -> torch.Tensor:
    """Magnitude-preserving SiLU: ``silu(x) / 0.596``.

    Output variance ≈ 1 when input variance is. The 0.596 constant is
    ``E_{x~N(0,1)}[silu(x)**2]**0.5``.
    """
    return F.silu(x) / _SILU_RMS


def mp_sum(a: torch.Tensor, b: torch.Tensor, t: float = 0.3) -> torch.Tensor:
    """Magnitude-preserving weighted sum.

    ``((1-t)*a + t*b) / sqrt((1-t)**2 + t**2)``

    Preserves unit variance assuming a, b are independent unit-variance
    tensors. Karras uses t=0.3 inside encoder/decoder ResNet blocks
    (residual branch contributes 30 %).
    """
    denom = math.sqrt((1.0 - t) ** 2 + t ** 2)
    return ((1.0 - t) * a + t * b) / denom

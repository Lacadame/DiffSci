
import torch

import diffsci.models


def test():
    normed_linear = diffsci.models.nets.normedlayers.MagnitudePreservingLinear(
        10, 20)
    x = torch.randn(100, 10)
    y = normed_linear(x)
    assert y.shape == (100, 20)

    normed_conv2d = diffsci.models.nets.normedlayers.MagnitudePreservingConv2d(
        3, 10, 3, padding='same')
    x = torch.randn(100, 3, 32, 32)
    y = normed_conv2d(x)

    assert y.shape == (100, 10, 32, 32)

    normed_conv3d = diffsci.models.nets.normedlayers.MagnitudePreservingConv3d(
        3, 10, 3, padding='same')
    x = torch.randn(100, 3, 32, 32, 32)
    y = normed_conv3d(x)
    assert y.shape == (100, 10, 32, 32, 32)

    x = torch.randn(1, 16, 14, 14)

    attn = diffsci.models.nets.attention.TwoDimensionalAttention(
        num_channels=16,
        num_heads=1,
        type='default',
        attn_residual=False,
        magnitude_preserving=False
    )

    assert (attn(x).shape == (1, 16, 14, 14))

    attn = diffsci.models.nets.attention.TwoDimensionalAttention(
        num_channels=16,
        num_heads=1,
        type='cosine',
        attn_residual=False,
        magnitude_preserving=False
    )

    assert (attn(x).shape == (1, 16, 14, 14))

    attn = diffsci.models.nets.attention.TwoDimensionalAttention(
        num_channels=16,
        num_heads=4,
        type='cosine',
        attn_residual=False,
        magnitude_preserving=False
    )

    assert (attn(x).shape == (1, 16, 14, 14))

    attn = diffsci.models.nets.attention.TwoDimensionalAttention(
        num_channels=16,
        num_heads=4,
        type='cosine',
        attn_residual=True,
        magnitude_preserving=False
    )

    assert (attn(x).shape == (1, 16, 14, 14))

    attn = diffsci.models.nets.attention.TwoDimensionalAttention(
        num_channels=16,
        num_heads=4,
        type='cosine',
        attn_residual=True,
        magnitude_preserving=True
    )

    assert (attn(x).shape == (1, 16, 14, 14))

    attn = diffsci.models.nets.attention.TwoDimensionalAttention(
        num_channels=16,
        num_heads=4,
        type='cosine',
        attn_residual=False,
        magnitude_preserving=True
    )

    assert (attn(x).shape == (1, 16, 14, 14))


if __name__ == '__main__':
    test()
    print('All tests passed')

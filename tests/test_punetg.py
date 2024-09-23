import torch

import diffsci.models


def test():

    x = torch.randn(16, 1, 128, 128)
    downsampler = diffsci.models.nets.commonlayers.DownSampler(1, 1, 2)
    assert (downsampler(x).shape == (16, 1, 64, 64))
    x = torch.randn(16, 1, 64, 64)
    upsampler = diffsci.models.nets.commonlayers.UpSampler(1, 1, 2)
    assert (upsampler(x).shape == (16, 1, 128, 128))

    netconfig = diffsci.models.nets.PUNetGConfig(model_channels=4)
    net = diffsci.models.nets.PUNetG(netconfig)

    x = torch.randn(16, 1, 32, 32)
    t = torch.rand(16)

    xe = net.convin(x)
    te = net.time_projection(t)

    assert (xe.shape == (16, 4, 32, 32))
    assert (te.shape == (16, 4))

    assert (net.forward(x, t).shape == x.shape)


if __name__ == "__main__":
    test()
    print("All tests passed")

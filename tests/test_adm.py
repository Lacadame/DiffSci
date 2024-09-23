import torch

import diffsci.models.nets.adm


def test():
    cin = 16
    cout = 32
    cembed = 24

    x = torch.randn(1, cin, 14, 14)
    te = torch.randn(1, cembed)

    encoder_block = diffsci.models.nets.adm.ADMEncoderBlock(cin, cout, cembed)
    assert (encoder_block(x, te).shape == (1, cout, 14, 14))
    encoder_block = diffsci.models.nets.adm.ADMEncoderBlock(
        cin, cout, cembed,
        has_downsample=True)
    assert (encoder_block(x, te).shape == (1, cout, 7, 7))

    x = torch.randn(1, cin, 14, 14, 14)
    te = torch.randn(1, cembed)
    encoder_block = diffsci.models.nets.adm.ADMEncoderBlock(cin,
                                                            cout,
                                                            cembed,
                                                            dimension=3)
    assert (encoder_block(x, te).shape == (1, cout, 14, 14, 14))

    encoder_block = diffsci.models.nets.adm.ADMEncoderBlock(
        cin, cout, cembed,
        has_residual=True,
        has_attn=True,
        has_downsample=True,
        attn_residual=True,
        dimension=3)
    assert (encoder_block(x, te).shape == (1, cout, 7, 7, 7))

    cin = 16
    cout = 32
    cembed = 24
    cskip = 12

    x = torch.randn(1, cin, 14, 14)
    te = torch.randn(1, cembed)

    encoder_block = diffsci.models.nets.adm.ADMDecoderBlock(cin, cout, cembed)
    assert (encoder_block(x, te).shape == (1, cout, 14, 14))

    encoder_block = diffsci.models.nets.adm.ADMDecoderBlock(
        cin, cout, cembed,
        has_upsample=True)
    assert (encoder_block(x, te).shape == (1, cout, 28, 28))

    encoder_block = diffsci.models.nets.adm.ADMDecoderBlock(
        cin, cout, cembed,
        has_residual=True,
        has_attn=True,
        has_upsample=True)
    assert (encoder_block(x, te).shape == (1, cout, 28, 28))

    xskip = torch.randn(1, 12, 14, 14)
    encoder_block = diffsci.models.nets.adm.ADMDecoderBlock(
        cin, cout, cembed,
        cskip,
        has_residual=True,
        has_attn=True,
        has_upsample=True)
    assert (encoder_block(x, te, xskip).shape == (1, cout, 28, 28))

    t = torch.randn(1)
    time_embed = diffsci.models.nets.adm.ADMTimeEmbedding(24, 128)
    assert (time_embed(t).shape == (1, 128))

    ye = torch.randn(1, 128)
    assert (time_embed(t, ye).shape == (1, 128))

    config = diffsci.models.nets.adm.ADMConfig(skip_integration_type='add')
    model = diffsci.models.nets.adm.ADM(config)

    x = torch.randn(4, 1, 16, 16)
    t = torch.randn(4)

    assert (model(x, t).shape == (4, 1, 16, 16))


if __name__ == '__main__':
    test()
    print("All tests passed")

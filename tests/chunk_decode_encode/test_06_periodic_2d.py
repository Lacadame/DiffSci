"""Test 06 - periodic boundaries (smoke test).

Periodic boundary support requires the encoder convolutions to use circular
padding too, otherwise the chunked-encode-with-periodic-reads will not match
the full-encoder-on-periodic-input. The standard
``convert_conv_to_circular`` helper assumes symmetric padding, but the VAE
encoder uses asymmetric padding on its strided downsample which makes the
residual paths shape-mismatched after conversion.

For now, this is a SMOKE TEST: verify chunk_encode runs with periodicity=True
without errors, output shape is correct, and values are finite. A full
bit-exact periodic correctness test is deferred.
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (
    setup_repo_path, load_toy_2d_vae, load_berea_slice, banner,
)
setup_repo_path()


def main():
    banner("test_06: periodic chunked encode SMOKE TEST")
    from diffsci2.extra.chunk_encode_2 import (
        chunk_encode_2d, prepare_encoder_for_cached_encode,
    )
    from diffsci2.nets.cached_norms import (
        set_all_norms_mode, clear_all_norm_caches,
    )
    vae_net, _ = load_toy_2d_vae('cpu')
    enc = vae_net.encoder
    enc.eval()
    prepare_encoder_for_cached_encode(enc, inplace=True)

    x = load_berea_slice(idx=1, crop=128)
    F = 2 ** (enc.config.num_resolutions - 1)
    L = x.shape[-1] // F  # 32

    with torch.inference_mode():
        clear_all_norm_caches(enc)
        set_all_norms_mode(enc, 'cache')
        _ = enc(x)
        set_all_norms_mode(enc, 'use_cached')

        z_chunked_per = chunk_encode_2d(
            enc, x, chunk_latent=(8, 8),
            device='cpu', use_cached_norms=False,
            periodicity=(True, True), debug=0,
        )
    assert z_chunked_per.shape == (1, 8, L, L)
    assert torch.isfinite(z_chunked_per).all()
    print(f"  z_chunked_per: shape {tuple(z_chunked_per.shape)}, "
          f"range [{float(z_chunked_per.min()):.3f}, {float(z_chunked_per.max()):.3f}]")
    banner("test_06: SMOKE PASSED")


if __name__ == '__main__':
    main()

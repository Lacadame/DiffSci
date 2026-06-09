#!/usr/bin/env python
"""
Integration test for spatial parallelism of PUNetG.

Tests that a spatially-distributed forward pass produces the same output
as a single-GPU forward pass, for every layer type and the full model.

Launch with:
    torchrun --nproc_per_node=2 scripts/test_spatial_parallel.py

Uses CUDA_VISIBLE_DEVICES to select GPUs (set externally).
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diffsci2.nets.punetg import PUNetG
from diffsci2.nets.punetg_config import PUNetGConfig
from diffsci2.nets.commonlayers import GroupLNorm, GroupRMSNorm, CircularConv3d
from diffsci2.distributed import (
    SpatialContext, convert_to_spatial_parallel,
    scatter_along_dim, gather_along_dim,
)
from diffsci2.distributed.layers import SpatialParallelConv3d, SpatialParallelGroupNorm
from diffsci2.distributed.halo_exchange import exchange_halos, pad_with_halos
from diffsci2.distributed.converter import count_spatial_parallel_layers


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def check_close(name, actual, expected, atol=1e-3, rtol=1e-3):
    """Check if two tensors are close, return (pass, max_diff)."""
    max_diff = (actual - expected).abs().max().item()
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    return ok, max_diff, f"  [{status}] {name}: max_diff={max_diff:.2e}"


# =========================================================================
# Individual layer tests
# =========================================================================

def make_and_broadcast(shape, device, rank):
    """Create a random tensor on rank 0 and broadcast to all ranks."""
    x = torch.randn(*shape, device=device) if rank == 0 else torch.empty(*shape, device=device)
    dist.broadcast(x, src=0)
    return x


def broadcast_model_state(model, rank):
    """Broadcast model parameters from rank 0."""
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b, src=0)


def test_conv3d(rank, world_size, ctx, device):
    """Test SpatialParallelConv3d matches standard Conv3d."""
    log(rank, "\n--- Test: Conv3d (standard, zero-pad) ---")
    torch.manual_seed(42)

    conv = nn.Conv3d(8, 16, kernel_size=3, padding='same', bias=True).to(device)
    broadcast_model_state(conv, rank)
    x_full = make_and_broadcast((1, 8, 16, 16, 16), device, rank)

    # Reference: single-GPU
    with torch.no_grad():
        ref = conv(x_full)

    # Parallel: convert and run
    from diffsci2.distributed.converter import _convert_conv3d
    sp_conv = _convert_conv3d(conv, ctx).to(device)
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)

    with torch.no_grad():
        out_local = sp_conv(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("Conv3d (zero-pad)", out_full, ref)
    log(rank, msg)
    return ok


def test_circular_conv3d(rank, world_size, ctx, device):
    """Test SpatialParallelConv3d wrapping CircularConv3d."""
    log(rank, "\n--- Test: CircularConv3d ---")
    torch.manual_seed(42)

    circ_conv = CircularConv3d(8, 16, kernel_size=3, circular_dims=[0, 1, 2]).to(device)
    broadcast_model_state(circ_conv, rank)
    x_full = make_and_broadcast((1, 8, 16, 16, 16), device, rank)

    with torch.no_grad():
        ref = circ_conv(x_full)

    from diffsci2.distributed.converter import _convert_circular_conv3d
    # For circular D-axis: periodic=True in ctx
    ctx_periodic = SpatialContext(
        rank=ctx.rank, world_size=ctx.world_size,
        process_group=ctx.process_group, split_dim=2, periodic=True
    )
    sp_conv = _convert_circular_conv3d(circ_conv, ctx_periodic).to(device)
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx_periodic)

    with torch.no_grad():
        out_local = sp_conv(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx_periodic)

    ok, diff, msg = check_close("CircularConv3d (all periodic)", out_full, ref)
    log(rank, msg)
    return ok


def test_group_rms_norm(rank, world_size, ctx, device):
    """Test SpatialParallelGroupNorm (RMS) matches GroupRMSNorm."""
    log(rank, "\n--- Test: GroupRMSNorm ---")
    torch.manual_seed(42)

    norm = GroupRMSNorm(num_groups=1, num_channels=8, affine=True).to(device)
    broadcast_model_state(norm, rank)
    x_full = make_and_broadcast((1, 8, 16, 16, 16), device, rank)

    with torch.no_grad():
        ref = norm(x_full)

    sp_norm = SpatialParallelGroupNorm(norm, 'GroupRMS', ctx).to(device)
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)

    with torch.no_grad():
        out_local = sp_norm(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("GroupRMSNorm", out_full, ref)
    log(rank, msg)
    return ok


def test_group_ln_norm(rank, world_size, ctx, device):
    """Test SpatialParallelGroupNorm (LN) matches GroupLNorm."""
    log(rank, "\n--- Test: GroupLNorm ---")
    torch.manual_seed(42)

    norm = GroupLNorm(num_groups=1, num_channels=8, affine=True).to(device)
    broadcast_model_state(norm, rank)
    x_full = make_and_broadcast((1, 8, 16, 16, 16), device, rank)

    with torch.no_grad():
        ref = norm(x_full)

    sp_norm = SpatialParallelGroupNorm(norm, 'GroupLN', ctx).to(device)
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)

    with torch.no_grad():
        out_local = sp_norm(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("GroupLNorm", out_full, ref)
    log(rank, msg)
    return ok


def test_maxpool3d(rank, world_size, ctx, device):
    """Test that MaxPool3d works correctly on split tensors (no halo needed)."""
    log(rank, "\n--- Test: MaxPool3d (no halo) ---")
    torch.manual_seed(42)

    pool = nn.MaxPool3d(2).to(device)
    x_full = make_and_broadcast((1, 8, 16, 16, 16), device, rank)

    with torch.no_grad():
        ref = pool(x_full)

    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)
    with torch.no_grad():
        out_local = pool(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("MaxPool3d", out_full, ref)
    log(rank, msg)
    return ok


def test_upsample(rank, world_size, ctx, device):
    """Test that Upsample (nearest) works correctly on split tensors."""
    log(rank, "\n--- Test: Upsample (nearest, 2x) ---")
    torch.manual_seed(42)

    up = nn.Upsample(scale_factor=2).to(device)
    x_full = make_and_broadcast((1, 8, 8, 8, 8), device, rank)

    with torch.no_grad():
        ref = up(x_full)

    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)
    with torch.no_grad():
        out_local = up(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("Upsample", out_full, ref)
    log(rank, msg)
    return ok


# =========================================================================
# Composition tests (narrowing down the bug)
# =========================================================================

def test_resnet_block(rank, world_size, ctx, device):
    """Test a single ResnetBlockC in isolation."""
    from diffsci2.nets.commonlayers import ResnetBlockC
    log(rank, "\n--- Test: Single ResnetBlockC ---")
    torch.manual_seed(42)

    block = ResnetBlockC(
        input_channels=64, time_embed_dim=64, output_channels=None,
        dimension=3, kernel_size=3, dropout=0.0,
        first_norm='GroupLN', second_norm='GroupRMS',
        affine_norm=True, convolution_type='default', bias=True
    ).to(device)
    broadcast_model_state(block, rank)

    x_full = make_and_broadcast((1, 64, 16, 16, 16), device, rank)
    te = make_and_broadcast((1, 64), device, rank)

    with torch.no_grad():
        ref = block(x_full, te)

    block_par = convert_to_spatial_parallel(block, ctx, inplace=False).to(device)
    block_par.eval()
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)

    with torch.no_grad():
        out_local = block_par(x_local, te)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("ResnetBlockC", out_full, ref)
    log(rank, msg)
    return ok


def test_downsampler(rank, world_size, ctx, device):
    """Test DownSampler (MaxPool + Conv) in isolation."""
    from diffsci2.nets.commonlayers import DownSampler
    log(rank, "\n--- Test: DownSampler ---")
    torch.manual_seed(42)

    ds = DownSampler(64, 128, dimension=3, scale_factor=2, kernel_size=3,
                     bias=True, convolution_type='default').to(device)
    broadcast_model_state(ds, rank)

    x_full = make_and_broadcast((1, 64, 16, 16, 16), device, rank)

    with torch.no_grad():
        ref = ds(x_full)

    ds_par = convert_to_spatial_parallel(ds, ctx, inplace=False).to(device)
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)

    with torch.no_grad():
        out_local = ds_par(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("DownSampler", out_full, ref)
    log(rank, msg)
    return ok


def test_upsampler(rank, world_size, ctx, device):
    """Test UpSampler (Upsample + Conv) in isolation."""
    from diffsci2.nets.commonlayers import UpSampler
    log(rank, "\n--- Test: UpSampler ---")
    torch.manual_seed(42)

    us = UpSampler(128, 64, dimension=3, scale_factor=2, kernel_size=3,
                   bias=True, convolution_type='default').to(device)
    broadcast_model_state(us, rank)

    x_full = make_and_broadcast((1, 128, 8, 8, 8), device, rank)

    with torch.no_grad():
        ref = us(x_full)

    us_par = convert_to_spatial_parallel(us, ctx, inplace=False).to(device)
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)

    with torch.no_grad():
        out_local = us_par(x_local)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("UpSampler", out_full, ref)
    log(rank, msg)
    return ok


def test_encode_path(rank, world_size, ctx, device):
    """Test the full encoder path (resnet blocks + downsamplers)."""
    log(rank, "\n--- Test: Encoder path ---")
    torch.manual_seed(42)

    config = PUNetGConfig(
        input_channels=4, output_channels=4, dimension=3,
        model_channels=64, channel_expansion=[2, 4],
        number_resnet_downward_block=2, number_resnet_upward_block=2,
        number_resnet_attn_block=0,
        number_resnet_before_attn_block=3, number_resnet_after_attn_block=3,
        kernel_size=3, in_out_kernel_size=3,
        transition_scale_factor=2, transition_kernel_size=3,
        dropout=0.0, cond_dropout=0.0,
        first_resblock_norm='GroupLN', second_resblock_norm='GroupRMS',
        affine_norm=True, convolution_type='default', num_groups=1, bias=True,
    )
    model = PUNetG(config).to(device)
    broadcast_model_state(model, rank)
    model.eval()

    x_full = make_and_broadcast((1, 4, 16, 16, 16), device, rank)
    t = make_and_broadcast((1,), device, rank)

    # Reference: run convin + encode
    with torch.no_grad():
        x_ref = model.convin(x_full)
        te = model.time_projection(t)
        x_ref, skips_ref = model.encode(x_ref, te)

    # Parallel
    model_par = convert_to_spatial_parallel(model, ctx, inplace=False).to(device)
    model_par.eval()
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)

    with torch.no_grad():
        x_par = model_par.convin(x_local)
        te_par = model_par.time_projection(t)

    # Check convin
    x_par_full = gather_along_dim(x_par, dim=2, ctx=ctx)
    ok_convin, _, msg = check_close("  convin", x_par_full, model.convin(x_full))
    log(rank, msg)

    # Run encode
    with torch.no_grad():
        x_par_enc, skips_par = model_par.encode(x_par, te_par)
    x_par_enc_full = gather_along_dim(x_par_enc, dim=2, ctx=ctx)

    ok_enc, _, msg = check_close("  encode output", x_par_enc_full, x_ref)
    log(rank, msg)

    # Check skips
    for i, (sp, sr) in enumerate(zip(skips_par, skips_ref)):
        sp_full = gather_along_dim(sp, dim=2, ctx=ctx)
        ok_s, _, msg = check_close(f"  skip {i}", sp_full, sr)
        log(rank, msg)

    return ok_convin and ok_enc


# =========================================================================
# Full model test
# =========================================================================

def test_full_punetg(rank, world_size, ctx, device):
    """Test full PUNetG forward pass: parallel vs sequential."""
    log(rank, "\n--- Test: Full PUNetG forward pass ---")
    torch.manual_seed(42)

    config = PUNetGConfig(
        input_channels=4,
        output_channels=4,
        dimension=3,
        model_channels=64,
        channel_expansion=[2, 4],
        number_resnet_downward_block=2,
        number_resnet_upward_block=2,
        number_resnet_attn_block=0,  # No attention
        number_resnet_before_attn_block=3,
        number_resnet_after_attn_block=3,
        kernel_size=3,
        in_out_kernel_size=3,
        transition_scale_factor=2,
        transition_kernel_size=3,
        dropout=0.0,  # Disable dropout for deterministic test
        cond_dropout=0.0,
        first_resblock_norm='GroupLN',
        second_resblock_norm='GroupRMS',
        affine_norm=True,
        convolution_type='default',
        num_groups=1,
        bias=True,
    )

    D, H, W = 16, 16, 16

    # Create model on rank 0, broadcast weights to all
    model_ref = PUNetG(config).to(device)
    broadcast_model_state(model_ref, rank)
    model_ref.eval()

    # Create input on rank 0, broadcast
    x_full = make_and_broadcast((1, 4, D, H, W), device, rank)
    t = make_and_broadcast((1,), device, rank)

    # --- Reference: single-GPU forward ---
    with torch.no_grad():
        ref = model_ref(x_full, t)

    # --- Parallel: convert and run ---
    model_par = convert_to_spatial_parallel(model_ref, ctx, inplace=False)
    model_par.to(device)
    model_par.eval()

    if rank == 0:
        counts = count_spatial_parallel_layers(model_par)
        log(rank, f"  Layer counts after conversion: {counts}")

    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)
    with torch.no_grad():
        out_local = model_par(x_local, t)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)

    ok, diff, msg = check_close("PUNetG full forward", out_full, ref, atol=5e-3, rtol=5e-3)
    log(rank, msg)
    return ok


# =========================================================================
# Main
# =========================================================================

def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    log(rank, f"=== Spatial Parallelism Test ===")
    log(rank, f"World size: {world_size}, Devices: {[f'cuda:{i}' for i in range(world_size)]}")

    ctx = SpatialContext(
        rank=rank, world_size=world_size,
        process_group=None,  # Default group
        split_dim=2, periodic=False,
    )

    results = []

    # Layer-level tests
    results.append(("Conv3d", test_conv3d(rank, world_size, ctx, device)))
    results.append(("CircularConv3d", test_circular_conv3d(rank, world_size, ctx, device)))
    results.append(("GroupRMSNorm", test_group_rms_norm(rank, world_size, ctx, device)))
    results.append(("GroupLNorm", test_group_ln_norm(rank, world_size, ctx, device)))
    results.append(("MaxPool3d", test_maxpool3d(rank, world_size, ctx, device)))
    results.append(("Upsample", test_upsample(rank, world_size, ctx, device)))

    # Composition tests
    results.append(("ResnetBlockC", test_resnet_block(rank, world_size, ctx, device)))
    results.append(("DownSampler", test_downsampler(rank, world_size, ctx, device)))
    results.append(("UpSampler", test_upsampler(rank, world_size, ctx, device)))
    results.append(("EncodePath", test_encode_path(rank, world_size, ctx, device)))

    # Full model test
    results.append(("PUNetG", test_full_punetg(rank, world_size, ctx, device)))

    # Summary
    log(rank, "\n=== Summary ===")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        log(rank, f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        log(rank, "\nAll tests passed!")
    else:
        log(rank, "\nSome tests FAILED.")

    dist.destroy_process_group()
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

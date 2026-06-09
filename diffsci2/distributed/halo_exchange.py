"""
Halo exchange primitives for spatial parallelism.

Handles point-to-point communication of boundary slices between neighboring GPUs.
"""

import torch
import torch.distributed as dist


def exchange_halos(tensor, halo_width, ctx):
    """Synchronous halo exchange along the split dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor shard, e.g. [B, C, D_local, H, W].
    halo_width : int
        Number of slices to exchange (typically kernel_size // 2).
    ctx : SpatialContext
        Distributed context.

    Returns
    -------
    recv_left : torch.Tensor or None
        Halo received from left neighbor. Shape [..., halo_width, ...].
        None if no left neighbor (non-periodic, rank=0).
    recv_right : torch.Tensor or None
        Halo received from right neighbor.
        None if no right neighbor (non-periodic, rank=last).
    """
    dim = ctx.split_dim

    # Build slice objects for extracting boundary faces
    def _slice_at(d, start, end):
        slices = [slice(None)] * tensor.ndim
        slices[d] = slice(start, end)
        return tuple(slices)

    send_left = tensor[_slice_at(dim, 0, halo_width)].contiguous()
    send_right = tensor[_slice_at(dim, -halo_width, None)].contiguous()

    recv_left = torch.zeros_like(send_right) if ctx.left_rank is not None else None
    recv_right = torch.zeros_like(send_left) if ctx.right_rank is not None else None

    ops = []

    # Phase 1 - "Right exchange": send right boundary rightward, receive left halo from left.
    # My right boundary becomes my right neighbor's recv_left.
    # My recv_left comes from my left neighbor's right boundary.
    # Ordering matters for FIFO matching when left_rank == right_rank (2 GPUs periodic).
    if ctx.right_rank is not None:
        ops.append(dist.P2POp(dist.isend, send_right, ctx.right_rank, group=ctx.process_group))
    if ctx.left_rank is not None:
        ops.append(dist.P2POp(dist.irecv, recv_left, ctx.left_rank, group=ctx.process_group))

    # Phase 2 - "Left exchange": send left boundary leftward, receive right halo from right.
    if ctx.left_rank is not None:
        ops.append(dist.P2POp(dist.isend, send_left, ctx.left_rank, group=ctx.process_group))
    if ctx.right_rank is not None:
        ops.append(dist.P2POp(dist.irecv, recv_right, ctx.right_rank, group=ctx.process_group))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return recv_left, recv_right


def pad_with_halos(tensor, recv_left, recv_right, dim):
    """Concatenate received halos onto the local tensor along dim.

    For boundary GPUs with no neighbor, pads with zeros.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor shard.
    recv_left : torch.Tensor or None
        Left halo (or None for zero-pad).
    recv_right : torch.Tensor or None
        Right halo (or None for zero-pad).
    dim : int
        Dimension to pad along.

    Returns
    -------
    torch.Tensor
        Padded tensor with halo_width extra slices on each side along dim.
    """
    parts = []

    if recv_left is not None:
        parts.append(recv_left)
    else:
        # Zero-pad: create a zero tensor with same shape except halo_width along dim
        # Infer halo width from the other side
        halo_width = recv_right.shape[dim] if recv_right is not None else 0
        if halo_width > 0:
            shape = list(tensor.shape)
            shape[dim] = halo_width
            parts.append(torch.zeros(shape, dtype=tensor.dtype, device=tensor.device))

    parts.append(tensor)

    if recv_right is not None:
        parts.append(recv_right)
    else:
        halo_width = recv_left.shape[dim] if recv_left is not None else 0
        if halo_width > 0:
            shape = list(tensor.shape)
            shape[dim] = halo_width
            parts.append(torch.zeros(shape, dtype=tensor.dtype, device=tensor.device))

    return torch.cat(parts, dim=dim)

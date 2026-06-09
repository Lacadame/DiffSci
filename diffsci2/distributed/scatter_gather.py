"""
Utilities for scattering input and gathering output along the split dimension.
"""

import torch
import torch.distributed as dist


def scatter_along_dim(tensor, dim, ctx):
    """Split tensor along dim, return the chunk for this rank.

    Parameters
    ----------
    tensor : torch.Tensor
        Full tensor (must be present on this rank).
    dim : int
        Dimension to split along.
    ctx : SpatialContext
        Distributed context.

    Returns
    -------
    torch.Tensor
        This rank's chunk, contiguous.
    """
    chunks = tensor.chunk(ctx.world_size, dim=dim)
    return chunks[ctx.rank].contiguous()


def gather_along_dim(tensor_local, dim, ctx):
    """AllGather local tensors along dim to reconstruct full tensor.

    Parameters
    ----------
    tensor_local : torch.Tensor
        This rank's local shard.
    dim : int
        Dimension to gather along.
    ctx : SpatialContext
        Distributed context.

    Returns
    -------
    torch.Tensor
        Full tensor reconstructed from all ranks.
    """
    gathered = [torch.empty_like(tensor_local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, tensor_local.contiguous(), group=ctx.process_group)
    return torch.cat(gathered, dim=dim)

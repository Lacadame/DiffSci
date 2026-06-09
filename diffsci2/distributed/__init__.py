"""
Spatial parallelism for PUNetG inference.

Distributes a PUNetG forward pass across N GPUs by splitting the spatial domain
along the depth (D) axis. Each GPU holds the full model weights but only 1/N
of the activation volume.

Usage:
    from diffsci2.distributed import SpatialContext, convert_to_spatial_parallel
    from diffsci2.distributed import scatter_along_dim, gather_along_dim

    ctx = SpatialContext(rank, world_size, process_group, split_dim=2, periodic=False)
    model = convert_to_spatial_parallel(model, ctx)
    x_local = scatter_along_dim(x_full, dim=2, ctx=ctx)
    out_local = model(x_local, t)
    out_full = gather_along_dim(out_local, dim=2, ctx=ctx)
"""

from .spatial_context import SpatialContext
from .converter import convert_to_spatial_parallel
from .scatter_gather import scatter_along_dim, gather_along_dim

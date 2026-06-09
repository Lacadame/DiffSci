"""
SpatialContext: lightweight config passed to all spatial-parallel layers.
"""

import torch.distributed as dist


class SpatialContext:
    """Holds distributed state for spatial parallelism along one axis.

    Parameters
    ----------
    rank : int
        This process's rank within the spatial group.
    world_size : int
        Number of GPUs in the spatial group.
    process_group : dist.ProcessGroup
        torch.distributed process group for communication.
    split_dim : int
        Tensor dimension to split. Default 2 = D in [B, C, D, H, W].
    periodic : bool
        Whether boundary conditions wrap around (ring topology).
    """

    def __init__(self, rank, world_size, process_group, split_dim=2, periodic=False):
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.split_dim = split_dim
        self.periodic = periodic

    @property
    def left_rank(self):
        if self.rank > 0:
            return self.rank - 1
        elif self.periodic:
            return self.world_size - 1
        else:
            return None

    @property
    def right_rank(self):
        if self.rank < self.world_size - 1:
            return self.rank + 1
        elif self.periodic:
            return 0
        else:
            return None

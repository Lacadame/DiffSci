"""Cached-target chunk loaders.

Two datasets:

  CachedChunkSampler (IterableDataset) -- training stream
      Random sampling from the regressor's precomputed train chunk index,
      with random Oh symmetry augmentation. Yields (x, y_raw, info) where
      y_raw is the cached morphological target. Rank-aware: the per-worker
      RNG mixes in the DDP rank so different GPUs see disjoint streams.

  EvalPackDataset (map-style Dataset) -- held-out validation
      A fixed subset of the regressor's test chunk index, no augmentation.
      Yields (x, y_raw). Lightning auto-distributes this across DDP ranks.

Convention (matches `diffsci2` and the regressor): 1 = solid, 0 = pore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from diffsci2.data.data_symmetries import CubeSymmetry

# `_paths` puts the poreregressor scripts dir on sys.path so these imports
# resolve. STONES + CHUNK_SIZE come from poreregressor.data_constants.
from . import _paths  # noqa: F401
from poreregressor.chunk_index import load_chunk_index
from poreregressor.data_constants import STONES, CHUNK_SIZE


@dataclass(frozen=True)
class ChunkSample:
    chunk_idx: int
    stone_idx: int
    i: int
    j: int
    k: int
    sym_id: int


# ---------------------------------------------------------------------------
# Training stream (IterableDataset)
# ---------------------------------------------------------------------------

class CachedChunkSampler(IterableDataset):
    """Infinite stream of `(x_real, y_true_raw, info)` from the train cache.

    `rank` and `world_size` are mixed into the per-worker RNG seed so that
    distinct DDP ranks generate disjoint sampling streams. Set them from the
    LightningModule's `train_dataloader` (where `self.trainer.global_rank`
    and `self.trainer.world_size` are available).
    """

    def __init__(
        self,
        split: str = "train",
        augment: bool = True,
        seed_base: int = 0,
        rank: int = 0,
        world_size: int = 1,
        chunks_path: Optional[str] = None,
        targets_path: Optional[str] = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(split)
        self.split = split
        self.augment = augment
        self.seed_base = seed_base
        self.rank = rank
        self.world_size = world_size

        if chunks_path is None:
            chunks_path = (_paths.DEFAULT_REGRESSOR_TRAIN_CHUNKS
                           if split == "train"
                           else _paths.DEFAULT_REGRESSOR_TEST_CHUNKS)
        if targets_path is None:
            targets_path = (_paths.DEFAULT_REGRESSOR_TRAIN_TARGETS
                            if split == "train"
                            else _paths.DEFAULT_REGRESSOR_TEST_TARGETS)

        cs = load_chunk_index(chunks_path)
        targets = np.load(targets_path).astype(np.float32)
        keep = np.isfinite(targets).all(axis=1)
        if keep.sum() == 0:
            raise RuntimeError(f"no finite targets in {targets_path}")

        self._stone_idx = cs.stone_idx[keep].astype(np.int32)
        self._i = cs.i[keep].astype(np.int32)
        self._j = cs.j[keep].astype(np.int32)
        self._k = cs.k[keep].astype(np.int32)
        self._targets = targets[keep]
        self._N = int(keep.sum())

        self._volumes: list[np.memmap] | None = None
        self._symm: CubeSymmetry | None = None
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        # An "epoch" worth of samples (not literally the stream length).
        # Lightning never needs this for IterableDataset training, but a few
        # debug paths inspect __len__.
        return self._N

    @property
    def targets(self) -> np.ndarray:
        return self._targets

    def _ensure_state(self):
        if self._volumes is None:
            self._volumes = [
                np.memmap(c.raw_path, dtype=np.uint8, mode="r", shape=c.shape)
                for c in STONES
            ]
        if self._symm is None:
            self._symm = CubeSymmetry()
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            wid = 0 if wi is None else wi.id
            self._rng = np.random.default_rng(
                self.seed_base + self.rank * 100003 + wid * 7919 + 1
            )

    def _read_chunk(self, idx: int) -> tuple[np.ndarray, int]:
        s = int(self._stone_idx[idx])
        i = int(self._i[idx]); j = int(self._j[idx]); k = int(self._k[idx])
        sub = np.asarray(
            self._volumes[s][i:i + CHUNK_SIZE,
                              j:j + CHUNK_SIZE,
                              k:k + CHUNK_SIZE]
        ).copy()
        if sub.dtype != np.bool_ and sub.max() > 1:
            sub = (sub > 0).astype(np.uint8)
        return sub, s

    def _sample_one(self) -> tuple[torch.Tensor, torch.Tensor, ChunkSample]:
        self._ensure_state()
        idx = int(self._rng.integers(0, self._N))
        sub, s = self._read_chunk(idx)
        sym_id = 0
        if self.augment:
            sym_id = int(self._rng.integers(0, 48))
            if sym_id != 0:
                sub = self._symm.apply(sub, sym_id)
        x = torch.from_numpy(sub.astype(np.float32, copy=False)).unsqueeze(0)
        y = torch.from_numpy(self._targets[idx])
        info = ChunkSample(
            chunk_idx=idx, stone_idx=s,
            i=int(self._i[idx]), j=int(self._j[idx]), k=int(self._k[idx]),
            sym_id=sym_id,
        )
        return x, y, info

    def __iter__(self):
        while True:
            yield self._sample_one()


def collate_cached(batch):
    """Drop the ChunkSample debug info; Lightning's `move_data_to_device`
    walks the batch and refuses to recurse into frozen dataclasses."""
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


# ---------------------------------------------------------------------------
# Validation pack (map-style)
# ---------------------------------------------------------------------------

class EvalPackDataset(Dataset):
    """Fixed held-out subset of the regressor's test chunks + cached targets.

    No augmentation; same seed every time the dataset is instantiated.
    """

    def __init__(
        self,
        n: int = 64,
        seed: int = 0,
        chunks_path: Optional[str] = None,
        targets_path: Optional[str] = None,
    ):
        if chunks_path is None:
            chunks_path = _paths.DEFAULT_REGRESSOR_TEST_CHUNKS
        if targets_path is None:
            targets_path = _paths.DEFAULT_REGRESSOR_TEST_TARGETS

        cs = load_chunk_index(chunks_path)
        targets = np.load(targets_path).astype(np.float32)
        keep = np.isfinite(targets).all(axis=1)
        idx = np.flatnonzero(keep)
        rng = np.random.default_rng(seed)
        take = rng.choice(idx, size=min(n, len(idx)), replace=False)
        take.sort()
        self.stone_idx = cs.stone_idx[take]
        self.i = cs.i[take]
        self.j = cs.j[take]
        self.k = cs.k[take]
        self.targets = targets[take].astype(np.float32)
        self._volumes: list[np.memmap] | None = None

    def __len__(self) -> int:
        return len(self.stone_idx)

    def _ensure_volumes(self):
        if self._volumes is None:
            self._volumes = [
                np.memmap(c.raw_path, dtype=np.uint8, mode="r", shape=c.shape)
                for c in STONES
            ]

    def __getitem__(self, idx: int):
        self._ensure_volumes()
        s = int(self.stone_idx[idx])
        i = int(self.i[idx]); j = int(self.j[idx]); k = int(self.k[idx])
        sub = np.asarray(
            self._volumes[s][i:i + CHUNK_SIZE,
                              j:j + CHUNK_SIZE,
                              k:k + CHUNK_SIZE]
        ).copy()
        if sub.dtype != np.bool_ and sub.max() > 1:
            sub = (sub > 0).astype(np.uint8)
        x = torch.from_numpy(sub.astype(np.float32, copy=False))[None]
        y = torch.from_numpy(self.targets[idx])
        return x, y

# chunk_decode_2.py
# -----------------------------------------------------------------------------
# Dimension-agnostic chunked (tiled) decode for VAE decoders using Strategy B:
# multi-stage, halo-propagating streaming with CPU stage buffers + periodic BCs.
#
# This is a refactored version of chunk_decode.py that supports both 2D and 3D.
#
# WHAT THIS FILE DOES (HIGH LEVEL):
#   1) Decode a large latent tensor z_latent: [B, C, *spatial] through a decoder
#      in tiles to avoid OOM.
#   2) Split work into "stages" (S0..SN). Each stage is a contiguous chunk of
#      decoder layers.
#   3) For each stage, build stage output piece-by-piece on CPU: read minimal
#      input with required halo, run only this stage on GPU, crop valid center,
#      write to CPU buffer, free GPU memory.
#   4) Periodicity (optional): wrap at boundaries using periodic_getitem.
#
# TENSOR LAYOUT:
#   - 2D: [B, C, H, W]
#   - 3D: [B, C, H, W, D]
#   - Spatial dimensions are always the trailing dimensions after [B, C].
#   - Parameters like chunk_latent use the same axis order as the tensor.
#
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass
import itertools

import numpy as np
import tempfile
import os

import torch

from diffsci2.torchutils import periodic_getitem
from diffsci2.nets.cached_norms import (
    convert_to_cached_norms,
    set_all_norms_mode,
    clear_all_norm_caches,
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _device_of(module: torch.nn.Module) -> torch.device:
    """Return device of module's first parameter, or CUDA if available."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_of(module: torch.nn.Module) -> torch.dtype:
    """Return dtype of module's first parameter, or float32."""
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def normalize_tuple(
    v: Union[int, Tuple, List],
    ndim: int,
    name: str
) -> Tuple[int, ...]:
    """
    Normalize an int or tuple/list to an ndim-tuple of ints.

    Examples:
        normalize_tuple(32, 2, "chunk") -> (32, 32)
        normalize_tuple([16, 32], 2, "chunk") -> (16, 32)
        normalize_tuple(32, 3, "chunk") -> (32, 32, 32)
    """
    if isinstance(v, int):
        return (v,) * ndim
    if isinstance(v, (tuple, list)) and len(v) == ndim:
        return tuple(int(x) for x in v)
    raise ValueError(f"{name} must be int or {ndim}-tuple/list. Got: {v!r}")


def normalize_bool_tuple(
    v: Union[bool, Tuple[bool, ...], List[bool]],
    ndim: int,
    name: str
) -> Tuple[bool, ...]:
    """Normalize a bool or tuple/list of bools to an ndim-tuple."""
    if isinstance(v, bool):
        return (v,) * ndim
    if isinstance(v, (tuple, list)) and len(v) == ndim:
        return tuple(bool(x) for x in v)
    raise ValueError(f"{name} must be bool or {ndim}-tuple/list. Got: {v!r}")


def make_center_spans_1d(L: int, chunk: int, radius: int) -> List[Tuple[int, int]]:
    """
    Decide center spans along one axis (in LATENT units) for Stage-0 tiling.

    Args:
        L: axis length in latent units
        chunk: desired Stage-0 tile extent including halos
        radius: Stage-0 halo radius (latent units)

    Returns:
        List of (start, end) pairs that partition [0, L).
    """
    if chunk >= L:
        return [(0, L)]

    valid = chunk - 2 * radius
    if valid <= 0:
        valid = 1

    spans = []
    pos = 0
    while pos < L:
        end = min(pos + valid, L)
        spans.append((pos, end))
        pos = end

    return spans


def iterate_nd_tiles(
    spans_per_axis: List[List[Tuple[int, int]]]
) -> Iterator[Tuple[Tuple[int, int], ...]]:
    """
    Iterate over all tile combinations across N dimensions.

    Args:
        spans_per_axis: List of span lists, one per spatial axis.
                       E.g., for 2D: [spans_H, spans_W]

    Yields:
        Tuples of (start, end) pairs, one per axis.
        E.g., for 2D: ((h0, h1), (w0, w1))
    """
    for combo in itertools.product(*spans_per_axis):
        yield combo


# ============================================================================
# RF AND SCALE COMPUTATION
# ============================================================================

def compute_stage_radii_and_scales(decoder) -> Tuple[List[int], List[int]]:
    """
    Compute per-stage cumulative RF radii and spatial scales.

    Returns:
        radii_latent: RF radius (in latent units) after each stage
        scales_after: Total spatial scale after each stage
    """
    cfg = decoder.config

    # Check for attention (makes exact chunking impossible)
    if cfg.has_mid_attn or len(cfg.attn_resolutions) > 0:
        raise NotImplementedError(
            "Chunked decoding requires NO attention in the decoder. "
            f"Found has_mid_attn={cfg.has_mid_attn}, "
            f"attn_resolutions={cfg.attn_resolutions}"
        )

    info = decoder.calculate_receptive_field()
    rf_per_block = int(info["rf_per_block"])
    rf_mid = int(info["rf_after_middle"])
    rf_final = int(info["rf_latent"])

    num_res = int(cfg.num_resolutions)
    num_res_blocks = int(cfg.num_res_blocks)
    per_up_rf = (num_res_blocks + 1) * rf_per_block

    # Cumulative radii after each stage
    radii: List[int] = [rf_mid // 2]  # After S0
    for s in range(1, num_res):
        rf_s = rf_mid + per_up_rf * s
        radii.append(rf_s // 2)
    radii.append(rf_final // 2)  # Final stage

    # Total scale after each stage
    scales: List[int] = [1]  # S0 stays at latent resolution
    for s in range(1, num_res):
        scales.append(2 ** s)
    scales.append(2 ** max(0, num_res - 1))  # Final doesn't upsample more

    return radii, scales


def compute_delta_radii(radii_latent: List[int]) -> List[int]:
    """Compute per-stage local radii (what each stage adds to RF)."""
    delta = []
    for s in range(len(radii_latent)):
        prev = radii_latent[s - 1] if s > 0 else 0
        delta.append(max(0, radii_latent[s] - prev))
    return delta


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunked decoding."""
    device: torch.device
    dtype: torch.dtype
    ndim: int                           # 2 or 3 spatial dimensions
    batch_size: int
    z_channels: int
    spatial_shape: Tuple[int, ...]      # (H, W) or (H, W, D)
    chunk_latent: Tuple[int, ...]       # Chunk sizes per axis
    num_stages: int
    radii_latent: List[int]
    scales_after: List[int]
    delta_r_lat: List[int]
    spans_per_axis: List[List[Tuple[int, int]]]
    caps: Tuple[Optional[int], ...]     # Max output size per axis (or None)
    periodic: Tuple[bool, ...]          # Periodicity per axis
    debug: int                          # 0=none, 1=per-stage, 2=per-tile, 3=memory tracking


# ============================================================================
# DEBUG / MEMORY TRACKING UTILITIES
# ============================================================================

def _format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if num_bytes < 0:
        return f"-{_format_bytes(-num_bytes)}"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _get_tensor_memory(tensor: torch.Tensor) -> int:
    """Get memory footprint of a tensor in bytes."""
    return tensor.element_size() * tensor.numel()


class MemoryTracker:
    """
    Tracks GPU and CPU memory usage throughout chunked decoding.

    Usage:
        tracker = MemoryTracker(device, enabled=debug>=3)
        tracker.checkpoint("before allocation")
        # ... do something ...
        tracker.checkpoint("after allocation")
        tracker.print_summary()
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.checkpoints: List[Tuple[str, dict]] = []
        self.peak_gpu_allocated = 0
        self.peak_gpu_reserved = 0

        if self.enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _get_gpu_stats(self) -> dict:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)

        # Get total GPU memory
        props = torch.cuda.get_device_properties(self.device)
        total = props.total_memory
        free = total - reserved

        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total,
        }

    def _get_cpu_stats(self) -> dict:
        """Get current CPU/process memory statistics."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss": mem_info.rss,  # Resident Set Size
                "vms": mem_info.vms,  # Virtual Memory Size
            }
        except ImportError:
            return {"rss": 0, "vms": 0}

    def checkpoint(self, label: str, tensor: Optional[torch.Tensor] = None):
        """Record a memory checkpoint with optional tensor size annotation."""
        if not self.enabled:
            return

        gpu_stats = self._get_gpu_stats()
        cpu_stats = self._get_cpu_stats()

        # Track peaks
        self.peak_gpu_allocated = max(self.peak_gpu_allocated, gpu_stats["allocated"])
        self.peak_gpu_reserved = max(self.peak_gpu_reserved, gpu_stats["reserved"])

        tensor_info = None
        if tensor is not None:
            tensor_info = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": _get_tensor_memory(tensor),
            }

        self.checkpoints.append((label, {
            "gpu": gpu_stats,
            "cpu": cpu_stats,
            "tensor": tensor_info,
        }))

    def print_checkpoint(self, label: str, tensor: Optional[torch.Tensor] = None):
        """Record checkpoint and immediately print it."""
        if not self.enabled:
            return

        self.checkpoint(label, tensor)

        # Get the checkpoint we just recorded
        _, stats = self.checkpoints[-1]
        gpu = stats["gpu"]
        cpu = stats["cpu"]
        tensor_info = stats["tensor"]

        # Calculate delta from previous checkpoint
        delta_str = ""
        if len(self.checkpoints) > 1:
            _, prev_stats = self.checkpoints[-2]
            delta_alloc = gpu["allocated"] - prev_stats["gpu"]["allocated"]
            delta_rss = cpu["rss"] - prev_stats["cpu"]["rss"]
            if delta_alloc != 0 or delta_rss != 0:
                delta_str = f" (Δ GPU: {_format_bytes(delta_alloc):>10}, Δ CPU: {_format_bytes(delta_rss):>10})"

        # Format tensor info
        tensor_str = ""
        if tensor_info:
            tensor_str = f" | Tensor: {tensor_info['shape']} {tensor_info['dtype']} = {_format_bytes(tensor_info['bytes'])}"

        print(f"      [MEM] {label:40s} | GPU: {_format_bytes(gpu['allocated']):>10} / {_format_bytes(gpu['reserved']):>10} | CPU RSS: {_format_bytes(cpu['rss']):>10}{delta_str}{tensor_str}")

    def print_stage_summary(self, stage: int):
        """Print memory summary for a stage."""
        if not self.enabled:
            return

        gpu_stats = self._get_gpu_stats()

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_alloc = self.peak_gpu_allocated
            peak_reserved = self.peak_gpu_reserved

        print(f"      ╔{'═'*70}╗")
        print(f"      ║ STAGE {stage} MEMORY SUMMARY{' '*47}║")
        print(f"      ╠{'═'*70}╣")
        print(f"      ║ GPU Allocated: {_format_bytes(gpu_stats['allocated']):>12} │ Peak: {_format_bytes(peak_alloc):>12}{' '*24}║")
        print(f"      ║ GPU Reserved:  {_format_bytes(gpu_stats['reserved']):>12} │ Peak: {_format_bytes(peak_reserved):>12}{' '*24}║")
        print(f"      ╚{'═'*70}╝")

    def print_final_summary(self):
        """Print final memory summary."""
        if not self.enabled:
            return

        gpu_stats = self._get_gpu_stats()
        cpu_stats = self._get_cpu_stats()

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_alloc = self.peak_gpu_allocated
            peak_reserved = self.peak_gpu_reserved

        print()
        print(f"  ╔{'═'*76}╗")
        print(f"  ║ FINAL MEMORY SUMMARY{' '*55}║")
        print(f"  ╠{'═'*76}╣")
        print(f"  ║ GPU Current Allocated: {_format_bytes(gpu_stats['allocated']):>12}{' '*39}║")
        print(f"  ║ GPU Peak Allocated:    {_format_bytes(peak_alloc):>12}{' '*39}║")
        print(f"  ║ GPU Reserved:          {_format_bytes(gpu_stats['reserved']):>12}{' '*39}║")
        print(f"  ║ GPU Peak Reserved:     {_format_bytes(peak_reserved):>12}{' '*39}║")
        print(f"  ╠{'─'*76}╣")
        print(f"  ║ CPU RSS:               {_format_bytes(cpu_stats['rss']):>12}{' '*39}║")
        print(f"  ║ CPU VMS:               {_format_bytes(cpu_stats['vms']):>12}{' '*39}║")
        print(f"  ╚{'═'*76}╝")


@dataclass
class TileSpec:
    """Specification of a tile's position in latent coordinates."""
    ranges: Tuple[Tuple[int, int], ...]  # (start, end) per axis


@dataclass
class ReadWindow:
    """Read window for fetching input to a stage."""
    lat_ranges: Tuple[Tuple[int, int], ...]   # Ranges in latent units
    src_ranges: Tuple[Tuple[int, int], ...]   # Ranges in source (prev stage) units


@dataclass
class CropSpec:
    """How to crop stage output and where to write it."""
    tile_slices: Tuple[slice, ...]   # Slices into y_tile (stage output)
    dest_ranges: Tuple[Tuple[int, int], ...]  # Global coords in dest buffer


# ============================================================================
# CPU STAGE BUFFER
# ============================================================================

class CPUStageBuffer:
    """
    CPU tensor buffer for an entire stage's output.
    Shape: [B, C, *spatial]
    Supports periodic reads.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        ndim: int,
        pin_memory: bool = False,
    ):
        # Note: pin_memory=True adds ~18s overhead for large buffers (32GB+)
        # due to page-locking. Only enable if transfers show improvement.
        if pin_memory and torch.cuda.is_available():
            self.tensor = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            self.tensor.zero_()
        else:
            self.tensor = torch.zeros(shape, dtype=dtype, device="cpu")
        self.ndim = ndim  # Number of spatial dimensions

    def write_block(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        tile: torch.Tensor
    ):
        """
        Write tile to destination coordinates.

        Args:
            ranges: (start, end) per spatial axis
            tile: Tensor to write, shape [B, C, *spatial_tile]
        """
        slices = [slice(None), slice(None)]  # B, C
        for (s, e) in ranges:
            slices.append(slice(s, e))
        # Copy tile to CPU buffer (synchronous for unpinned memory)
        self.tensor[tuple(slices)].copy_(tile.detach().cpu())

    def read_block_periodic(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Read block with periodic wrapping.

        Args:
            ranges: (start, end) per spatial axis (may be out of bounds)
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor [B, C, *spatial_tile] on target device
        """
        B, C = self.tensor.shape[:2]
        spatial_shape = self.tensor.shape[2:]

        # Flatten [B, C, *spatial] -> [B*C, *spatial] for periodic_getitem
        flat = self.tensor.reshape(B * C, *spatial_shape)

        # Build slices for periodic_getitem
        indices = [slice(None)]  # Keep B*C dimension
        for (s, e) in ranges:
            indices.append(slice(s, e, None))

        # Periodic fetch
        sub = periodic_getitem(flat, *indices)

        # Reshape back to [B, C, *spatial_tile]
        new_spatial = sub.shape[1:]
        result = sub.reshape(B, C, *new_spatial)

        return result.to(device=device, dtype=dtype, non_blocking=True).contiguous()


# Torch dtype -> numpy dtype mapping for mmap
_TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.bfloat16: np.float32,  # numpy has no bfloat16; store as float32
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
}


class MmapStageBuffer:
    """
    Disk-backed stage buffer using numpy memory-mapped files.

    Same interface as CPUStageBuffer but stores data on disk, letting the OS
    page data in/out as needed. This avoids allocating multi-TB CPU RAM buffers
    for very large volumes (e.g. 2304^3 with 64 channels).

    Parameters
    ----------
    shape : tuple of int
        Buffer shape [B, C, *spatial].
    dtype : torch.dtype
        Data type for the buffer.
    ndim : int
        Number of spatial dimensions (2 or 3).
    offload_dir : str or None
        Directory for the mmap file. If None, uses system temp directory.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        ndim: int,
        offload_dir: Optional[str] = None,
    ):
        self.ndim = ndim
        self._shape = shape
        self._torch_dtype = dtype

        # Handle bfloat16: numpy doesn't support it, so we use float32 on disk
        self._needs_bf16_cast = (dtype == torch.bfloat16)
        np_dtype = _TORCH_TO_NUMPY_DTYPE.get(dtype)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype for mmap: {dtype}")
        self._np_dtype = np_dtype

        # Create temp file
        if offload_dir is not None:
            os.makedirs(offload_dir, exist_ok=True)
        fd, self._filepath = tempfile.mkstemp(
            suffix='.mmap', prefix='stage_buf_', dir=offload_dir
        )
        os.close(fd)

        # Create memory-mapped array (zero-initialized via mode='w+')
        self._mmap = np.memmap(
            self._filepath, dtype=self._np_dtype, mode='w+', shape=shape
        )

        # Create a torch view over the mmap (zero-copy)
        self.tensor = torch.from_numpy(self._mmap)
        if self._needs_bf16_cast:
            # tensor is float32 on disk; callers reading .tensor get float32.
            # write_block and read_block_periodic handle the cast.
            pass

    def write_block(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        tile: torch.Tensor
    ):
        """
        Write tile to destination coordinates (same interface as CPUStageBuffer).
        """
        slices = [slice(None), slice(None)]  # B, C
        for (s, e) in ranges:
            slices.append(slice(s, e))

        src = tile.detach().cpu()
        if self._needs_bf16_cast:
            src = src.float()
        self.tensor[tuple(slices)].copy_(src)

    def read_block_periodic(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Read block with periodic wrapping (same interface as CPUStageBuffer).
        """
        B, C = self.tensor.shape[:2]
        spatial_shape = self.tensor.shape[2:]

        # Flatten [B, C, *spatial] -> [B*C, *spatial] for periodic_getitem
        flat = self.tensor.reshape(B * C, *spatial_shape)

        # Build slices for periodic_getitem
        indices = [slice(None)]  # Keep B*C dimension
        for (s, e) in ranges:
            indices.append(slice(s, e, None))

        # Periodic fetch
        sub = periodic_getitem(flat, *indices)

        # Reshape back to [B, C, *spatial_tile]
        new_spatial = sub.shape[1:]
        result = sub.reshape(B, C, *new_spatial)

        return result.to(device=device, dtype=dtype, non_blocking=True).contiguous()

    def close(self):
        """Flush and release the mmap, then delete the backing file."""
        if hasattr(self, '_mmap') and self._mmap is not None:
            # Drop the torch view first
            self.tensor = None
            # Flush and delete the mmap
            self._mmap.flush()
            del self._mmap
            self._mmap = None
        if hasattr(self, '_filepath') and self._filepath and os.path.exists(self._filepath):
            os.remove(self._filepath)
            self._filepath = None

    def __del__(self):
        self.close()

    def __repr__(self):
        size_bytes = int(np.prod(self._shape)) * np.dtype(self._np_dtype).itemsize
        return (
            f"MmapStageBuffer(shape={self._shape}, dtype={self._torch_dtype}, "
            f"file={self._filepath}, size={_format_bytes(size_bytes)})"
        )


def _make_stage_buffer(shape, dtype, ndim, use_disk_offload=False, disk_offload_dir=None):
    """Factory: create either a CPUStageBuffer or MmapStageBuffer."""
    if use_disk_offload:
        return MmapStageBuffer(shape, dtype, ndim, offload_dir=disk_offload_dir)
    else:
        return CPUStageBuffer(shape, dtype, ndim)


def _close_stage_buffer(buf):
    """Close an MmapStageBuffer (no-op for CPUStageBuffer)."""
    if isinstance(buf, MmapStageBuffer):
        buf.close()


# ============================================================================
# GEOMETRY COMPUTATIONS
# ============================================================================

def compute_read_window(
    tile: TileSpec,
    stage_local_lat: int,
    src_scale: int,
    spatial_shape: Tuple[int, ...],
    periodic: Tuple[bool, ...]
) -> ReadWindow:
    """
    Compute read window for a tile at a given stage.

    Args:
        tile: The tile specification (center ranges in latent units)
        stage_local_lat: How many latent cells this stage adds to RF
        src_scale: Scale factor from latent to source (prev stage) units
        spatial_shape: Full spatial shape in latent units
        periodic: Periodicity flags per axis

    Returns:
        ReadWindow with latent and source coordinate ranges
    """
    lat_ranges = []
    src_ranges = []

    for i, ((s, e), L, per) in enumerate(zip(tile.ranges, spatial_shape, periodic)):
        if per:
            # Allow negative/over-range; periodic_getitem will wrap
            rs = s - stage_local_lat
            re = e + stage_local_lat
        else:
            # Clamp to valid range
            rs = max(0, s - stage_local_lat)
            re = min(L, e + stage_local_lat)

        lat_ranges.append((rs, re))
        src_ranges.append((rs * src_scale, re * src_scale))

    return ReadWindow(
        lat_ranges=tuple(lat_ranges),
        src_ranges=tuple(src_ranges)
    )


def compute_crop_spec(
    tile: TileSpec,
    read_win: ReadWindow,
    src_scale: int,
    dest_scale: int,
    up_factor: int,
    spatial_shape: Tuple[int, ...],
    periodic: Tuple[bool, ...]
) -> CropSpec:
    """
    Compute how to crop stage output and where to write it.

    Args:
        tile: Tile specification (center in latent units)
        read_win: Read window used to fetch input
        src_scale: Scale of source (prev stage) relative to latent
        dest_scale: Scale of destination (this stage) relative to latent
        up_factor: Upsampling factor of this stage (dest_scale / src_scale)
        spatial_shape: Full spatial shape in latent units
        periodic: Periodicity flags per axis

    Returns:
        CropSpec with tile slices and destination ranges
    """
    tile_slices = []
    dest_ranges = []

    for i, ((cs, ce), (rs, _), L, per) in enumerate(
        zip(tile.ranges, read_win.lat_ranges, spatial_shape, periodic)
    ):
        # Compute offset within the read window
        if per:
            left_lat = (cs - rs) % L
            right_lat = (ce - rs) % L
            # Handle wrap-around case
            if right_lat <= left_lat and ce != cs:
                right_lat = left_lat + (ce - cs)
        else:
            left_lat = cs - rs
            right_lat = ce - rs

        # Scale to source and then to output units
        left_src = left_lat * src_scale
        right_src = right_lat * src_scale
        y_start = left_src * up_factor
        y_end = right_src * up_factor

        # Global destination coords
        g0 = cs * dest_scale
        g1 = ce * dest_scale

        # Verify consistency
        assert (y_end - y_start) == (g1 - g0), \
            f"Length mismatch on axis {i}: tile={y_end-y_start}, dest={g1-g0}"

        tile_slices.append(slice(y_start, y_end))
        dest_ranges.append((g0, g1))

    return CropSpec(
        tile_slices=tuple(tile_slices),
        dest_ranges=tuple(dest_ranges)
    )


def generate_sub_tiles(
    tile: TileSpec,
    dest_scale: int,
    caps: Tuple[Optional[int], ...]
) -> List[TileSpec]:
    """
    Split a center tile into sub-tiles to respect output size caps.

    Args:
        tile: Original tile in latent units
        dest_scale: Scale of destination stage relative to latent
        caps: Maximum output size per axis (or None for no limit)

    Returns:
        List of sub-tile specifications
    """
    # Compute sub-tile sizes per axis
    sub_sizes = []
    for (s, e), cap in zip(tile.ranges, caps):
        length = e - s
        if cap is None:
            sub_sizes.append(length)
        else:
            # sub_len * dest_scale <= cap
            sub_lat = max(1, min(length, cap // max(dest_scale, 1)))
            sub_sizes.append(sub_lat)

    # Generate sub-tile ranges per axis
    ranges_per_axis = []
    for (s, e), sub_size in zip(tile.ranges, sub_sizes):
        axis_ranges = []
        pos = s
        while pos < e:
            end = min(pos + sub_size, e)
            axis_ranges.append((pos, end))
            pos = end
        ranges_per_axis.append(axis_ranges)

    # Combine all axes
    sub_tiles = []
    for combo in itertools.product(*ranges_per_axis):
        sub_tiles.append(TileSpec(ranges=combo))

    return sub_tiles


# ============================================================================
# STAGE RUNNERS FOR VAE DECODER
# ============================================================================

@torch.inference_mode()
def run_vae_stage0(
    decoder,
    z: torch.Tensor,
    temb: Optional[torch.Tensor]
) -> torch.Tensor:
    """Run Stage 0: post_quant_conv -> conv_in -> mid blocks."""
    h = decoder.post_quant_conv(z)
    h = decoder.conv_in(h)
    h = decoder.mid.block_1(h, temb)
    if hasattr(decoder.mid, "attn_1"):
        h = decoder.mid.attn_1(h)
    h = decoder.mid.block_2(h, temb)
    return h


@torch.inference_mode()
def run_vae_up_stage(
    decoder,
    x: torch.Tensor,
    level_index: int,
    temb: Optional[torch.Tensor]
) -> torch.Tensor:
    """Run an upsampling stage: up[level_index].blocks + upsample."""
    up = decoder.up[level_index]
    h = x
    for i in range(len(up.block)):
        h = up.block[i](h, temb)
        if len(up.attn) > i:
            h = up.attn[i](h)
    if level_index != 0:
        h = up.upsample(h)
    return h


@torch.inference_mode()
def run_vae_final_stage(
    decoder,
    x: torch.Tensor,
    temb: Optional[torch.Tensor]
) -> torch.Tensor:
    """Run final stage: up[0].blocks -> norm_out -> swish -> conv_out."""
    up0 = decoder.up[0]
    h = x
    for i in range(len(up0.block)):
        h = up0.block[i](h, temb)
        if len(up0.attn) > i:
            h = up0.attn[i](h)
    h = decoder.norm_out(h)
    h = h * torch.sigmoid(h)  # swish
    h = decoder.conv_out(h)
    if getattr(decoder.config, "tanh_out", False):
        h = torch.tanh(h)
    return h


def make_vae_stage_runner(
    decoder,
    stage_idx: int,
    num_stages: int,
    temb: Optional[torch.Tensor]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a stage runner function for a specific stage.

    Args:
        decoder: VAE decoder module
        stage_idx: Stage index (0 to num_stages-1)
        num_stages: Total number of stages
        temb: Time embedding (optional)

    Returns:
        Callable that takes input tensor and returns stage output
    """
    num_res = int(decoder.config.num_resolutions)

    if stage_idx == 0:
        return lambda x: run_vae_stage0(decoder, x, temb)
    elif 1 <= stage_idx <= (num_res - 1):
        level_index = num_res - stage_idx
        return lambda x: run_vae_up_stage(decoder, x, level_index, temb)
    else:
        return lambda x: run_vae_final_stage(decoder, x, temb)


# ============================================================================
# CACHED NORMALIZATION SUPPORT
# ============================================================================

def prepare_decoder_for_cached_decode(
    decoder,
    inplace: bool = True
):
    """
    Convert a decoder's normalization layers to cached versions.

    This must be called once before using cached norm mode.

    For PixelNorm-based decoders, this is a no-op (PixelNorm has spatial
    RF=1, so cached norms is unnecessary).

    Args:
        decoder: VAE decoder module
        inplace: If True, modify decoder in place

    Returns:
        Decoder with cached norm layers (or the unchanged decoder for
        PixelNorm-based models)
    """
    if not _needs_cached_norms(decoder):
        return decoder
    return convert_to_cached_norms(decoder, inplace=inplace)


def _needs_cached_norms(model) -> bool:
    """Return True iff `model` uses GroupNorm (the only norm where cached-norm
    calibration is actually necessary). PixelNorm has spatial RF=1, so cached
    norms is a no-op — we silently skip the conversion and any per-tile
    calibration for `norm_type='pixel'` VAEs."""
    return getattr(getattr(model, "config", None), "norm_type", "group") == "group"


def _has_cached_norms(model: torch.nn.Module) -> bool:
    """Check if model has any cached norm layers."""
    from diffsci2.nets.cached_norms import (
        CachedGroupNorm, CachedGroupRMSNorm, CachedGroupLNorm
    )
    for module in model.modules():
        if isinstance(module, (CachedGroupNorm, CachedGroupRMSNorm, CachedGroupLNorm)):
            return True
    return False


# ============================================================================
# MULTI-GPU PARALLEL PROCESSING SUPPORT
# ============================================================================

import copy
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Module-level cache for replicated decoders.
# Key: (id(original_decoder), tuple of device strings)
# Value: (weak_ref to original decoder, list of decoder copies)
_decoder_replica_cache: dict = {}


def get_cached_decoder_replicas(
    decoder: torch.nn.Module,
    devices: List[torch.device],
    debug: int = 0,
) -> List[torch.nn.Module]:
    """
    Get or create cached decoder replicas for multiple devices.

    This avoids expensive deep-copy operations on repeated calls with the same
    decoder and devices. The cache uses a weak reference to the original decoder,
    so cache entries are automatically invalidated when the decoder is garbage
    collected or modified.

    Args:
        decoder: Source decoder module
        devices: List of target devices
        debug: Debug level for logging

    Returns:
        List of decoder copies, one per device
    """
    # Create cache key
    decoder_id = id(decoder)
    device_tuple = tuple(str(d) for d in devices)
    cache_key = (decoder_id, device_tuple)

    # Check if we have a valid cached entry
    if cache_key in _decoder_replica_cache:
        weak_ref, cached_decoders = _decoder_replica_cache[cache_key]
        original = weak_ref()
        if original is decoder:
            # Cache hit: original decoder still exists and matches
            if debug >= 1:
                print("  [Cache HIT] Reusing cached decoder replicas")
            return cached_decoders
        else:
            # Weak ref is dead or points to different object, remove stale entry
            del _decoder_replica_cache[cache_key]

    # Cache miss: create new replicas
    if debug >= 1:
        print("  [Cache MISS] Creating new decoder replicas...")

    decoders = replicate_decoder_to_devices(decoder, devices)

    # Store in cache with weak reference to original
    weak_ref = weakref.ref(decoder)
    _decoder_replica_cache[cache_key] = (weak_ref, decoders)

    return decoders


def clear_decoder_replica_cache():
    """Clear all cached decoder replicas. Call this if you modify the decoder."""
    global _decoder_replica_cache
    _decoder_replica_cache.clear()


def replicate_decoder_to_devices(
    decoder: torch.nn.Module,
    devices: List[torch.device],
) -> List[torch.nn.Module]:
    """
    Create copies of decoder on multiple devices.

    Args:
        decoder: Source decoder (will be moved to devices[0])
        devices: List of target devices

    Returns:
        List of decoder copies, one per device
    """
    decoders = []
    for i, device in enumerate(devices):
        if i == 0:
            # Move original to first device
            dec = decoder.to(device)
        else:
            # Deep copy for other devices
            dec = copy.deepcopy(decoder).to(device)
        dec.eval()
        decoders.append(dec)
    return decoders


def sync_cached_norms_across_devices(
    decoders: List[torch.nn.Module],
    source_idx: int = 0,
):
    """
    Sync cached normalization statistics from source decoder to all others.

    After the first tile is processed on device 0, this copies the cached
    stats to all other device copies so they use identical normalization.

    Args:
        decoders: List of decoder copies on different devices
        source_idx: Index of source decoder with cached stats (default: 0)
    """
    from diffsci2.nets.cached_norms import (
        CachedGroupNorm, CachedGroupRMSNorm, CachedGroupLNorm
    )

    source = decoders[source_idx]
    targets = [d for i, d in enumerate(decoders) if i != source_idx]

    # Get all cached norm layers from source
    source_norms = {}
    for name, module in source.named_modules():
        if isinstance(module, (CachedGroupNorm, CachedGroupRMSNorm, CachedGroupLNorm)):
            source_norms[name] = module

    # Copy cached stats to each target
    for target in targets:
        target_device = next(target.parameters()).device
        for name, module in target.named_modules():
            if name in source_norms:
                src_module = source_norms[name]
                # Copy cached stats based on norm type
                if isinstance(module, CachedGroupRMSNorm):
                    if src_module._cached_rms is not None:
                        module._cached_rms = src_module._cached_rms.to(target_device)
                elif isinstance(module, (CachedGroupLNorm, CachedGroupNorm)):
                    if hasattr(src_module, '_cached_mean') and src_module._cached_mean is not None:
                        module._cached_mean = src_module._cached_mean.to(target_device)
                    if hasattr(src_module, '_cached_std') and src_module._cached_std is not None:
                        module._cached_std = src_module._cached_std.to(target_device)
                    if hasattr(src_module, '_cached_var') and src_module._cached_var is not None:
                        module._cached_var = src_module._cached_var.to(target_device)
                # Copy mode
                module._mode = src_module._mode


class ParallelTileProcessor:
    """
    Processes tiles in parallel across multiple GPU devices.

    Each device has its own decoder copy. Tiles are distributed round-robin
    across devices and processed concurrently using a thread pool.
    """

    def __init__(
        self,
        decoders: List[torch.nn.Module],
        devices: List[torch.device],
        stage_runners: List[List[Callable]],  # [device_idx][stage_idx] -> runner
        max_workers: Optional[int] = None,
        aggressive_cleaning: bool = False,
    ):
        self.decoders = decoders
        self.devices = devices
        self.stage_runners = stage_runners
        self.num_devices = len(devices)
        self.max_workers = max_workers or self.num_devices
        self.aggressive_cleaning = aggressive_cleaning

        # Thread-local storage for CUDA streams
        self._local = threading.local()

        # Note: No write lock needed because tiles write to non-overlapping regions.
        # Each tile has its own destination range (crop.dest_ranges), so parallel
        # writes to different parts of the buffer are safe.

    def _get_stream(self, device_idx: int) -> torch.cuda.Stream:
        """Get or create CUDA stream for this thread/device."""
        if not hasattr(self._local, 'streams'):
            self._local.streams = {}
        if device_idx not in self._local.streams:
            self._local.streams[device_idx] = torch.cuda.Stream(self.devices[device_idx])
        return self._local.streams[device_idx]

    def process_tile(
        self,
        device_idx: int,
        stage_idx: int,
        x_in: torch.Tensor,
        crop: CropSpec,
        dest_buf: CPUStageBuffer,
    ) -> None:
        """
        Process a single tile on specified device (legacy sync-per-tile version).

        Args:
            device_idx: Which device to use
            stage_idx: Current stage index
            x_in: Input tile (on CPU)
            crop: Crop specification for output
            dest_buf: Destination CPU buffer
        """
        # NOTE: inference_mode is thread-local, so we must enter it in each thread.
        # The main thread runs with @torch.inference_mode() decorator, which means
        # the CPU buffers are inference tensors. Without entering inference_mode here,
        # inplace updates (copy_) to those buffers would fail.
        with torch.inference_mode():
            device = self.devices[device_idx]
            runner = self.stage_runners[device_idx][stage_idx]

            # Use dedicated stream for this device
            stream = self._get_stream(device_idx)

            with torch.cuda.stream(stream):
                # Transfer to GPU
                x_gpu = x_in.to(device=device, non_blocking=True)

                # Run stage
                y_tile = runner(x_gpu)

                # Crop valid center
                slices = [slice(None), slice(None)]  # B, C
                slices.extend(crop.tile_slices)
                y_center = y_tile[tuple(slices)]

                # Clean up intermediate tensors
                del y_tile, x_gpu

            # Synchronize stream to ensure GPU compute is done
            stream.synchronize()

            # Write to buffer - no lock needed, tiles write to non-overlapping regions
            dest_buf.write_block(crop.dest_ranges, y_center)

            # Clean up GPU tensor reference
            del y_center

            # Aggressive cleaning: empty cache after every tile (slower but minimal memory)
            if self.aggressive_cleaning:
                torch.cuda.empty_cache()

    def _compute_tile_on_gpu(
        self,
        device_idx: int,
        stage_idx: int,
        x_in: torch.Tensor,
        crop: CropSpec,
    ) -> Tuple[torch.Tensor, CropSpec, int]:
        """
        Compute a single tile on GPU and sync before returning.

        Returns:
            Tuple of (y_center on GPU ready for CPU transfer, crop spec, device_idx)
        """
        with torch.inference_mode():
            device = self.devices[device_idx]
            runner = self.stage_runners[device_idx][stage_idx]
            stream = self._get_stream(device_idx)

            with torch.cuda.stream(stream):
                # Transfer to GPU
                x_gpu = x_in.to(device=device, non_blocking=True)

                # Run stage
                y_tile = runner(x_gpu)

                # Crop valid center
                slices = [slice(None), slice(None)]  # B, C
                slices.extend(crop.tile_slices)
                y_center = y_tile[tuple(slices)].contiguous()

                # Clean up intermediate tensors
                del y_tile, x_gpu

            # Sync stream before returning to ensure GPU compute is complete
            # This is needed because streams are thread-local and can't be
            # synced from the main thread
            stream.synchronize()

            return (y_center, crop, device_idx)

    def process_tiles_parallel(
        self,
        stage_idx: int,
        tile_jobs: List[Tuple[torch.Tensor, CropSpec]],  # [(x_in, crop), ...]
        dest_buf: CPUStageBuffer,
        debug: int = 0,
    ) -> None:
        """
        Process multiple tiles in parallel across devices with batched sync.

        Uses a two-phase approach per batch:
        1. Submit GPU work for num_devices tiles (one per GPU)
        2. Sync all streams, write results to CPU
        3. Repeat for next batch

        Batching by num_devices ensures we don't exceed GPU memory while still
        allowing full parallel utilization of all GPUs.

        Args:
            stage_idx: Current stage index
            tile_jobs: List of (input_tensor, crop_spec) tuples
            dest_buf: Destination CPU buffer
            debug: Debug level (0=none, 1=per-stage, 2=per-tile)
        """
        if not tile_jobs:
            return

        # Process in batches of num_devices tiles (one tile per GPU per batch)
        batch_size = self.num_devices
        num_batches = (len(tile_jobs) + batch_size - 1) // batch_size

        if debug >= 2:
            device_counts = [0] * self.num_devices
            for i in range(len(tile_jobs)):
                device_counts[i % self.num_devices] += 1
            print(f"    ┌─ Parallel batch dispatch ─────────────────────────────────")
            print(f"    │ Total tiles: {len(tile_jobs)}, Batches: {num_batches}")
            print(f"    │ Tiles per device: {device_counts}")
            print(f"    └{'─'*55}")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tile_jobs))
            batch_jobs = tile_jobs[batch_start:batch_end]

            # Phase 1: Submit batch to GPUs (no sync)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i, (x_in, crop) in enumerate(batch_jobs):
                    device_idx = i % self.num_devices
                    future = executor.submit(
                        self._compute_tile_on_gpu,
                        device_idx, stage_idx, x_in, crop
                    )
                    futures.append(future)

                # Collect batch results (GPU tensors)
                gpu_results = []
                for future in as_completed(futures):
                    result = future.result()
                    gpu_results.append(result)

            # Phase 2: Write batch results to CPU buffer
            # (streams already synced in _compute_tile_on_gpu)
            with torch.inference_mode():
                for y_center, crop, device_idx in gpu_results:
                    dest_buf.write_block(crop.dest_ranges, y_center)
                    del y_center

        # Aggressive cleaning: empty cache once after all batches
        if self.aggressive_cleaning:
            torch.cuda.empty_cache()

        if debug >= 2:
            print(f"    ✓ Completed {len(tile_jobs)} tiles in parallel")


def create_stage_runners_for_devices(
    decoders: List[torch.nn.Module],
    num_stages: int,
    time: Optional[torch.Tensor] = None,
) -> List[List[Callable]]:
    """
    Create stage runner functions for each device.

    Returns:
        List of lists: stage_runners[device_idx][stage_idx] -> callable
    """
    all_runners = []
    for decoder in decoders:
        device_runners = [
            make_vae_stage_runner(decoder, s, num_stages, time)
            for s in range(num_stages)
        ]
        all_runners.append(device_runners)
    return all_runners


# ============================================================================
# CONFIGURATION SETUP
# ============================================================================

def setup_chunk_config(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple, List],
    device: Optional[torch.device],
    max_stage_out_chunk: Optional[Union[int, Tuple, List]],
    periodicity: Union[bool, Tuple[bool, ...], List[bool]],
    debug: int
) -> ChunkConfig:
    """Build configuration for chunked decoding."""

    # Determine spatial dimensions
    ndim = z_latent.dim() - 2  # Subtract batch and channel dims
    if ndim not in (2, 3):
        raise ValueError(f"z_latent must be 4D (2D) or 5D (3D), got {z_latent.dim()}D")

    # Device and dtype
    if device is None:
        device = _device_of(decoder)
    else:
        device = torch.device(device)
    dtype = _dtype_of(decoder)

    # Extract shapes
    batch_size = z_latent.shape[0]
    z_channels = z_latent.shape[1]
    spatial_shape = z_latent.shape[2:]

    # Normalize parameters
    chunk = normalize_tuple(chunk_latent, ndim, "chunk_latent")
    periodic = normalize_bool_tuple(periodicity, ndim, "periodicity")

    if max_stage_out_chunk is None:
        caps = (None,) * ndim
    else:
        caps = normalize_tuple(max_stage_out_chunk, ndim, "max_stage_out_chunk")
        caps = tuple(int(c) if c is not None else None for c in caps)

    # Compute RF info
    radii_latent, scales_after = compute_stage_radii_and_scales(decoder)
    num_stages = int(decoder.config.num_resolutions) + 1
    delta_r_lat = compute_delta_radii(radii_latent)

    # Compute center spans per axis
    spans_per_axis = []
    for i, (L, ch) in enumerate(zip(spatial_shape, chunk)):
        spans = make_center_spans_1d(L, ch, radii_latent[0])
        spans_per_axis.append(spans)

    if debug >= 1:
        total_tiles = 1
        for spans in spans_per_axis:
            total_tiles *= len(spans)

        print()
        print(f"  ┌{'─'*60}┐")
        print(f"  │ CHUNKED DECODE CONFIGURATION{' '*30}│")
        print(f"  ├{'─'*60}┤")
        print(f"  │ Spatial shape:   {str(spatial_shape):20s} ({ndim}D){' '*14}│")
        print(f"  │ Chunk (latent):  {str(chunk):40s}│")
        print(f"  │ Num stages:      {num_stages:<40d}│")
        print(f"  │ Total tiles:     {total_tiles:<40d}│")
        print(f"  │ Periodicity:     {str(periodic):40s}│")
        print(f"  │ Output caps:     {str(caps):40s}│")
        print(f"  ├{'─'*60}┤")
        print(f"  │ Radii (latent):  {str(radii_latent):40s}│")
        print(f"  │ Delta radii:     {str(delta_r_lat):40s}│")
        print(f"  │ Scales after:    {str(scales_after):40s}│")
        print(f"  └{'─'*60}┘")
        if debug >= 2:
            print(f"  Tile spans per axis:")
            for i, spans in enumerate(spans_per_axis):
                print(f"    Axis {i}: {spans}")

    return ChunkConfig(
        device=device,
        dtype=dtype,
        ndim=ndim,
        batch_size=batch_size,
        z_channels=z_channels,
        spatial_shape=spatial_shape,
        chunk_latent=chunk,
        num_stages=num_stages,
        radii_latent=radii_latent,
        scales_after=scales_after,
        delta_r_lat=delta_r_lat,
        spans_per_axis=spans_per_axis,
        caps=caps,
        periodic=periodic,
        debug=debug
    )


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

@torch.inference_mode()
def chunk_decode_strategy_b(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, ...], List[int]],
    *,
    device: Optional[Union[str, torch.device]] = None,
    time: Optional[torch.Tensor] = None,
    debug: int = 0,
    max_stage_out_chunk: Optional[Union[int, Tuple[int, ...], List[int]]] = 128,
    periodicity: Union[bool, Tuple[bool, ...], List[bool]] = False,
    use_cached_norms: bool = False,
    aggressive_cleaning: bool = False,
    max_stages: Optional[int] = None,
    use_disk_offload: bool = False,
    disk_offload_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Dimension-agnostic chunked decode with Strategy B + optional periodic BCs.

    This function decodes a latent tensor through a VAE decoder in tiles,
    using CPU buffers for intermediate stage outputs to minimize GPU memory.

    Args:
        decoder: VAE decoder module (must have calculate_receptive_field method)
        z_latent: Latent tensor [B, C, H, W] (2D) or [B, C, H, W, D] (3D)
        chunk_latent: Tile size in latent units (int or tuple per axis)
        device: Compute device for stage tiles (default: decoder's device)
        time: Optional time embedding tensor
        debug: Debug verbosity level:
               0 = no output
               1 = per-stage summaries
               2 = per-tile details with nice formatting
               3 = full memory tracking (GPU allocated/reserved, CPU RSS, deltas)
        max_stage_out_chunk: Cap per-stage output size (int or tuple per axis)
        periodicity: Enable periodic BCs (bool or tuple per axis)
        use_cached_norms: If True, use first-tile norm caching to eliminate
                         boundary artifacts. Requires decoder to have cached
                         norm layers (call prepare_decoder_for_cached_decode first).
        aggressive_cleaning: If True, call torch.cuda.empty_cache() after every
                            tile (slower but uses minimal GPU memory). Default False
                            only cleans once per stage.
        max_stages: If set, stop after this many stages (for profiling/debugging).
                   Output will be intermediate stage buffer, not final decoded image.
        use_disk_offload: If True, use memory-mapped files on disk instead of CPU
                         RAM for intermediate stage buffers. Essential for very large
                         volumes (e.g. 2304^3) where stage buffers exceed available RAM.
        disk_offload_dir: Directory for mmap temp files when use_disk_offload=True.
                         If None, uses the system temp directory.

    Returns:
        Decoded output tensor [B, C_out, *spatial_out] on CPU

    Note on use_cached_norms:
        When enabled, for each stage:
        1. First tile: compute and cache normalization statistics
        2. Subsequent tiles: use cached statistics
        3. Clear cache before next stage
        This ensures consistent normalization across all tiles within a stage,
        eliminating the "patch" artifacts at tile boundaries.
    """
    # Setup configuration
    cfg = setup_chunk_config(
        decoder, z_latent, chunk_latent, device,
        max_stage_out_chunk, periodicity, debug
    )

    # Check if cached norms are available when requested
    if use_cached_norms and not _has_cached_norms(decoder):
        if not _needs_cached_norms(decoder):
            # PixelNorm-based decoder: cached norms is a no-op. Silently
            # downgrade rather than erroring on a benign request.
            use_cached_norms = False
        else:
            raise RuntimeError(
                "use_cached_norms=True but decoder has no cached norm layers. "
                "Call prepare_decoder_for_cached_decode(decoder) first."
            )

    # Create stage runners
    stage_runners = [
        make_vae_stage_runner(decoder, s, cfg.num_stages, time)
        for s in range(cfg.num_stages)
    ]

    # Stage buffers (CPU tensors or disk-backed mmap)
    stage_bufs = [None] * cfg.num_stages
    prev_scale = 1
    prev_buf = None

    # Save training state
    was_training = decoder.training
    decoder.eval()

    # Initialize memory tracker for debug level 3
    mem_tracker = MemoryTracker(cfg.device, enabled=(cfg.debug >= 3))
    if cfg.debug >= 3:
        mem_tracker.print_checkpoint("Initial state")

    # Determine number of stages to run
    num_stages_to_run = cfg.num_stages
    if max_stages is not None:
        num_stages_to_run = min(max_stages, cfg.num_stages)
        if cfg.debug >= 1:
            print(f"  Limiting to {num_stages_to_run} stages (out of {cfg.num_stages})")

    # Process each stage
    for s in range(num_stages_to_run):
        stage_local_lat = cfg.delta_r_lat[s]
        dest_scale = cfg.scales_after[s]
        src_scale = prev_scale
        up_factor = dest_scale // src_scale

        if cfg.debug >= 1:
            print()
            print(f"  ╔{'═'*60}╗")
            print(f"  ║ STAGE {s}/{num_stages_to_run-1}{' '*50}║")
            print(f"  ╠{'═'*60}╣")
            print(f"  ║ Local RF (latent): {stage_local_lat:<38d}║")
            print(f"  ║ Source scale:      {src_scale:<38d}║")
            print(f"  ║ Dest scale:        {dest_scale:<38d}║")
            print(f"  ║ Upsample factor:   {up_factor:<38d}║")
            print(f"  ╚{'═'*60}╝")
        if cfg.debug >= 3:
            mem_tracker.print_checkpoint(f"Stage {s} start")

        dest_buf = stage_bufs[s]
        dest_created = dest_buf is not None

        # For cached norms: clear cache and set to "cache" mode at start of stage
        # First tile will compute and cache stats, then we switch to "use_cached"
        if use_cached_norms:
            clear_all_norm_caches(decoder)
            set_all_norms_mode(decoder, "cache")
            first_tile_of_stage = True
            if cfg.debug >= 1:
                print(f"    [CachedNorms] Ready to cache from first tile")

        # Iterate over all center tiles
        tile_count = 0
        for center_ranges in iterate_nd_tiles(cfg.spans_per_axis):
            tile = TileSpec(ranges=center_ranges)

            # Split into sub-tiles if needed
            sub_tiles = generate_sub_tiles(tile, dest_scale, cfg.caps)

            for sub_idx, sub_tile in enumerate(sub_tiles):
                # Compute read window
                read_win = compute_read_window(
                    sub_tile, stage_local_lat, src_scale,
                    cfg.spatial_shape, cfg.periodic
                )

                if cfg.debug >= 2:
                    print(f"    ┌─ Tile {tile_count}.{sub_idx} ─────────────────────────────────────────")
                    print(f"    │ Center (latent): {sub_tile.ranges}")
                    print(f"    │ Read window:     lat={read_win.lat_ranges}  src={read_win.src_ranges}")

                # Fetch input from source
                if s == 0:
                    # Source is z_latent
                    B, C = z_latent.shape[:2]
                    flat = z_latent.reshape(B * C, *cfg.spatial_shape)
                    indices = [slice(None)]  # Keep B*C
                    for (rs, re) in read_win.lat_ranges:
                        indices.append(slice(rs, re, None))
                    x_in_flat = periodic_getitem(flat, *indices)
                    new_spatial = x_in_flat.shape[1:]
                    x_in = x_in_flat.reshape(B, C, *new_spatial)
                    x_in = x_in.to(device=cfg.device, dtype=z_latent.dtype).contiguous()
                else:
                    # Source is previous stage buffer
                    x_in = prev_buf.read_block_periodic(
                        read_win.src_ranges, cfg.device, cfg.dtype
                    )

                if cfg.debug >= 3:
                    mem_tracker.print_checkpoint(f"Tile {tile_count}.{sub_idx}: input loaded", x_in)

                # Run stage
                y_tile = stage_runners[s](x_in)

                if cfg.debug >= 3:
                    mem_tracker.print_checkpoint(f"Tile {tile_count}.{sub_idx}: stage computed", y_tile)

                # For cached norms: after first tile, switch to use_cached mode
                if use_cached_norms and first_tile_of_stage:
                    set_all_norms_mode(decoder, "use_cached")
                    first_tile_of_stage = False
                    if cfg.debug >= 1:
                        print(f"    [CachedNorms] Stats cached, using for remaining tiles")

                # Allocate destination buffer if needed
                if not dest_created:
                    B_ = cfg.batch_size
                    C_out = int(y_tile.shape[1])
                    out_spatial = tuple(L * dest_scale for L in cfg.spatial_shape)
                    stage_bufs[s] = _make_stage_buffer(
                        shape=(B_, C_out, *out_spatial),
                        dtype=y_tile.dtype,
                        ndim=cfg.ndim,
                        use_disk_offload=use_disk_offload,
                        disk_offload_dir=disk_offload_dir,
                    )
                    dest_buf = stage_bufs[s]
                    dest_created = True
                    buf_shape = (B_, C_out, *out_spatial)
                    buf_bytes = B_ * C_out * int(torch.tensor(out_spatial).prod().item()) * y_tile.element_size()
                    buf_type = "disk-backed mmap" if use_disk_offload else "CPU"
                    if cfg.debug >= 1:
                        print(f"    [Buffer] Allocated {buf_type} buffer: {buf_shape} = {_format_bytes(buf_bytes)}")
                    if cfg.debug >= 3:
                        mem_tracker.print_checkpoint(f"Stage {s} buffer allocated", dest_buf.tensor)

                # Compute crop coordinates
                crop = compute_crop_spec(
                    sub_tile, read_win, src_scale, dest_scale, up_factor,
                    cfg.spatial_shape, cfg.periodic
                )

                if cfg.debug >= 2:
                    print(f"    │ Output shape:    {tuple(y_tile.shape)}")
                    print(f"    │ Crop slices:     {crop.tile_slices}")
                    print(f"    │ Dest ranges:     {crop.dest_ranges}")

                # Crop valid center
                slices = [slice(None), slice(None)]  # B, C
                slices.extend(crop.tile_slices)
                y_center = y_tile[tuple(slices)]

                # Write to destination buffer
                dest_buf.write_block(crop.dest_ranges, y_center)

                if cfg.debug >= 2:
                    print(f"    │ Written:         {tuple(y_center.shape)}")
                    print(f"    └{'─'*55}")
                if cfg.debug >= 3:
                    mem_tracker.print_checkpoint(f"Tile {tile_count}.{sub_idx}: written to buffer")

                # Free GPU tensor references (actual memory freed lazily by CUDA)
                del y_tile, y_center, x_in

                # Aggressive cleaning: empty cache after every tile (slower but minimal memory)
                if aggressive_cleaning and torch.cuda.is_available():
                    torch.cuda.synchronize(cfg.device)
                    torch.cuda.empty_cache()

            tile_count += 1

        if cfg.debug >= 1:
            print(f"    ✓ Stage {s} complete: processed {tile_count} tiles")

        # Synchronize to ensure all async transfers complete before next stage reads
        if torch.cuda.is_available():
            torch.cuda.synchronize(cfg.device)

        # Clean up GPU memory once per stage (unless already doing aggressive cleaning)
        if not aggressive_cleaning and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.debug >= 3:
            mem_tracker.print_checkpoint(f"Stage {s} cleanup complete")
            mem_tracker.print_stage_summary(s)

        # Update for next stage: close old source buffer (mmap files) to free disk
        if prev_buf is not None and prev_buf is not dest_buf:
            _close_stage_buffer(prev_buf)
            # Clear from stage_bufs list so it won't be closed again
            for i, b in enumerate(stage_bufs):
                if b is prev_buf:
                    stage_bufs[i] = None
        prev_scale = dest_scale
        prev_buf = dest_buf

    # Restore training state and cleanup cached norms
    decoder.train(was_training)
    if use_cached_norms:
        set_all_norms_mode(decoder, "normal")
        clear_all_norm_caches(decoder)

    if cfg.debug >= 3:
        mem_tracker.print_final_summary()
    if cfg.debug >= 1:
        print(f"\n  ✓ Decode complete. Output shape: {tuple(prev_buf.tensor.shape)}")

    # Return last processed stage (may not be final if max_stages was set)
    # For mmap buffers, clone to a regular CPU tensor before the file is cleaned up
    result = prev_buf.tensor.clone() if isinstance(prev_buf, MmapStageBuffer) else prev_buf.tensor
    _close_stage_buffer(prev_buf)
    return result


def chunk_decode_2d(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int], List[int]],
    **kwargs
) -> torch.Tensor:
    """
    2D chunked decode convenience wrapper.

    Args:
        decoder: VAE decoder (2D)
        z_latent: Latent tensor [B, C, H, W]
        chunk_latent: Tile size (int or (H, W) tuple)
        **kwargs: Additional arguments for chunk_decode_strategy_b

    Returns:
        Decoded tensor [B, C_out, H_out, W_out] on CPU
    """
    assert z_latent.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {z_latent.dim()}D"
    return chunk_decode_strategy_b(decoder, z_latent, chunk_latent, **kwargs)


def chunk_decode_3d(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    **kwargs
) -> torch.Tensor:
    """
    3D chunked decode convenience wrapper.

    Args:
        decoder: VAE decoder (3D)
        z_latent: Latent tensor [B, C, H, W, D]
        chunk_latent: Tile size (int or (H, W, D) tuple)
        **kwargs: Additional arguments for chunk_decode_strategy_b

    Returns:
        Decoded tensor [B, C_out, H_out, W_out, D_out] on CPU
    """
    assert z_latent.dim() == 5, f"Expected 5D tensor [B, C, H, W, D], got {z_latent.dim()}D"
    return chunk_decode_strategy_b(decoder, z_latent, chunk_latent, **kwargs)


# ============================================================================
# MULTI-GPU PARALLEL CHUNK DECODE
# ============================================================================

@torch.inference_mode()
def chunk_decode_parallel(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, ...], List[int]],
    devices: List[Union[str, torch.device]],
    *,
    time: Optional[torch.Tensor] = None,
    debug: int = 0,
    max_stage_out_chunk: Optional[Union[int, Tuple[int, ...], List[int]]] = 128,
    periodicity: Union[bool, Tuple[bool, ...], List[bool]] = False,
    use_cached_norms: bool = False,
    aggressive_cleaning: bool = False,
    max_stages: Optional[int] = None,
    use_disk_offload: bool = False,
    disk_offload_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Multi-GPU parallel chunked decode.

    Distributes tiles across multiple CUDA devices for parallel processing.
    Each device gets a copy of the decoder and processes tiles concurrently.

    Args:
        decoder: VAE decoder module
        z_latent: Latent tensor [B, C, H, W] (2D) or [B, C, H, W, D] (3D)
        chunk_latent: Tile size in latent units
        devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        time: Optional time embedding tensor
        debug: Debug verbosity level:
               0 = no output
               1 = per-stage summaries
               2 = per-tile details with nice formatting
               3 = full memory tracking (GPU allocated/reserved, CPU RSS, deltas)
        max_stage_out_chunk: Cap per-stage output size
        periodicity: Enable periodic BCs
        use_cached_norms: If True, use first-tile norm caching (requires cached norm layers)
        aggressive_cleaning: If True, call empty_cache() after every tile (slower but minimal memory)
        max_stages: If set, stop after this many stages (for profiling/debugging).

    Returns:
        Decoded output tensor [B, C_out, *spatial_out] on CPU

    Example:
        # Use 4 GPUs in parallel
        devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        result = chunk_decode_parallel(
            decoder, z_latent, chunk_latent=32,
            devices=devices, use_cached_norms=True
        )

    Note:
        - First tile of each stage is always processed on devices[0]
        - If use_cached_norms=True, cached stats are synced to all devices after first tile
        - Speedup scales roughly linearly with number of devices (for many tiles)
    """
    # Normalize devices
    devices = [torch.device(d) for d in devices]
    num_devices = len(devices)

    if num_devices < 1:
        raise ValueError("At least one device required")

    if num_devices == 1:
        # Fall back to single-device version
        return chunk_decode_strategy_b(
            decoder, z_latent, chunk_latent,
            device=devices[0], time=time, debug=debug,
            max_stage_out_chunk=max_stage_out_chunk,
            periodicity=periodicity, use_cached_norms=use_cached_norms,
            max_stages=max_stages,
        )

    if debug >= 1:
        print()
        print(f"  ┌{'─'*60}┐")
        print(f"  │ PARALLEL DECODE: {num_devices} devices{' '*35}│")
        print(f"  │ Devices: {str([str(d) for d in devices])[:48]:48s}│")
        print(f"  └{'─'*60}┘")

    # Check cached norms
    if use_cached_norms and not _has_cached_norms(decoder):
        if not _needs_cached_norms(decoder):
            # PixelNorm-based decoder: cached norms is a no-op. Silently
            # downgrade rather than erroring on a benign request.
            use_cached_norms = False
        else:
            raise RuntimeError(
                "use_cached_norms=True but decoder has no cached norm layers. "
                "Call prepare_decoder_for_cached_decode(decoder) first."
            )

    # Setup configuration (use first device for config)
    cfg = setup_chunk_config(
        decoder, z_latent, chunk_latent, devices[0],
        max_stage_out_chunk, periodicity, debug
    )

    # Get or create decoder replicas (cached to avoid expensive deep-copy)
    if debug >= 1:
        print("  Getting decoder replicas...")
    decoders = get_cached_decoder_replicas(decoder, devices, debug=debug)

    # Create stage runners for each device
    stage_runners = create_stage_runners_for_devices(decoders, cfg.num_stages, time)

    # Create parallel processor
    parallel_processor = ParallelTileProcessor(
        decoders, devices, stage_runners,
        max_workers=num_devices,
        aggressive_cleaning=aggressive_cleaning,
    )

    # Stage buffers (CPU tensors or disk-backed mmap)
    stage_bufs = [None] * cfg.num_stages
    prev_scale = 1
    prev_buf = None

    # Initialize memory tracker for debug level 3
    mem_tracker = MemoryTracker(devices[0], enabled=(debug >= 3))
    if debug >= 3:
        mem_tracker.print_checkpoint("Initial state (parallel)")

    # Determine number of stages to run
    num_stages_to_run = cfg.num_stages
    if max_stages is not None:
        num_stages_to_run = min(max_stages, cfg.num_stages)
        if debug >= 1:
            print(f"  Limiting to {num_stages_to_run} stages (out of {cfg.num_stages})")

    # Process each stage
    for s in range(num_stages_to_run):
        stage_local_lat = cfg.delta_r_lat[s]
        dest_scale = cfg.scales_after[s]
        src_scale = prev_scale
        up_factor = dest_scale // src_scale

        if debug >= 1:
            print()
            print(f"  ╔{'═'*60}╗")
            print(f"  ║ STAGE {s}/{num_stages_to_run-1} (PARALLEL){' '*42}║")
            print(f"  ╠{'═'*60}╣")
            print(f"  ║ Local RF (latent): {stage_local_lat:<38d}║")
            print(f"  ║ Source scale:      {src_scale:<38d}║")
            print(f"  ║ Dest scale:        {dest_scale:<38d}║")
            print(f"  ║ Upsample factor:   {up_factor:<38d}║")
            print(f"  ╚{'═'*60}╝")
        if debug >= 3:
            mem_tracker.print_checkpoint(f"Stage {s} start")

        dest_buf = stage_bufs[s]
        dest_created = dest_buf is not None

        # For cached norms: set all decoders to cache mode at start of stage
        if use_cached_norms:
            for dec in decoders:
                clear_all_norm_caches(dec)
                set_all_norms_mode(dec, "cache")
            first_tile_of_stage = True
            if debug >= 1:
                print(f"    [CachedNorms] Ready to cache from first tile")

        # Collect all tiles for this stage
        all_tiles = list(iterate_nd_tiles(cfg.spans_per_axis))

        # Process tiles
        tile_idx = 0
        while tile_idx < len(all_tiles):
            center_ranges = all_tiles[tile_idx]
            tile = TileSpec(ranges=center_ranges)
            sub_tiles = generate_sub_tiles(tile, dest_scale, cfg.caps)

            # Process first tile on device 0 (for cached norms)
            if use_cached_norms and first_tile_of_stage:
                # Process first sub-tile on device 0 only
                first_sub = sub_tiles[0]
                read_win = compute_read_window(
                    first_sub, stage_local_lat, src_scale,
                    cfg.spatial_shape, cfg.periodic
                )

                # Fetch input
                if s == 0:
                    B, C = z_latent.shape[:2]
                    flat = z_latent.reshape(B * C, *cfg.spatial_shape)
                    indices = [slice(None)]
                    for (rs, re) in read_win.lat_ranges:
                        indices.append(slice(rs, re, None))
                    x_in_flat = periodic_getitem(flat, *indices)
                    new_spatial = x_in_flat.shape[1:]
                    x_in = x_in_flat.reshape(B, C, *new_spatial)
                    x_in = x_in.to(device=devices[0], dtype=z_latent.dtype).contiguous()
                else:
                    x_in = prev_buf.read_block_periodic(
                        read_win.src_ranges, devices[0], cfg.dtype
                    )

                # Run on device 0 (caches stats)
                y_tile = stage_runners[0][s](x_in)

                # Allocate buffer if needed
                if not dest_created:
                    B_ = cfg.batch_size
                    C_out = int(y_tile.shape[1])
                    out_spatial = tuple(L * dest_scale for L in cfg.spatial_shape)
                    stage_bufs[s] = _make_stage_buffer(
                        shape=(B_, C_out, *out_spatial),
                        dtype=y_tile.dtype,
                        ndim=cfg.ndim,
                        use_disk_offload=use_disk_offload,
                        disk_offload_dir=disk_offload_dir,
                    )
                    dest_buf = stage_bufs[s]
                    dest_created = True
                    buf_shape = (B_, C_out, *out_spatial)
                    buf_bytes = B_ * C_out * int(torch.tensor(out_spatial).prod().item()) * y_tile.element_size()
                    buf_type = "disk-backed mmap" if use_disk_offload else "CPU"
                    if debug >= 1:
                        print(f"    [Buffer] Allocated {buf_type} buffer: {buf_shape} = {_format_bytes(buf_bytes)}")
                    if debug >= 3:
                        mem_tracker.print_checkpoint(f"Stage {s} buffer allocated", dest_buf.tensor)

                # Crop and write
                crop = compute_crop_spec(
                    first_sub, read_win, src_scale, dest_scale, up_factor,
                    cfg.spatial_shape, cfg.periodic
                )
                slices = [slice(None), slice(None)]
                slices.extend(crop.tile_slices)
                y_center = y_tile[tuple(slices)]
                dest_buf.write_block(crop.dest_ranges, y_center.cpu())

                del y_tile, y_center, x_in
                torch.cuda.synchronize(devices[0])
                torch.cuda.empty_cache()

                # Switch all decoders to use_cached mode and sync stats
                for dec in decoders:
                    set_all_norms_mode(dec, "use_cached")
                sync_cached_norms_across_devices(decoders, source_idx=0)

                if debug >= 1:
                    print(f"    [CachedNorms] Stats synced to all {num_devices} devices")

                first_tile_of_stage = False

                # Remove processed sub-tile
                sub_tiles = sub_tiles[1:]
                if not sub_tiles:
                    tile_idx += 1
                    continue

            # Batch remaining sub-tiles for parallel processing
            tile_jobs = []
            for sub_idx, sub_tile in enumerate(sub_tiles):
                read_win = compute_read_window(
                    sub_tile, stage_local_lat, src_scale,
                    cfg.spatial_shape, cfg.periodic
                )

                if debug >= 2:
                    print(f"    ┌─ Tile {tile_idx}.{sub_idx} (parallel) ───────────────────────────────")
                    print(f"    │ Center (latent): {sub_tile.ranges}")
                    print(f"    │ Read window:     lat={read_win.lat_ranges}  src={read_win.src_ranges}")
                    print(f"    └{'─'*55}")

                # Fetch input (to CPU)
                if s == 0:
                    B, C = z_latent.shape[:2]
                    flat = z_latent.reshape(B * C, *cfg.spatial_shape)
                    indices = [slice(None)]
                    for (rs, re) in read_win.lat_ranges:
                        indices.append(slice(rs, re, None))
                    x_in_flat = periodic_getitem(flat, *indices)
                    new_spatial = x_in_flat.shape[1:]
                    x_in = x_in_flat.reshape(B, C, *new_spatial)
                    x_in = x_in.to(dtype=z_latent.dtype).contiguous()  # CPU
                else:
                    x_in = prev_buf.read_block_periodic(
                        read_win.src_ranges, torch.device("cpu"), cfg.dtype
                    )

                # Allocate buffer if needed (use a dummy run to get output shape)
                if not dest_created:
                    # Run one tile to get output channels
                    x_probe = x_in.to(device=devices[0])
                    y_probe = stage_runners[0][s](x_probe)
                    B_ = cfg.batch_size
                    C_out = int(y_probe.shape[1])
                    out_spatial = tuple(L * dest_scale for L in cfg.spatial_shape)
                    stage_bufs[s] = _make_stage_buffer(
                        shape=(B_, C_out, *out_spatial),
                        dtype=y_probe.dtype,
                        ndim=cfg.ndim,
                        use_disk_offload=use_disk_offload,
                        disk_offload_dir=disk_offload_dir,
                    )
                    dest_buf = stage_bufs[s]
                    dest_created = True
                    buf_shape = (B_, C_out, *out_spatial)
                    buf_bytes = B_ * C_out * int(torch.tensor(out_spatial).prod().item()) * y_probe.element_size()
                    del x_probe, y_probe
                    torch.cuda.empty_cache()
                    buf_type = "disk-backed mmap" if use_disk_offload else "CPU"
                    if debug >= 1:
                        print(f"    [Buffer] Allocated {buf_type} buffer: {buf_shape} = {_format_bytes(buf_bytes)}")
                    if debug >= 3:
                        mem_tracker.print_checkpoint(f"Stage {s} buffer allocated", dest_buf.tensor)

                crop = compute_crop_spec(
                    sub_tile, read_win, src_scale, dest_scale, up_factor,
                    cfg.spatial_shape, cfg.periodic
                )
                tile_jobs.append((x_in, crop))

            # Process batch in parallel
            if tile_jobs:
                parallel_processor.process_tiles_parallel(
                    s, tile_jobs, dest_buf, debug=debug
                )

            tile_idx += 1

        if debug >= 1:
            print(f"    ✓ Stage {s} complete: processed {tile_idx} tiles")

        # Synchronize all devices to ensure async transfers complete before next stage reads
        if torch.cuda.is_available():
            for device in devices:
                torch.cuda.synchronize(device)

        # Clean up GPU memory once per stage (unless already doing aggressive cleaning)
        if not aggressive_cleaning and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if debug >= 3:
            mem_tracker.print_checkpoint(f"Stage {s} cleanup complete")
            mem_tracker.print_stage_summary(s)

        # Update for next stage: close old source buffer (mmap files) to free disk
        if prev_buf is not None and prev_buf is not dest_buf:
            _close_stage_buffer(prev_buf)
            for i, b in enumerate(stage_bufs):
                if b is prev_buf:
                    stage_bufs[i] = None
        prev_scale = dest_scale
        prev_buf = dest_buf

    # Cleanup: reset norms on all decoders
    if use_cached_norms:
        for dec in decoders:
            set_all_norms_mode(dec, "normal")
            clear_all_norm_caches(dec)

    if debug >= 3:
        mem_tracker.print_final_summary()
    if debug >= 1:
        print(f"\n  ✓ Parallel decode complete. Output shape: {tuple(prev_buf.tensor.shape)}")

    # Return last processed stage (may not be final if max_stages was set)
    result = prev_buf.tensor.clone() if isinstance(prev_buf, MmapStageBuffer) else prev_buf.tensor
    _close_stage_buffer(prev_buf)
    return result


def chunk_decode_2d_parallel(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int], List[int]],
    devices: List[Union[str, torch.device]],
    **kwargs
) -> torch.Tensor:
    """2D multi-GPU parallel chunked decode. See chunk_decode_parallel for details."""
    assert z_latent.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {z_latent.dim()}D"
    return chunk_decode_parallel(decoder, z_latent, chunk_latent, devices, **kwargs)


def chunk_decode_3d_parallel(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    devices: List[Union[str, torch.device]],
    **kwargs
) -> torch.Tensor:
    """3D multi-GPU parallel chunked decode. See chunk_decode_parallel for details."""
    assert z_latent.dim() == 5, f"Expected 5D tensor [B, C, H, W, D], got {z_latent.dim()}D"
    return chunk_decode_parallel(decoder, z_latent, chunk_latent, devices, **kwargs)


# ============================================================================
# HIGH-LEVEL API WITH CACHED NORMS
# ============================================================================

@torch.inference_mode()
def chunk_decode_with_cached_norms(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, ...], List[int]],
    *,
    device: Optional[Union[str, torch.device]] = None,
    time: Optional[torch.Tensor] = None,
    debug: int = 0,
    max_stage_out_chunk: Optional[Union[int, Tuple[int, ...], List[int]]] = 128,
    periodicity: Union[bool, Tuple[bool, ...], List[bool]] = False,
    convert_norms: bool = True,
) -> torch.Tensor:
    """
    Chunked decode with cached normalization statistics to eliminate tile boundary artifacts.

    This is the recommended high-level API for chunked decoding. It:
    1. Converts decoder's norm layers to cached versions (if convert_norms=True)
    2. For each stage: first tile computes and caches norm stats, all other tiles reuse them
    3. No memory blowup - stats gathered from first tile only, not full data

    The Problem:
        Standard chunk decode computes normalization statistics per-tile, which differ
        between tiles, causing visible "patches" at boundaries.

    The Solution:
        For each decoder stage, compute normalization statistics from the FIRST tile only,
        then use those cached statistics for ALL subsequent tiles in that stage.
        Cache is cleared between stages.

    Args:
        decoder: VAE decoder module
        z_latent: Latent tensor [B, C, H, W] (2D) or [B, C, H, W, D] (3D)
        chunk_latent: Tile size in latent units
        device: Compute device
        time: Optional time embedding
        debug: Enable debug prints
        max_stage_out_chunk: Cap per-stage output size
        periodicity: Enable periodic BCs
        convert_norms: If True, convert norm layers to cached versions.
                      Set False if already converted (e.g., for multiple decodes).

    Returns:
        Decoded output tensor [B, C_out, *spatial_out] on CPU

    Example:
        # Basic usage (handles everything automatically):
        result = chunk_decode_with_cached_norms(decoder, z_latent, chunk_latent=32)

        # For multiple decodes, convert once:
        prepare_decoder_for_cached_decode(decoder)
        for z in latents:
            result = chunk_decode_with_cached_norms(
                decoder, z, chunk_latent=32,
                convert_norms=False  # Already converted
            )
    """
    # Step 1: Convert norm layers to cached versions (one-time)
    if convert_norms:
        if debug >= 1:
            print("  Converting norm layers to cached versions...")
        prepare_decoder_for_cached_decode(decoder, inplace=True)

    # Step 2: Run chunked decode with per-stage first-tile caching
    if debug >= 1:
        print("  Running chunked decode with first-tile norm caching...")

    result = chunk_decode_strategy_b(
        decoder, z_latent, chunk_latent,
        device=device, time=time, debug=debug,
        max_stage_out_chunk=max_stage_out_chunk,
        periodicity=periodicity,
        use_cached_norms=True,  # This enables per-stage first-tile caching
    )

    return result


def chunk_decode_2d_cached(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int], List[int]],
    **kwargs
) -> torch.Tensor:
    """
    2D chunked decode with cached norms (convenience wrapper).

    See chunk_decode_with_cached_norms for full documentation.
    """
    assert z_latent.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {z_latent.dim()}D"
    return chunk_decode_with_cached_norms(decoder, z_latent, chunk_latent, **kwargs)


def chunk_decode_3d_cached(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    **kwargs
) -> torch.Tensor:
    """
    3D chunked decode with cached norms (convenience wrapper).

    See chunk_decode_with_cached_norms for full documentation.
    """
    assert z_latent.dim() == 5, f"Expected 5D tensor [B, C, H, W, D], got {z_latent.dim()}D"
    return chunk_decode_with_cached_norms(decoder, z_latent, chunk_latent, **kwargs)


# ============================================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================================

# Alias for backward compatibility with original chunk_decode.py
chunk_decode_strategy_b_3d = chunk_decode_3d


# ============================================================================
# EXAMPLE / TEST
# ============================================================================

if __name__ == "__main__":
    print("chunk_decode_2.py - Dimension-agnostic chunked decoder")
    print("Supports both 2D [B, C, H, W] and 3D [B, C, H, W, D] tensors")
    print()
    print("=" * 70)
    print("BASIC USAGE (without cached norms):")
    print("=" * 70)
    print()
    print("  from diffsci2.extra.chunk_decode_2 import chunk_decode_2d, chunk_decode_3d")
    print()
    print("  # For 2D:")
    print("  result = chunk_decode_2d(decoder, z_latent_2d, chunk_latent=32)")
    print()
    print("  # For 3D:")
    print("  result = chunk_decode_3d(decoder, z_latent_3d, chunk_latent=(16, 32, 32))")
    print()
    print("  # With periodicity:")
    print("  result = chunk_decode_3d(decoder, z, chunk, periodicity=(True, True, True))")
    print()
    print("=" * 70)
    print("WITH CACHED NORMS (eliminates tile boundary artifacts):")
    print("=" * 70)
    print()
    print("  How it works:")
    print("    - For each decoder stage, the FIRST tile computes norm stats")
    print("    - All subsequent tiles in that stage USE those cached stats")
    print("    - Cache is cleared when moving to next stage")
    print("    - NO memory blowup (stats from first tile only, not full data)")
    print()
    print("  from diffsci2.extra.chunk_decode_2 import (")
    print("      chunk_decode_with_cached_norms,")
    print("      chunk_decode_3d_cached,")
    print("      prepare_decoder_for_cached_decode,")
    print("  )")
    print()
    print("  # Simple usage (handles everything automatically):")
    print("  result = chunk_decode_with_cached_norms(decoder, z_latent, chunk_latent=32)")
    print()
    print("  # Or use low-level API with use_cached_norms flag:")
    print("  prepare_decoder_for_cached_decode(decoder)  # One-time conversion")
    print("  result = chunk_decode_3d(decoder, z, chunk, use_cached_norms=True)")
    print()
    print("  # For multiple decodes (convert once, reuse):")
    print("  prepare_decoder_for_cached_decode(decoder)")
    print("  for z in latents:")
    print("      result = chunk_decode_with_cached_norms(")
    print("          decoder, z, chunk_latent=32,")
    print("          convert_norms=False  # Already converted")
    print("      )")
    print()
    print("=" * 70)
    print("MULTI-GPU PARALLEL (process tiles on multiple GPUs):")
    print("=" * 70)
    print()
    print("  How it works:")
    print("    - Decoder is replicated to each GPU")
    print("    - Tiles are distributed round-robin across GPUs")
    print("    - All GPUs process tiles concurrently (ThreadPoolExecutor)")
    print("    - Results collected to CPU buffer")
    print("    - If using cached norms: first tile caches stats on GPU 0,")
    print("      then stats are synced to all other GPUs")
    print()
    print("  from diffsci2.extra.chunk_decode_2 import (")
    print("      chunk_decode_parallel,")
    print("      chunk_decode_3d_parallel,")
    print("      prepare_decoder_for_cached_decode,")
    print("  )")
    print()
    print("  # Use 4 GPUs in parallel:")
    print("  devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']")
    print("  result = chunk_decode_parallel(")
    print("      decoder, z_latent, chunk_latent=32,")
    print("      devices=devices")
    print("  )")
    print()
    print("  # With cached norms on multiple GPUs:")
    print("  prepare_decoder_for_cached_decode(decoder)")
    print("  result = chunk_decode_3d_parallel(")
    print("      decoder, z_latent, chunk_latent=32,")
    print("      devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],")
    print("      use_cached_norms=True")
    print("  )")
    print()
    print("=" * 70)
    print("DEBUG LEVELS:")
    print("=" * 70)
    print()
    print("  debug=0: No output (default)")
    print("  debug=1: Per-stage summaries")
    print("           - Configuration overview")
    print("           - Stage parameters (RF, scale, upsample factor)")
    print("           - Buffer allocations")
    print("           - Stage completion status")
    print()
    print("  debug=2: Per-tile details (nice box formatting)")
    print("           - Tile coordinates and read windows")
    print("           - Output shapes and crop specifications")
    print("           - Parallel batch dispatch info")
    print()
    print("  debug=3: Full memory tracking")
    print("           - GPU allocated/reserved memory at each step")
    print("           - CPU RSS memory usage")
    print("           - Memory deltas between operations")
    print("           - Per-stage and final memory summaries")
    print("           - Tensor size annotations")
    print()
    print("  Example:")
    print("  result = chunk_decode_3d(decoder, z, chunk, debug=3)")

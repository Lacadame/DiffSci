# chunk_encode_2.py
# -----------------------------------------------------------------------------
# Dimension-agnostic chunked (tiled) ENCODE for VAE encoders. Mirror of
# chunk_decode_2.py with the spatial direction reversed: input is at HIGH
# resolution and output (latent) at LOW resolution.
#
# Stage decomposition for VAEEncoder (N+1 stages, mirror of decoder):
#   S0  = conv_in + down[0].block[*]                         (at input res)
#   Sk  = down[k-1].downsample + down[k].block[*]            (1 <= k <= N-1)
#   SN  = mid.block_1 + mid.block_2 + norm_out + nonlin
#         + conv_out + quant_conv                            (at latent res)
#
# Tiles are partitioned in LATENT (output) units, matching how chunk_decode
# does it. For each latent tile, the read window is computed in the source
# stage's coordinate system using per-stage halo radii expressed in INPUT
# pixel units.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

import torch

from diffsci2.torchutils import periodic_getitem
from diffsci2.nets.cached_norms import (
    convert_to_cached_norms,
    set_all_norms_mode,
    clear_all_norm_caches,
)

# Reuse shared primitives from the decoder module to avoid duplication.
from diffsci2.extra.chunk_decode_2 import (
    CPUStageBuffer,
    MmapStageBuffer,
    _make_stage_buffer,
    _close_stage_buffer,
    normalize_tuple,
    normalize_bool_tuple,
    iterate_nd_tiles,
    TileSpec,
    _format_bytes,
    MemoryTracker,
    _device_of,
    _dtype_of,
    _has_cached_norms,
)


# ============================================================================
# RF / SCALE COMPUTATION FOR THE ENCODER
# ============================================================================

def compute_encoder_stage_radii_and_scales(encoder):
    """Per-stage radii and scales for the VAE encoder.

    Returns
    -------
    delta : list[int]
        Input-pixel RF radius added by each stage (length N+1).
    radii_cum : list[int]
        Cumulative input-pixel RF radius after each stage (length N+1).
    scales_after : list[int]
        Spatial scale (input-pixels-per-cell) of each stage's OUTPUT.
    scales_src : list[int]
        Spatial scale of each stage's INPUT (i.e. the source buffer it reads
        from). For stage 0 this is 1 (input image scale).

    Stage decomposition matches the natural cut described at the top of the
    file. Radii are computed analytically using
        rf_out = rf_in + (k - 1) * j_in
    with j (jump) accumulating across strided convolutions.
    """
    cfg = encoder.config
    if cfg.has_mid_attn or len(cfg.attn_resolutions) > 0:
        raise NotImplementedError(
            "Chunked encoding requires NO attention in the encoder. "
            f"has_mid_attn={cfg.has_mid_attn}, attn_resolutions={cfg.attn_resolutions}"
        )

    N = int(cfg.num_resolutions)
    R = int(cfg.num_res_blocks)

    delta: List[int] = []
    scales_after: List[int] = []
    scales_src: List[int] = []

    j = 1  # current jump (= scale of current intermediate tensor)

    # Stage 0: conv_in + down[0].block[*]  (R resblocks, 2 convs each, all at j=1)
    src = j
    s0 = 1 + 2 * R
    delta.append(s0)
    scales_src.append(src)
    scales_after.append(j)  # no downsample yet

    # Stages 1..N-1
    for _k in range(1, N):
        src = j
        # downsample: 3x3 stride 2 at j -> radius += 1*j, then j *= 2
        s_k = j
        j *= 2
        # R resblocks at new j -> 2R convs each adding 1*j
        s_k += 2 * R * j
        delta.append(s_k)
        scales_src.append(src)
        scales_after.append(j)

    # Stage N (final): mid.block_1 + mid.block_2 + conv_out (5 convs at current j)
    # quant_conv is 1x1, no contribution. norm_out / nonlinearity have no
    # spatial extent (in cached-norm mode: pointwise).
    src = j
    s_N = 5 * j
    delta.append(s_N)
    scales_src.append(src)
    scales_after.append(j)

    radii_cum = [sum(delta[: i + 1]) for i in range(len(delta))]
    return delta, radii_cum, scales_after, scales_src


# ============================================================================
# STAGE RUNNERS FOR THE VAE ENCODER
# ============================================================================

@torch.inference_mode()
def run_vae_enc_stage0(
    encoder,
    x: torch.Tensor,
    temb: Optional[torch.Tensor],
) -> torch.Tensor:
    """Stage 0: conv_in + down[0].block[*] (no downsample yet)."""
    h = encoder.conv_in(x)
    down0 = encoder.down[0]
    for i in range(len(down0.block)):
        h = down0.block[i](h, temb)
        if len(down0.attn) > i:
            h = down0.attn[i](h)
    return h


@torch.inference_mode()
def run_vae_enc_down_stage(
    encoder,
    x: torch.Tensor,
    level_index: int,
    temb: Optional[torch.Tensor],
) -> torch.Tensor:
    """Stage k (1 <= k <= N-1): down[k-1].downsample + down[k].block[*].

    The downsample lives on the PREVIOUS level's module (down[k-1]) but it is
    the operation that takes us into level k. We pull it from there to keep
    the decomposition symmetric with the decoder.
    """
    if level_index < 1:
        raise ValueError(f"level_index must be >= 1 for run_vae_enc_down_stage, got {level_index}")
    prev = encoder.down[level_index - 1]
    h = prev.downsample(x)
    cur = encoder.down[level_index]
    for i in range(len(cur.block)):
        h = cur.block[i](h, temb)
        if len(cur.attn) > i:
            h = cur.attn[i](h)
    return h


@torch.inference_mode()
def run_vae_enc_final_stage(
    encoder,
    x: torch.Tensor,
    temb: Optional[torch.Tensor],
) -> torch.Tensor:
    """Final stage: mid blocks + norm_out + nonlin + conv_out + quant_conv."""
    h = encoder.mid.block_1(x, temb)
    if hasattr(encoder.mid, 'attn_1'):
        h = encoder.mid.attn_1(h)
    h = encoder.mid.block_2(h, temb)
    h = encoder.norm_out(h)
    h = h * torch.sigmoid(h)  # SiLU / swish (matches `nonlinearity` in vaenet.py)
    h = encoder.conv_out(h)
    h = encoder.quant_conv(h)
    return h


def make_vae_encoder_stage_runner(
    encoder,
    stage_idx: int,
    num_stages: int,
    temb: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Pick the right stage runner for stage_idx in [0, num_stages-1]."""
    N = int(encoder.config.num_resolutions)
    if stage_idx == 0:
        return lambda x: run_vae_enc_stage0(encoder, x, temb)
    if 1 <= stage_idx <= N - 1:
        k = stage_idx
        return lambda x: run_vae_enc_down_stage(encoder, x, k, temb)
    return lambda x: run_vae_enc_final_stage(encoder, x, temb)


# ============================================================================
# CACHED NORMALIZATION SUPPORT (alias of decoder helper)
# ============================================================================

def _needs_cached_norms(model) -> bool:
    """Return True iff `model` uses GroupNorm (the only norm where cached-norm
    calibration is actually necessary). PixelNorm has spatial RF=1, so cached
    norms is a no-op — we silently skip the conversion and any per-tile
    calibration for `norm_type='pixel'` VAEs."""
    return getattr(getattr(model, "config", None), "norm_type", "group") == "group"


def prepare_encoder_for_cached_encode(encoder, inplace: bool = True):
    """Convert encoder norm layers to cached versions. Alias of decoder helper.

    For PixelNorm-based encoders, this is a no-op (cached norms have nothing
    to cache when the receptive field of the norm is 1 pixel).
    """
    if not _needs_cached_norms(encoder):
        return encoder
    return convert_to_cached_norms(encoder, inplace=inplace)


# ============================================================================
# CHUNK SPAN HELPERS (latent units)
# ============================================================================

def _make_center_spans_1d_enc(L_lat: int, chunk_lat: int) -> List[Tuple[int, int]]:
    """Partition [0, L_lat) into contiguous tiles of size <= chunk_lat.

    Unlike the decoder, the encoder does not need to subtract twice the halo
    here: the halo lives in INPUT coords and only enlarges the read window.
    The latent-space partition is just a clean tiling.
    """
    if chunk_lat >= L_lat:
        return [(0, L_lat)]
    spans = []
    pos = 0
    while pos < L_lat:
        end = min(pos + chunk_lat, L_lat)
        spans.append((pos, end))
        pos = end
    return spans


def _aligned_halo(halo_raw: int, down_factor: int) -> int:
    """Round halo up to a multiple of down_factor for stride alignment."""
    if down_factor <= 1:
        return halo_raw
    return ((halo_raw + down_factor - 1) // down_factor) * down_factor


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EncodeChunkConfig:
    device: torch.device
    dtype: torch.dtype
    ndim: int
    batch_size: int
    in_channels: int
    input_shape: Tuple[int, ...]      # spatial shape of input image
    latent_shape: Tuple[int, ...]     # spatial shape of latent (= input_shape // F)
    F: int                             # total downsampling factor
    chunk_latent: Tuple[int, ...]
    num_stages: int                    # N+1
    delta_input: List[int]             # per-stage RF growth (input pixel units)
    radii_cum_input: List[int]
    scales_after: List[int]
    scales_src: List[int]
    halo_src_aligned: List[int]        # halo in source-buffer coords, aligned to down_factor
    spans_per_axis: List[List[Tuple[int, int]]]
    periodic: Tuple[bool, ...]
    debug: int


def setup_encode_config(
    encoder,
    x: torch.Tensor,
    chunk_latent: Union[int, Tuple, List],
    device: Optional[torch.device],
    periodicity: Union[bool, Tuple[bool, ...], List[bool]],
    debug: int,
) -> EncodeChunkConfig:
    """Build the per-call configuration for chunked encoding."""
    cfg = encoder.config
    ndim = x.dim() - 2
    if ndim not in (2, 3):
        raise ValueError(f"x must be 4D (2D image) or 5D (3D volume), got {x.dim()}D")
    if cfg.dimension != ndim:
        raise ValueError(
            f"encoder config dimension {cfg.dimension} != input ndim {ndim}"
        )

    if device is None:
        device = _device_of(encoder)
    else:
        device = torch.device(device)
    dtype = _dtype_of(encoder)

    F = 2 ** (int(cfg.num_resolutions) - 1)
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    input_shape = tuple(int(s) for s in x.shape[2:])
    for i, s in enumerate(input_shape):
        if s % F != 0:
            raise ValueError(
                f"input spatial dim {i} = {s} is not a multiple of F={F}"
            )
    latent_shape = tuple(s // F for s in input_shape)

    chunk = normalize_tuple(chunk_latent, ndim, "chunk_latent")
    periodic = normalize_bool_tuple(periodicity, ndim, "periodicity")

    delta, radii_cum, scales_after, scales_src = (
        compute_encoder_stage_radii_and_scales(encoder)
    )
    num_stages = len(delta)
    halo_src_aligned = []
    for s in range(num_stages):
        halo_raw = delta[s] // scales_src[s]   # halo in source-buffer coords
        # down_factor for stage s = scales_after[s] / scales_src[s]; need rs aligned to it
        df = scales_after[s] // scales_src[s]
        halo_src_aligned.append(_aligned_halo(halo_raw, df))

    spans_per_axis = [
        _make_center_spans_1d_enc(L, ch) for L, ch in zip(latent_shape, chunk)
    ]

    if debug >= 1:
        total_tiles = 1
        for spans in spans_per_axis:
            total_tiles *= len(spans)
        print()
        print(f"  ┌{'─'*60}┐")
        print(f"  │ CHUNKED ENCODE CONFIGURATION{' '*30}│")
        print(f"  ├{'─'*60}┤")
        print(f"  │ Input shape:     {str(input_shape):20s} ({ndim}D){' '*14}│")
        print(f"  │ Latent shape:    {str(latent_shape):40s}│")
        print(f"  │ F:               {F:<40d}│")
        print(f"  │ Chunk (latent):  {str(chunk):40s}│")
        print(f"  │ Num stages:      {num_stages:<40d}│")
        print(f"  │ Total tiles:     {total_tiles:<40d}│")
        print(f"  │ Periodicity:     {str(periodic):40s}│")
        print(f"  ├{'─'*60}┤")
        print(f"  │ delta (input px):       {str(delta):31s}│")
        print(f"  │ radii_cum (input px):   {str(radii_cum):31s}│")
        print(f"  │ scales_after:           {str(scales_after):31s}│")
        print(f"  │ scales_src:             {str(scales_src):31s}│")
        print(f"  │ halo_src (aligned):     {str(halo_src_aligned):31s}│")
        print(f"  └{'─'*60}┘")

    return EncodeChunkConfig(
        device=device, dtype=dtype, ndim=ndim,
        batch_size=batch_size, in_channels=in_channels,
        input_shape=input_shape, latent_shape=latent_shape, F=F,
        chunk_latent=chunk, num_stages=num_stages,
        delta_input=delta, radii_cum_input=radii_cum,
        scales_after=scales_after, scales_src=scales_src,
        halo_src_aligned=halo_src_aligned,
        spans_per_axis=spans_per_axis, periodic=periodic, debug=debug,
    )


# ============================================================================
# READ-WINDOW + CROP COMPUTATION
# ============================================================================

@dataclass
class _EncReadCrop:
    src_ranges: Tuple[Tuple[int, int], ...]   # in source-buffer coords
    tile_slices: Tuple[slice, ...]            # crop into stage output (in stage output coords)
    dest_ranges: Tuple[Tuple[int, int], ...]  # write target in dest buffer (stage output coords)


def _compute_encoder_read_and_crop(
    tile: TileSpec,
    cfg: EncodeChunkConfig,
    stage_idx: int,
) -> _EncReadCrop:
    """Compute (src_ranges, crop_slices, dest_ranges) for a tile at stage_idx."""
    src_scale = cfg.scales_src[stage_idx]
    dst_scale = cfg.scales_after[stage_idx]
    down_factor = dst_scale // src_scale  # 1 for stage 0 and final, 2 for others
    src_unit_per_lat = cfg.F // src_scale
    dst_unit_per_lat = cfg.F // dst_scale
    halo = cfg.halo_src_aligned[stage_idx]

    src_ranges = []
    tile_slices = []
    dest_ranges = []

    for ((a, b), L_lat, per) in zip(
        tile.ranges, cfg.latent_shape, cfg.periodic
    ):
        center_src_lo = a * src_unit_per_lat
        center_src_hi = b * src_unit_per_lat
        if per:
            rs = center_src_lo - halo
            re = center_src_hi + halo
        else:
            L_src = L_lat * src_unit_per_lat
            rs = max(0, center_src_lo - halo)
            re = min(L_src, center_src_hi + halo)

        # Stage-output crop offset is (a * dst_unit_per_lat) - (rs / down_factor),
        # measured in stage output coords. With aligned halo, rs is divisible by
        # down_factor (also at clamped boundaries since L_src and 0 are both
        # multiples of down_factor for non-final stages).
        if rs % down_factor != 0 or (re - rs) % down_factor != 0:
            raise RuntimeError(
                f"alignment failure at stage {stage_idx}: "
                f"rs={rs}, re={re}, down_factor={down_factor}"
            )
        offset = a * dst_unit_per_lat - rs // down_factor
        length = (b - a) * dst_unit_per_lat
        src_ranges.append((rs, re))
        tile_slices.append(slice(offset, offset + length))
        dest_ranges.append((a * dst_unit_per_lat, b * dst_unit_per_lat))

    return _EncReadCrop(
        src_ranges=tuple(src_ranges),
        tile_slices=tuple(tile_slices),
        dest_ranges=tuple(dest_ranges),
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

@torch.inference_mode()
def chunk_encode_strategy_b(
    encoder,
    x: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, ...], List[int]],
    *,
    device: Optional[Union[str, torch.device]] = None,
    debug: int = 0,
    periodicity: Union[bool, Tuple[bool, ...], List[bool]] = False,
    use_cached_norms: bool = False,
    aggressive_cleaning: bool = False,
    use_disk_offload: bool = False,
    disk_offload_dir: Optional[str] = None,
    max_stages: Optional[int] = None,
) -> torch.Tensor:
    """Dimension-agnostic chunked encode (mirror of chunk_decode_strategy_b).

    Parameters
    ----------
    encoder : VAEEncoder
        Encoder module (no attention, please).
    x : torch.Tensor
        Input image [B, C, *spatial] in INPUT pixel units.
    chunk_latent : int or tuple
        Tile size in LATENT (output) units.
    device : str or torch.device or None
        Compute device for stage runs (default: encoder's device).
    debug : int
        0=quiet, 1=per-stage summary, 2=per-tile, 3=memory tracking.
    periodicity : bool or tuple
        Per-axis periodic boundary flag.
    use_cached_norms : bool
        Cache GroupNorm stats from first tile of each stage; reuses for all
        other tiles. Requires prepare_encoder_for_cached_encode() first.
    aggressive_cleaning : bool
        torch.cuda.empty_cache() after each tile (slower, less memory).
    use_disk_offload : bool
        Stage buffers as numpy memmap files instead of CPU tensors.
    disk_offload_dir : str or None
        Directory for mmap files when use_disk_offload=True.
    max_stages : int or None
        Stop after this many stages (debugging). Returns the stage buffer's
        contents instead of the final encoder output.

    Returns
    -------
    z : torch.Tensor
        Encoder output [B, 2*z_dim, *latent] on CPU. (Unsplit; caller can
        split into mean/logvar via torch.chunk(z, 2, dim=1).)
    """
    cfg = setup_encode_config(
        encoder, x, chunk_latent, device, periodicity, debug,
    )

    if use_cached_norms and not _has_cached_norms(encoder):
        if not _needs_cached_norms(encoder):
            # PixelNorm-based encoder: cached norms is a no-op. Silently
            # downgrade rather than erroring on a benign request.
            use_cached_norms = False
        else:
            raise RuntimeError(
                "use_cached_norms=True but encoder has no cached norm layers. "
                "Call prepare_encoder_for_cached_encode(encoder) first."
            )

    stage_runners = [
        make_vae_encoder_stage_runner(encoder, s, cfg.num_stages, None)
        for s in range(cfg.num_stages)
    ]

    was_training = encoder.training
    encoder.eval()

    mem_tracker = MemoryTracker(cfg.device, enabled=(cfg.debug >= 3))
    if cfg.debug >= 3:
        mem_tracker.print_checkpoint("Initial state")

    num_stages_to_run = cfg.num_stages
    if max_stages is not None:
        num_stages_to_run = min(max_stages, cfg.num_stages)
        if cfg.debug >= 1:
            print(f"  Limiting to {num_stages_to_run} stages of {cfg.num_stages}")

    stage_bufs: List[Optional[Union[CPUStageBuffer, MmapStageBuffer]]] = [
        None
    ] * cfg.num_stages
    prev_buf = None  # for stage 0, source is the input image x

    # Cache to keep prev x reference alive only while needed
    x_cpu = x  # keep original; transfers to GPU happen per tile

    for s in range(num_stages_to_run):
        if cfg.debug >= 1:
            print()
            print(f"  ╔{'═'*60}╗")
            print(f"  ║ STAGE {s}/{num_stages_to_run-1} (encode){' '*42}║")
            print(f"  ║   src_scale={cfg.scales_src[s]} dst_scale={cfg.scales_after[s]} "
                  f"halo_src={cfg.halo_src_aligned[s]}{' '*10}║")
            print(f"  ╚{'═'*60}╝")

        if use_cached_norms:
            clear_all_norm_caches(encoder)
            set_all_norms_mode(encoder, "cache")
            first_tile = True

        dest_buf = stage_bufs[s]
        dest_created = dest_buf is not None
        tile_count = 0

        for tile_ranges in iterate_nd_tiles(cfg.spans_per_axis):
            tile = TileSpec(ranges=tile_ranges)
            rc = _compute_encoder_read_and_crop(tile, cfg, s)

            # Fetch input
            if s == 0:
                # Source is the input image x (on whatever device it lives).
                B, C = x_cpu.shape[:2]
                flat = x_cpu.reshape(B * C, *cfg.input_shape)
                indices = [slice(None)]
                for (rs, re) in rc.src_ranges:
                    indices.append(slice(rs, re))
                x_in_flat = periodic_getitem(flat, *indices)
                new_spatial = x_in_flat.shape[1:]
                x_in = x_in_flat.reshape(B, C, *new_spatial).to(
                    device=cfg.device, dtype=x_cpu.dtype
                ).contiguous()
            else:
                x_in = prev_buf.read_block_periodic(
                    rc.src_ranges, cfg.device, cfg.dtype
                )

            if cfg.debug >= 2:
                print(f"    ┌─ Tile {tile_count} stage {s} ─────────────────────")
                print(f"    │ Center latent: {tile.ranges}")
                print(f"    │ src_ranges:    {rc.src_ranges}")
                print(f"    │ x_in shape:    {tuple(x_in.shape)}")

            y_tile = stage_runners[s](x_in)

            if use_cached_norms and first_tile:
                set_all_norms_mode(encoder, "use_cached")
                first_tile = False
                if cfg.debug >= 1:
                    print(f"    [CachedNorms] cached from first tile")

            if not dest_created:
                C_out = int(y_tile.shape[1])
                dst_unit_per_lat = cfg.F // cfg.scales_after[s]
                out_spatial = tuple(L * dst_unit_per_lat for L in cfg.latent_shape)
                stage_bufs[s] = _make_stage_buffer(
                    shape=(cfg.batch_size, C_out, *out_spatial),
                    dtype=y_tile.dtype, ndim=cfg.ndim,
                    use_disk_offload=use_disk_offload,
                    disk_offload_dir=disk_offload_dir,
                )
                dest_buf = stage_bufs[s]
                dest_created = True
                buf_bytes = (
                    cfg.batch_size * C_out
                    * int(torch.tensor(out_spatial).prod().item())
                    * y_tile.element_size()
                )
                buf_type = "disk-mmap" if use_disk_offload else "CPU"
                if cfg.debug >= 1:
                    print(f"    [Buffer] {buf_type} stage{s}: "
                          f"{(cfg.batch_size, C_out, *out_spatial)} = "
                          f"{_format_bytes(buf_bytes)}")

            slices = [slice(None), slice(None)]
            slices.extend(rc.tile_slices)
            y_center = y_tile[tuple(slices)]
            dest_buf.write_block(rc.dest_ranges, y_center)

            if cfg.debug >= 2:
                print(f"    │ y_tile shape:  {tuple(y_tile.shape)}")
                print(f"    │ tile_slices:   {rc.tile_slices}")
                print(f"    │ dest_ranges:   {rc.dest_ranges}")
                print(f"    └{'─'*55}")

            del y_tile, y_center, x_in
            if aggressive_cleaning and cfg.device.type == 'cuda':
                torch.cuda.synchronize(cfg.device)
                torch.cuda.empty_cache()
            tile_count += 1

        if cfg.debug >= 1:
            print(f"    ✓ Stage {s} done: {tile_count} tiles")

        if cfg.device.type == 'cuda':
            torch.cuda.synchronize(cfg.device)
            if not aggressive_cleaning:
                torch.cuda.empty_cache()

        # Free the previous stage buffer (we don't need it anymore)
        if prev_buf is not None and prev_buf is not dest_buf:
            _close_stage_buffer(prev_buf)
            for i, b in enumerate(stage_bufs):
                if b is prev_buf:
                    stage_bufs[i] = None
        prev_buf = dest_buf

    encoder.train(was_training)
    if use_cached_norms:
        set_all_norms_mode(encoder, "normal")
        clear_all_norm_caches(encoder)

    if cfg.debug >= 1:
        print(f"\n  ✓ Encode complete. Output shape: {tuple(prev_buf.tensor.shape)}")

    result = prev_buf.tensor.clone() if isinstance(prev_buf, MmapStageBuffer) else prev_buf.tensor
    _close_stage_buffer(prev_buf)
    return result


def chunk_encode_2d(
    encoder,
    x: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int], List[int]],
    **kwargs,
) -> torch.Tensor:
    """2D chunked encode wrapper. See chunk_encode_strategy_b for details."""
    assert x.dim() == 4, f"Expected 4D tensor [B, C, H, W], got {x.dim()}D"
    return chunk_encode_strategy_b(encoder, x, chunk_latent, **kwargs)


def chunk_encode_3d(
    encoder,
    x: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    **kwargs,
) -> torch.Tensor:
    """3D chunked encode wrapper. See chunk_encode_strategy_b for details."""
    assert x.dim() == 5, f"Expected 5D tensor [B, C, D, H, W], got {x.dim()}D"
    return chunk_encode_strategy_b(encoder, x, chunk_latent, **kwargs)

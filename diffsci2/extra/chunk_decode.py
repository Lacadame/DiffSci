# chunk_decode_strategy_b_3d.py
# -----------------------------------------------------------------------------
# General 3D chunked (tiled) decode for a VAEDecoder using Strategy B:
# multi-stage, halo-propagating streaming with CPU stage buffers + periodic BCs.
#
# WHAT THIS FILE DOES (HIGH LEVEL):
#   1) We decode a huge latent volume z_latent: [B, z_dim, H, W, D] through your decoder,
#      but instead of running the full forward (which OOMs), we run it in tiles.
#   2) We split the work into "stages" (S0 .. SN). Each stage is a contiguous chunk of
#      decoder layers (S0 = post_quant+conv_in+mid, S1..S_{N-1} = each "up" stage, SN = final).
#   3) For each stage, we build the stage output *piece-by-piece* on CPU: read the minimal
#      input from the previous stage buffer with the required halo, run only this stage on GPU,
#      crop the valid center from the stage output, write that center to a CPU tensor, free GPU.
#   4) Periodicity (optional): when reading halos near boundaries, we wrap (like torus)
#      using periodic_getitem, so the convolution sees correct neighbors.
#
# WHY THIS IS EXACT (no approximation):
#   - The decoder has only local ops (3x3, stride 1, upsample) → finite receptive field.
#   - We compute the per-stage halo in LATENT units and *only* read that much at each stage.
#   - The center we write is exactly what the full-volume forward would have produced there.
#
# TENSOR LAYOUT (VERY IMPORTANT):
#   - Input z_latent is [B, z_dim, H, W, D]  (channels after batch, D last).
#   - All spatial math below uses axis order (D, H, W) in function parameters where relevant,
#     because chunk_latent is specified as (D, H, W). Internally, we always index
#     z_latent as [..., H_slice, W_slice, D_slice].
#
# YOU'LL SEE MANY COMMENTS LIKE:
#   - SHAPE: x  → explains tensor shapes at this line.
#   - WHY:   x  → explains the reason behind a step (index mapping, halo choice, etc).
#   - NOTE / TODO: things to double-check when you adapt this (e.g., periodic windows).
#
# -----------------------------------------------------------------------------


from __future__ import annotations
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import torch
from diffsci2.torchutils import periodic_getitem, periodic_setitem  # setitem unused here (writes don't wrap)


# ------------------------------- tiny helpers ------------------------------ #

def _device_of(module: torch.nn.Module) -> torch.device:
    """
    Return a sensible device for 'module' (first parameter's device).
    Fallback: cuda if available, else cpu.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_of(module: torch.nn.Module) -> torch.dtype:
    """
    Return a sensible dtype for 'module' (first parameter's dtype).
    Fallback: float32.
    """
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _norm3(v: Union[int, Tuple[int, int, int], List[int]], *, name: str) -> Tuple[int, int, int]:
    """
    Normalize an int or 3-tuple/list to a 3-tuple of ints.
    ORDERING CHOICE:
      - We accept (D, H, W) to match typical "Z,Y,X" speak.
      - Internally we always slice as [..., H, W, D], so keep this mapping in mind.
    """
    if isinstance(v, int):
        return (v, v, v)
    if isinstance(v, (tuple, list)) and len(v) == 3:
        D, H, W = int(v[0]), int(v[1]), int(v[2])
        return (D, H, W)
    raise ValueError(f"{name} must be int or 3-tuple/list (D, H, W). Got: {v!r}")


def _norm3_bool(v: Union[bool, Tuple[bool, bool, bool], List[bool]], *, name: str) -> Tuple[bool, bool, bool]:
    """
    Normalize a bool or 3-tuple/list of bools into (D, H, W).
    Same ordering convention as _norm3.
    """
    if isinstance(v, bool):
        return (v, v, v)
    if isinstance(v, (tuple, list)) and len(v) == 3:
        D, H, W = bool(v[0]), bool(v[1]), bool(v[2])
        return (D, H, W)
    raise ValueError(f"{name} must be bool or 3-tuple/list (D, H, W). Got: {v!r}")


def _make_center_spans_1d(L: int, chunk: int, radius0: int) -> List[Tuple[int, int]]:
    """
    Decide the center spans along ONE axis (in LATENT units) for Stage-0 tiling.

    INPUTS:
      L       : axis length in latent units (e.g., D=64 or H=32)
      chunk   : desired Stage-0 tile extent including halos (latent units)
      radius0 : Stage-0 halo radius (latent)

    RULES:
      - If chunk >= L  → return [(0, L)]    (no tiling in that axis).
      - Else:
          valid0 = max(1, chunk - 2*radius0)    # center width inside each tile
          then partition [0, L) by steps of 'valid0'.

    RETURNS:
      List of (cs, ce) pairs (LATENT coords) that partition [0, L).
    """
    if chunk >= L:
        return [(0, L)]
    valid0 = chunk - 2 * radius0
    if valid0 <= 0:
        # If the requested chunk is too small (halo eats it), force 1 latent cell step.
        valid0 = 1
    spans = []
    pos = 0
    while pos < L:
        end = min(pos + valid0, L)
        spans.append((pos, end))
        pos = end
    return spans


# ----------------------- RF radii and per-stage scales --------------------- #

def _compute_stage_radii_and_scales(decoder) -> Tuple[List[int], List[int]]:
    """
    For the given decoder, compute:
      - radii_latent[s]  : floor( RF_after_finishing_stage_s / 2 ) in LATENT units
      - scales_after[s]  : total spatial upsample factor (relative to latent)
                           after stage s (e.g., 1,2,4,...,2^(N-1), same for final).

    WHY:
      Strategy B depends on *stage-local* halos, which we get from differences
      of cumulative radii (see below). Cumulative radii come from an RF analysis
      of your architecture (encoder/decoder code already provides that).
    """
    cfg = decoder.config

    # IMPORTANT GUARD:
    # This code assumes NO attention (global RF) in the decoder. If there is attention,
    # exact tiling without approximation is not possible; you'd need long-range context.
    if cfg.has_mid_attn or (len(cfg.attn_resolutions) > 0):
        raise NotImplementedError("This chunked decoder assumes NO attention in the decoder.")

    info = decoder.calculate_receptive_field()
    rf_per_block = int(info["rf_per_block"])   # 4 for std blocks; 2 for minimal_rf blocks
    rf_mid       = int(info["rf_after_middle"])
    rf_final     = int(info["rf_latent"])

    num_res        = int(cfg.num_resolutions)          # number of "up" modules
    num_res_blocks = int(cfg.num_res_blocks)
    per_up_rf      = (num_res_blocks + 1) * rf_per_block  # each up has (num_blocks+1) ResBlocks

    # CUMULATIVE radii after each stage s:
    radii: List[int] = [rf_mid // 2]                  # after S0
    for s in range(1, num_res):                       # S1..S_{N-1}
        rf_s = rf_mid + per_up_rf * s
        radii.append(rf_s // 2)
    radii.append(rf_final // 2)                       # final stage SN

    # TOTAL SCALE (relative to latent) after each stage s:
    scales: List[int] = [1]
    for s in range(1, num_res):
        scales.append(2 ** s)                         # each "up" doubles spatial dims
    scales.append(2 ** max(0, num_res - 1))           # final stage doesn't upsample more

    return radii, scales


# ----------------------------- Stage runners ------------------------------- #
# (These just peel off the exact pieces of the decoder to run per stage.)

@torch.inference_mode()
def _run_stage0(decoder, z: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
    """
    RUNS: post_quant_conv -> conv_in -> mid.block_1 -> (mid.attn?) -> mid.block_2
    INPUT SHAPE:  z: [B, z_dim, h, w, d]   (z at latent resolution)
    OUTPUT SHAPE:   [B, C0,    h, w, d]
    """
    h = decoder.post_quant_conv(z)
    h = decoder.conv_in(h)
    h = decoder.mid.block_1(h, temb)
    if hasattr(decoder.mid, "attn_1"):
        h = decoder.mid.attn_1(h)
    h = decoder.mid.block_2(h, temb)
    return h


@torch.inference_mode()
def _run_up_stage(decoder, x: torch.Tensor, level_index: int, temb: Optional[torch.Tensor]) -> torch.Tensor:
    """
    RUNS: up[level_index].blocks (and atten if any) + (if level != 0) upsample x2
    INPUT SHAPE:  x: [B, C_in, h, w, d]
    OUTPUT SHAPE:   [B, C_out, h', w', d'] where (h',w',d') = (h,w,d) or doubled
    """
    up = decoder.up[level_index]
    h = x
    for i in range(len(up.block)):
        h = up.block[i](h, temb)
        if len(up.attn) > i:
            h = up.attn[i](h)
    if level_index != 0:
        h = up.upsample(h)  # doubles H,W,D
    return h


@torch.inference_mode()
def _run_final_stage(decoder, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
    """
    RUNS: up[0].blocks -> norm_out -> swish -> conv_out -> (tanh if set)
    INPUT SHAPE:  x: [B, C_in, h, w, d]
    OUTPUT SHAPE:   [B, out_channels, h, w, d]
    """
    up0 = decoder.up[0]
    h = x
    for i in range(len(up0.block)):
        h = up0.block[i](h, temb)
        if len(up0.attn) > i:
            h = up0.attn[i](h)
    h = decoder.norm_out(h)
    h = h * torch.sigmoid(h)  # swish activation
    h = decoder.conv_out(h)
    if getattr(decoder.config, "tanh_out", False):
        h = torch.tanh(h)
    return h


# ------------------------------ CPU stage buf ------------------------------ #

class _CPUStageBuffer:
    """
    A CPU tensor that holds the ENTIRE output of some stage.
    SHAPE: [B, C, H_stage, W_stage, D_stage].
    We fill it incrementally by writing the valid centers from tiles.
    """
    def __init__(self, shape: Tuple[int, int, int, int, int], dtype: torch.dtype):
        self.tensor = torch.zeros(shape, dtype=dtype, device="cpu")

    def write_block(self,
                    z0: int, z1: int,
                    y0: int, y1: int,
                    x0: int, x1: int,
                    tile: torch.Tensor):
        """
        Copy 'tile' into the destination coordinates.
        EXPECTED tile shape: [B, C, (y1-y0), (x1-x0), (z1-z0)].
        """
        self.tensor[..., y0:y1, x0:x1, z0:z1].copy_(tile.detach().to("cpu"))

    def read_block_periodic(
        self,
        z0: int, z1: int,
        y0: int, y1: int,
        x0: int, x1: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Read a block from this stage with periodic wrapping.
        - Indices z0:z1, y0:y1, x0:x1 MAY be outside [0, size). We wrap using periodic_getitem.
        - We reshape [B, C, H, W, D] → [B*C, H, W, D] so "channels" are first.

        RETURNS: [B, C, (y1-y0), (x1-x0), (z1-z0)] on 'device' with 'dtype'.
        """
        B, C, Hs, Ws, Ds = self.tensor.shape

        # SHAPE: [B*C, H, W, D]
        flat = self.tensor.reshape(B * C, Hs, Ws, Ds)

        # Build slices (possibly out-of-range). periodic_getitem will wrap.
        sy = slice(y0, y1, None)  # H axis (second)
        sx = slice(x0, x1, None)  # W axis (third)
        sz = slice(z0, z1, None)  # D axis (fourth)

        # WRAPPED FETCH on spatial dims (channels=first dim)
        sub = periodic_getitem(flat, slice(None), sy, sx, sz)

        # Back to [B, C, h, w, d]
        sub = sub.reshape(B, C, sub.shape[1], sub.shape[2], sub.shape[3])
        return sub.to(device=device, dtype=dtype, non_blocking=False).contiguous()


# -------------------------- Configuration & Data structures ---------------- #

@dataclass
class ChunkDecodeConfig:
    """
    Immutable configuration we compute once before running stages.
    Keeps the tiling plan, radii, scales, periodic flags, etc.
    """
    device: torch.device
    model_dtype: torch.dtype
    B: int
    zC: int
    H: int
    W: int
    D: int
    chD: int
    chH: int
    chW: int
    num_stages: int
    radii_latent: List[int]
    scales_after: List[int]
    delta_r_lat: List[int]
    spans_D: List[Tuple[int, int]]
    spans_H: List[Tuple[int, int]]
    spans_W: List[Tuple[int, int]]
    capD: Optional[int]
    capH: Optional[int]
    capW: Optional[int]
    periodicD: bool
    periodicH: bool
    periodicW: bool
    debug: bool


@dataclass
class SubTileRange:
    """
    A sub-range (in LATENT units) INSIDE a center block.
    We split a center block into sub-tiles if needed to keep stage tensors small.
    """
    z0: int; z1: int   # D axis in LATENT coords
    y0: int; y1: int   # H axis in LATENT coords
    x0: int; x1: int   # W axis in LATENT coords


@dataclass
class ReadWindow:
    """
    A read window for fetching input to a stage.
    - *_lat are in LATENT units (may be outside 0..L if periodic).
    - *_src are the same window scaled into the SOURCE grid (prev stage).
    """
    # Latent (possibly out-of-range if periodic)
    rsD_lat: int; reD_lat: int
    rsH_lat: int; reH_lat: int
    rsW_lat: int; reW_lat: int
    # Source (prev-stage) coords (just multiply by src_scale)
    rsD_src: int; reD_src: int
    rsH_src: int; reH_src: int
    rsW_src: int; reW_src: int


@dataclass
class TileCropCoords:
    """
    How to crop the stage tile output (y_tile) to its valid center, and where
    to place that center inside the destination CPU stage buffer.
    """
    # Tile offsets (where to crop from y_tile)
    yD_start: int; yD_end: int
    yH_start: int; yH_end: int
    yW_start: int; yW_end: int
    # Global write coords in DESTINATION stage buffer
    gD0: int; gD1: int
    gH0: int; gH1: int
    gW0: int; gW1: int


# -------------------------- Setup & Helper Functions ----------------------- #

def _setup_chunk_decode_config(
    decoder,
    z_latent: torch.Tensor,
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    device: Optional[Union[str, torch.device]],
    max_stage_out_chunk: Optional[Union[int, Tuple[int, int, int], List[int]]],
    periodicity: Union[bool, Tuple[bool, bool, bool], List[bool]],
    debug: bool
) -> ChunkDecodeConfig:
    """
    Gather all static info for the run (shapes, radii, scales, spans, caps, flags).
    """
    # Resolve device/dtype
    if device is None:
        device = _device_of(decoder)
    else:
        device = torch.device(device)
    model_dtype = _dtype_of(decoder)

    # SHAPE: z_latent = [B, z_dim, H, W, D]
    assert z_latent.dim() == 5, "z_latent must be [B, z_dim, H, W, D]"
    B, zC, H, W, D = z_latent.shape

    # Normalize chunk sizes and periodic flags (in (D, H, W) order).
    chD, chH, chW = _norm3(chunk_latent, name="chunk_latent")
    perD, perH, perW = _norm3_bool(periodicity, name="periodicity")

    # Stage cumulative radii & scales
    radii_latent, scales_after = _compute_stage_radii_and_scales(decoder)
    num_res    = int(decoder.config.num_resolutions)
    num_stages = num_res + 1  # S0..SN

    # STAGE-LOCAL radii (what each stage adds) in LATENT units:
    delta_r_lat = []
    for s in range(num_stages):
        prev = radii_latent[s - 1] if s > 0 else 0
        delta_r_lat.append(max(0, radii_latent[s] - prev))

    # Stage-0 center spans per axis:
    spans_D = _make_center_spans_1d(D, chD, radii_latent[0])
    spans_H = _make_center_spans_1d(H, chH, radii_latent[0])
    spans_W = _make_center_spans_1d(W, chW, radii_latent[0])

    # Per-stage output caps (to avoid large y_tile tensors). Normalize to (D,H,W) in DEST units.
    if max_stage_out_chunk is None:
        capD, capH, capW = None, None, None
    else:
        cD, cH, cW = _norm3(max_stage_out_chunk, name="max_stage_out_chunk")
        capD, capH, capW = int(cD), int(cH), int(cW)

    if debug:
        print(f"Latent H,W,D: H={H}, W={W}, D={D}")
        print("radii_latent =", radii_latent)
        print("delta_r_lat  =", delta_r_lat)
        print("scales_after =", scales_after)
        print("spans_D =", spans_D)
        print("spans_H =", spans_H)
        print("spans_W =", spans_W)
        print("periodicity (D,H,W) =", (perD, perH, perW))
        print("caps (dest units):", (capD, capH, capW))

    return ChunkDecodeConfig(
        device=device, model_dtype=model_dtype,
        B=B, zC=zC, H=H, W=W, D=D,
        chD=chD, chH=chH, chW=chW,
        num_stages=num_stages,
        radii_latent=radii_latent,
        scales_after=scales_after,
        delta_r_lat=delta_r_lat,
        spans_D=spans_D, spans_H=spans_H, spans_W=spans_W,
        capD=capD, capH=capH, capW=capW,
        periodicD=perD, periodicH=perH, periodicW=perW,
        debug=debug
    )


def _compute_sub_tile_size(center_len_lat: int, cap_dest: Optional[int], dest_scale: int) -> int:
    """
    For a given center length (LATENT) and per-stage cap (DEST units), return the sub-tile length in LATENT.
    We enforce: (sub_len_lat * dest_scale) <= cap_dest   if cap provided.
    """
    if cap_dest is None:
        return center_len_lat
    return max(1, min(center_len_lat, cap_dest // max(dest_scale, 1)))


def _generate_sub_tiles(
    csD: int, ceD: int,
    csH: int, ceH: int,
    csW: int, ceW: int,
    dest_scale: int,
    capD: Optional[int],
    capH: Optional[int],
    capW: Optional[int]
) -> List[SubTileRange]:
    """
    Split a center block (cs:ce per axis, LATENT units) into sub-tiles so that the
    per-stage tile output never exceeds the cap in DEST units.
    """
    lenD_lat = ceD - csD
    lenH_lat = ceH - csH
    lenW_lat = ceW - csW

    subD = _compute_sub_tile_size(lenD_lat, capD, dest_scale)
    subH = _compute_sub_tile_size(lenH_lat, capH, dest_scale)
    subW = _compute_sub_tile_size(lenW_lat, capW, dest_scale)

    sub_tiles = []
    z0 = csD
    while z0 < ceD:
        z1 = min(z0 + subD, ceD)
        y0 = csH
        while y0 < ceH:
            y1 = min(y0 + subH, ceH)
            x0 = csW
            while x0 < ceW:
                x1 = min(x0 + subW, ceW)
                sub_tiles.append(SubTileRange(z0, z1, y0, y1, x0, x1))
                x0 = x1
            y0 = y1
        z0 = z1

    return sub_tiles


def _compute_read_window(
    sub_tile: SubTileRange,
    stage_local_lat: int,
    src_scale: int,
    H: int, W: int, D: int,
    perD: bool, perH: bool, perW: bool
) -> ReadWindow:
    """
    Compute the READ WINDOW for this sub-tile at this stage.

    INPUTS:
      sub_tile      : the sub-center [z0:z1], [y0:y1], [x0:x1] (LATENT units).
      stage_local_lat: how many LATENT cells this stage adds to RF on each side.
      src_scale     : scale factor from LATENT to SOURCE (prev-stage) units.
      H,W,D         : latent sizes.
      per*          : periodic flags per axis.

    IMPORTANT:
      - If an axis is periodic, we *allow* read window to go negative or > size,
        then rely on periodic_getitem to wrap.
      - If not periodic, we clamp to [0, size].
    """
    # D (last axis)
    if perD:
        # NOTE: Using modulo here collapses the window into [0..D) and loses the fact
        # that the window spans across the border. It's *often* better to leave rs/re
        # as raw (possibly negative/over) and let periodic_getitem handle wrapping.
        # This code keeps modulo because it's "same code"; consider removing '%' if needed.
        # NOTE: We've removed the modulo here as suggested by the author of the code.
        rsD_lat = (sub_tile.z0 - stage_local_lat)
        reD_lat = (sub_tile.z1 + stage_local_lat)
    else:
        rsD_lat = max(0, sub_tile.z0 - stage_local_lat)
        reD_lat = min(D, sub_tile.z1 + stage_local_lat)

    # H (height)
    if perH:
        rsH_lat = (sub_tile.y0 - stage_local_lat)
        reH_lat = (sub_tile.y1 + stage_local_lat)
    else:
        rsH_lat = max(0, sub_tile.y0 - stage_local_lat)
        reH_lat = min(H, sub_tile.y1 + stage_local_lat)

    # W (width)
    if perW:
        rsW_lat = (sub_tile.x0 - stage_local_lat)
        reW_lat = (sub_tile.x1 + stage_local_lat)
    else:
        rsW_lat = max(0, sub_tile.x0 - stage_local_lat)
        reW_lat = min(W, sub_tile.x1 + stage_local_lat)

    # SCALE to source coords
    return ReadWindow(
        rsD_lat=rsD_lat, reD_lat=reD_lat,
        rsH_lat=rsH_lat, reH_lat=reH_lat,
        rsW_lat=rsW_lat, reW_lat=reW_lat,
        rsD_src=rsD_lat * src_scale,
        reD_src=reD_lat * src_scale,
        rsH_src=rsH_lat * src_scale,
        reH_src=reH_lat * src_scale,
        rsW_src=rsW_lat * src_scale,
        reW_src=reW_lat * src_scale,
    )


def _compute_tile_crop_coords(
    sub_tile: SubTileRange,
    read_win: ReadWindow,
    src_scale: int,
    dest_scale: int,
    up_factor: int,
    *,
    # latent sizes and periodic flags (per axis)
    D_lat: int, H_lat: int, W_lat: int,
    perD: bool, perH: bool, perW: bool,
) -> TileCropCoords:
    """
    Compute how much of y_tile to keep (the valid center) and where to place it
    in the destination buffer, with correct handling of periodic reads.

    KEY IDEA (per axis):
      - The tile 'x_in' was fetched starting at read_win.rs*_lat (latent coords),
        possibly wrapping across the border. Inside this tile, offsets must be
        measured along the wrapped sequence that starts at rs*_lat.
      - So we compute distances modulo the axis length if that axis is periodic.

      Let wrap_delta(start, target, size, periodic):
        returns (target - start)          if not periodic
                (target - start) % size   if periodic

      Then:
        left_lat  = wrap_delta(rs_lat,  center_start,  size, periodic)
        right_lat = wrap_delta(rs_lat,  center_end,    size, periodic)

        left_src  = left_lat  * src_scale
        right_src = right_lat * src_scale

        y_start   = left_src  * up_factor
        y_end     = right_src * up_factor

        g0        = center_start * dest_scale
        g1        = center_end   * dest_scale
    """

    def wrap_delta(start: int, target: int, size: int, periodic: bool) -> int:
        return ((target - start) % size) if periodic else (target - start)

    # ---- D (last axis) ----
    leftD_lat  = wrap_delta(read_win.rsD_lat, sub_tile.z0, D_lat, perD)
    rightD_lat = wrap_delta(read_win.rsD_lat, sub_tile.z1, D_lat, perD)
    leftD_src  = leftD_lat  * src_scale
    rightD_src = rightD_lat * src_scale
    yD_start   = leftD_src  * up_factor
    yD_end     = rightD_src * up_factor
    gD0        = sub_tile.z0 * dest_scale
    gD1        = sub_tile.z1 * dest_scale

    # ---- H (height) ----
    leftH_lat  = wrap_delta(read_win.rsH_lat, sub_tile.y0, H_lat, perH)
    rightH_lat = wrap_delta(read_win.rsH_lat, sub_tile.y1, H_lat, perH)
    leftH_src  = leftH_lat  * src_scale
    rightH_src = rightH_lat * src_scale
    yH_start   = leftH_src  * up_factor
    yH_end     = rightH_src * up_factor
    gH0        = sub_tile.y0 * dest_scale
    gH1        = sub_tile.y1 * dest_scale

    # ---- W (width) ----
    leftW_lat  = wrap_delta(read_win.rsW_lat, sub_tile.x0, W_lat, perW)
    rightW_lat = wrap_delta(read_win.rsW_lat, sub_tile.x1, W_lat, perW)
    leftW_src  = leftW_lat  * src_scale
    rightW_src = rightW_lat * src_scale
    yW_start   = leftW_src  * up_factor
    yW_end     = rightW_src * up_factor
    gW0        = sub_tile.x0 * dest_scale
    gW1        = sub_tile.x1 * dest_scale

    # Safety: lengths must match along each axis.
    assert (yD_end - yD_start) == (gD1 - gD0), "D length mismatch"
    assert (yH_end - yH_start) == (gH1 - gH0), "H length mismatch"
    assert (yW_end - yW_start) == (gW1 - gW0), "W length mismatch"

    return TileCropCoords(
        yD_start=yD_start, yD_end=yD_end,
        yH_start=yH_start, yH_end=yH_end,
        yW_start=yW_start, yW_end=yW_end,
        gD0=gD0, gD1=gD1,
        gH0=gH0, gH1=gH1,
        gW0=gW0, gW1=gW1,
    )


# ------------------------------ Main entry --------------------------------- #

@torch.inference_mode()
def chunk_decode_strategy_b_3d(  # noqa: C901 (complex by design; heavily commented)
    decoder,                    # your VAEDecoder
    z_latent: torch.Tensor,     # [B, z_dim, H, W, D] (CPU or GPU)
    chunk_latent: Union[int, Tuple[int, int, int], List[int]],
    *,
    # Compute device for each per-stage tile:
    device: Optional[Union[str, torch.device]] = None,
    # Optional time embedding input (forwarded to each stage):
    time: Optional[torch.Tensor] = None,
    # Debug prints (heavy):
    debug: bool = False,
    # Cap any per-stage CUDA tensor spatial size (in that stage's OUTPUT units).
    # Accepts int or (D_out, H_out, W_out).
    max_stage_out_chunk: Optional[Union[int, Tuple[int, int, int], List[int]]] = 128,
    # NEW: periodicity flags in order (D, H, W). True means wrap that axis.
    periodicity: Union[bool, Tuple[bool, bool, bool], List[bool]] = False,
) -> torch.Tensor:
    """
    GENERAL 3D CHUNKED DECODE with Strategy B + optional periodic reads.

    NOTE ABOUT PERIODICITY IMPLEMENTATION:
      - The *reads* use periodic_getitem on [B*C, H, W, D] views.
      - The computation of read windows for periodic axes uses modulo here (same code).
        If your periodic_getitem expects raw (possibly negative) slice bounds, consider
        removing the '%' in _compute_read_window so [rs, re] can be negative/over.
        Then let periodic_getitem do the wrapping, which is usually safer.
    """
    # Build configuration (tiling plan, radii, scales, caps, flags)
    cfg = _setup_chunk_decode_config(
        decoder, z_latent, chunk_latent, device, max_stage_out_chunk, periodicity, debug
    )
    num_res = int(decoder.config.num_resolutions)

    # Choose which chunk of the decoder to run per stage 's'
    def run_stage(s: int, x: torch.Tensor) -> torch.Tensor:
        if s == 0:
            return _run_stage0(decoder, x, time)
        elif 1 <= s <= (num_res - 1):
            level_index = (num_res - s)  # maps s=1..N-1 to level=(N-1..1)
            return _run_up_stage(decoder, x, level_index, time)
        else:
            return _run_final_stage(decoder, x, time)

    # CPU buffers: one per stage, allocated lazily when we see the first y_tile
    stage_bufs: List[Optional[_CPUStageBuffer]] = [None] * cfg.num_stages
    prev_scale = 1  # total scale after previous stage
    prev_buf: Optional[_CPUStageBuffer] = None

    # Ensure eval-only mode for the decoder; restore at the end
    was_training = decoder.training
    decoder.eval()

    # ========================== MAIN STAGE LOOP ============================
    for s in range(cfg.num_stages):
        stage_local_lat = cfg.delta_r_lat[s]    # how many LATENT cells this stage adds to RF
        dest_scale      = cfg.scales_after[s]   # total scale after this stage
        # BUG/NOTE: original line had "src_scale = crop = prev_scale". The 'crop' is unused.
        # Keeping the same tokens but be aware it's just 'src_scale = prev_scale'.
        src_scale       = prev_scale
        up_factor       = dest_scale // src_scale  # either 1 or 2 (per-stage upsample)

        if cfg.debug:
            print(f"\n=== Stage {s} ===")
            print(f"radius_total_lat={cfg.radii_latent[s]}, stage_local_lat={stage_local_lat}, "
                  f"src_scale={src_scale}, dest_scale={dest_scale}, up_factor={up_factor}")

        dest_buf = stage_bufs[s]
        dest_created = dest_buf is not None  # false until we allocate

        # OUTER LOOPS over Stage-0 center blocks in (D,H,W).
        # We then sub-tile each block if the DEST cap would be exceeded.
        for (csD, ceD) in cfg.spans_D:
            for (csH, ceH) in cfg.spans_H:
                for (csW, ceW) in cfg.spans_W:

                    # Compute sub-tile sizes (in LATENT units) so that
                    # (sub_len_lat * dest_scale) <= cap along each axis (if provided).
                    def _sub_len_lat(center_len_lat: int, cap_dest: Optional[int]) -> int:
                        if cap_dest is None:
                            return center_len_lat
                        return max(1, min(center_len_lat, cap_dest // max(dest_scale, 1)))

                    lenD_lat = ceD - csD
                    lenH_lat = ceH - csH
                    lenW_lat = ceW - csW
                    subD = _sub_len_lat(lenD_lat, cfg.capD)
                    subH = _sub_len_lat(lenH_lat, cfg.capH)
                    subW = _sub_len_lat(lenW_lat, cfg.capW)

                    if cfg.debug:
                        print(f"Stage {s} subD: {subD}, subH: {subH}, subW: {subW}")
                        print(f"Stage {s} csD: {csD}, ceD: {ceD}, csH: {csH}, ceH: {ceH}, csW: {csW}, ceW: {ceW}")
                        print(f"Stage {s} stage_local_lat: {stage_local_lat}, src_scale: {src_scale}, dest_scale: {dest_scale}, up_factor: {up_factor}")

                    # INNER LOOPS over sub-tiles (LATENT coords)
                    z0 = csD
                    while z0 < ceD:
                        z1 = min(z0 + subD, ceD)
                        y0 = csH
                        while y0 < ceH:
                            y1 = min(y0 + subH, ceH)
                            x0 = csW
                            while x0 < ceW:
                                x1 = min(x0 + subW, ceW)
                                sub_tile = SubTileRange(z0, z1, y0, y1, x0, x1)

                                # --- READ WINDOW (in LATENT and SOURCE units)
                                # NOTE on periodic axes:
                                #   - This function currently applies '% size' (same code you posted),
                                #     which loses the fact a window can cross borders. If you want
                                #     to *fully* delegate to periodic_getitem, remove the modulo and
                                #     let rs/re be negative/over. periodic_getitem will wrap both sides.
                                read_win = _compute_read_window(
                                    sub_tile, stage_local_lat, src_scale,
                                    cfg.H, cfg.W, cfg.D,
                                    cfg.periodicD, cfg.periodicH, cfg.periodicW
                                )

                                # --- FETCH SOURCE BLOCK (periodic_getitem) ---
                                if s == 0:
                                    # SOURCE is z_latent @ LATENT scale
                                    # SHAPE: z_latent [B, zC, H, W, D]  -> flatten channels to [B*zC, H, W, D]
                                    B0, C0, Hs, Ws, Ds = z_latent.shape
                                    flat = z_latent.reshape(B0 * C0, Hs, Ws, Ds)

                                    # Build possibly out-of-range slices for H,W,D (periodic axes wrap)
                                    sy = slice(read_win.rsH_lat, read_win.reH_lat, None)
                                    sx = slice(read_win.rsW_lat, read_win.reW_lat, None)
                                    sz = slice(read_win.rsD_lat, read_win.reD_lat, None)

                                    # periodic_getitem on [B*zC, H, W, D]
                                    x_in_flat = periodic_getitem(flat, slice(None), sy, sx, sz)

                                    # Back to [B, zC, h, w, d] on the target device/dtype
                                    x_in = x_in_flat.reshape(B0, C0, x_in_flat.shape[1], x_in_flat.shape[2], x_in_flat.shape[3])
                                    x_in = x_in.to(device=cfg.device, dtype=z_latent.dtype, non_blocking=False).contiguous()
                                else:
                                    # SOURCE is previous stage output (CPU buffer), currently [B, C, Hs, Ws, Ds].
                                    # Flatten to [B*C, Hs, Ws, Ds] so periodic_getitem sees channels first.
                                    Bp, Cp, Hs, Ws, Ds = prev_buf.tensor.shape
                                    flat = prev_buf.tensor.reshape(Bp * Cp, Hs, Ws, Ds)

                                    # Build slices in SOURCE units
                                    sy = slice(read_win.rsH_src, read_win.reH_src, None)
                                    sx = slice(read_win.rsW_src, read_win.reW_src, None)
                                    sz = slice(read_win.rsD_src, read_win.reD_src, None)

                                    x_in_flat = periodic_getitem(flat, slice(None), sy, sx, sz)

                                    # Back to [B, C, h, w, d]
                                    x_in = x_in_flat.reshape(Bp, Cp, x_in_flat.shape[1], x_in_flat.shape[2], x_in_flat.shape[3])
                                    x_in = x_in.to(device=cfg.device, dtype=cfg.model_dtype, non_blocking=False).contiguous()

                                # --- RUN ONLY THIS STAGE ---
                                y_tile = run_stage(s, x_in)  # SHAPE: [B, C_out, h*, w*, d*] in this stage

                                # --- ALLOCATE DEST BUFFER (for whole stage) ON FIRST TILE ---
                                if not dest_created:
                                    B_, C_out = cfg.B, int(y_tile.shape[1])
                                    Hs_out = cfg.H * dest_scale
                                    Ws_out = cfg.W * dest_scale
                                    Ds_out = cfg.D * dest_scale
                                    stage_bufs[s] = _CPUStageBuffer(
                                        shape=(B_, C_out, Hs_out, Ws_out, Ds_out),
                                        dtype=y_tile.dtype
                                    )
                                    dest_buf = stage_bufs[s]
                                    dest_created = True
                                    if cfg.debug:
                                        print(f"[S{s}] Alloc dest buffer shape={(B_, C_out, Hs_out, Ws_out, Ds_out)}")

                                # --- CROP VALID CENTER FROM y_tile & WRITE TO CPU DEST ---
                                crop = _compute_tile_crop_coords(
                                    sub_tile, read_win, src_scale, dest_scale, up_factor,
                                    D_lat=cfg.D, H_lat=cfg.H, W_lat=cfg.W,
                                    perD=cfg.periodicD, perH=cfg.periodicH, perW=cfg.periodicW,
                                )

                                if cfg.debug:
                                    print('--------------------------------')
                                    print(f"[S{s}] Crop Coords:"
                                          f" H_start: {crop.yH_start}, H_end: {crop.yH_end},"
                                          f" W_start: {crop.yW_start}, W_end: {crop.yW_end},"
                                          f" D_start: {crop.yD_start}, D_end: {crop.yD_end}")

                                    # Additional analytics
                                    print(f"[S{s}] Global Coords:"
                                          f" gH0: {crop.gH0}, gH1: {crop.gH1},"
                                          f" gW0: {crop.gW0}, gW1: {crop.gW1},"
                                          f" gD0: {crop.gD0}, gD1: {crop.gD1}")
                                    print(f"[S{s}] Tile Shape:"
                                          f" Height: {crop.yH_end - crop.yH_start},"
                                          f" Width: {crop.yW_end - crop.yW_start},"
                                          f" Depth: {crop.yD_end - crop.yD_start}")
                                    print('--------------------------------')
                                # Slice valid center (note axis order: [..., H, W, D] in the tail)
                                y_center = y_tile[...,
                                                  crop.yH_start:crop.yH_end,
                                                  crop.yW_start:crop.yW_end,
                                                  crop.yD_start:crop.yD_end]

                                # Write to CPU buffer in global DEST coords
                                dest_buf.write_block(
                                    z0=crop.gD0, z1=crop.gD1,
                                    y0=crop.gH0, y1=crop.gH1,
                                    x0=crop.gW0, x1=crop.gW1,
                                    tile=y_center
                                )

                                # --- FREE ASAP (limit GPU memory) ---
                                del y_tile, y_center, x_in, x_in_flat
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize(cfg.device)
                                    torch.cuda.empty_cache()

                                # Advance sub-tiles in W
                                x0 = x1
                            # Advance sub-tiles in H
                            y0 = y1
                        # Advance sub-tiles in D
                        z0 = z1

        # NEXT STAGE will read from this stage's buffer
        prev_scale = dest_scale
        prev_buf   = dest_buf

    # Restore training flag and return final stage output (CPU)
    decoder.train(was_training)
    return stage_bufs[-1].tensor


# ------------------------------ Example (off) ------------------------------ #
if __name__ == "__main__":
    # Example usage (disabled by default):
    #
    # from yourmodule import VAENetConfig, VAEDecoder
    # cfg = VAENetConfig(
    #     dimension=3, in_channels=1, out_channels=1, z_dim=4, z_channels=4,
    #     ch=32, ch_mult=[1,2,4,4], num_res_blocks=2,
    #     attn_type="none", has_mid_attn=False, patch_size=None
    # )
    # dec = VAEDecoder(cfg).eval().to("cuda:0")
    #
    # z = torch.randn(1, cfg.z_dim, 32, 32, 64)  # [B,zC,H,W,D]
    #
    # y = chunk_decode_strategy_b_3d(
    #     dec, z,
    #     chunk_latent=(16, 64, 64),        # tile in D only; (D,H,W) in LATENT units
    #     device="cuda:0",
    #     debug=True,
    #     max_stage_out_chunk=(64, 128, 128),
    #     periodicity=(True, False, False),  # wrap only D axis
    # )
    #
    pass

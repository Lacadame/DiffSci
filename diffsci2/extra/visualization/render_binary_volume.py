"""
Efficient 3D binary volume renderer for large porous media volumes.

Two rendering modes:
  - "volume" (default): GPU volume ray casting — fast, shows pore structure.
  - "surface": VTK FlyingEdges isosurface — classic solid surface look.

Works on headless servers (offscreen VTK/PyVista). Handles volumes up to
1024^2 x 4096 by memory-mapping and stride-downsampling.

Usage (CLI):
    python -m diffsci2.extra.visualization.render_binary_volume volume.npy -o out.png
    python -m diffsci2.extra.visualization.render_binary_volume volume.npy --invert --square
    python -m diffsci2.extra.visualization.render_binary_volume volume.npy --zoom 0.7 --pan 0.05 -0.02

Usage (Python / Jupyter):
    from diffsci2.extra.visualization import render_binary_volume
    render_binary_volume(vol, "out.png", invert=True, zoom=0.7, pan=(0.05, -0.02))
"""

import argparse
import time
from pathlib import Path

import numpy as np


def render_binary_volume(
    volume,
    output_path="volume_render.png",
    invert=False,
    mode="volume",
    max_size=384,
    # Camera
    zoom=1.0,
    elevation=30.0,
    azimuth=45.0,
    pan=(0.0, 0.0),
    camera_position=None,
    # Image
    window_size=(1080, 1080),
    background_color="white",
    crop=None,
    # Appearance
    mesh_color="#c2956b",
    mesh_opacity=1.0,
    smooth_sigma=0.0,
    # Lighting
    ambient=0.3,
    diffuse=0.7,
    specular=0.2,
    # Surface mode extras
    pbr=True,
    metallic=0.05,
    roughness=0.6,
    # Widgets
    show_axes=False,
    show_bounds=False,
    verbose=True,
):
    """
    Render a binary 3D volume and save to PNG.

    Parameters
    ----------
    volume : np.ndarray or str or Path
        Binary 3D array (0/1), or path to ``.npy`` file (memory-mapped).
    output_path : str or Path
        Destination PNG.
    invert : bool
        By default renders the 1-phase. ``invert=True`` renders the 0-phase.
    mode : str
        ``"volume"`` — GPU volume ray casting (fast, shows pore depth).
        ``"surface"`` — VTK FlyingEdges isosurface (solid surface).
    max_size : int
        Stride-downsample so largest dim <= this. 0 or None to disable.

    zoom : float
        Camera zoom. <1 zooms out, >1 zooms in. Default 1.0.
    elevation : float
        Camera elevation angle in degrees (vertical tilt). Default 30.
    azimuth : float
        Camera azimuth angle in degrees (horizontal rotation). Default 45.
    pan : tuple of float
        ``(pan_x, pan_y)`` — translate the rock in the image, as fractions
        of the scene size. Positive pan_x shifts rock right, positive
        pan_y shifts rock up.
    camera_position : list or None
        Explicit PyVista camera ``[(cam), (focal), (up)]``.
        Overrides elevation/azimuth/pan if given.

    window_size : tuple of int
        Render resolution ``(width, height)`` in pixels. Default 1080x1080
        (square). The render always happens at this size; use ``crop``
        to trim afterwards.
    background_color : str
        Background colour name or hex.
    crop : tuple of int or None
        Post-render crop margins ``(left, top, right, bottom)`` in pixels.
        Removes that many pixels from each side of the rendered image.

    mesh_color : str
        Surface / volume colour (name or hex).
    mesh_opacity : float
        Opacity in [0, 1].
    smooth_sigma : float
        Gaussian sigma applied before rendering (smooths voxel edges).
        0 = no smoothing. Try 0.6-1.0 for softer look.

    ambient : float
        Ambient lighting coefficient.
    diffuse : float
        Diffuse lighting coefficient.
    specular : float
        Specular lighting coefficient.

    pbr : bool
        Use physically-based rendering (surface mode only).
    metallic : float
        PBR metallic (surface mode only).
    roughness : float
        PBR roughness (surface mode only).

    show_axes : bool
        Show XYZ orientation axes widget.
    show_bounds : bool
        Show bounding box with tick labels.
    verbose : bool
        Print progress.

    Returns
    -------
    str
        Absolute path to saved PNG.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    t0 = time.time()

    # ── Load ────────────────────────────────────────────────────────
    if isinstance(volume, (str, Path)):
        path = str(volume)
        if verbose:
            print(f"Memory-mapping {path} ...")
        volume = np.load(path, mmap_mode="r")

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {volume.shape}")

    original_shape = volume.shape
    if verbose:
        print(f"Volume shape: {original_shape}  dtype: {volume.dtype}")

    # ── Stride downsample ───────────────────────────────────────────
    stride = 1
    if max_size and max(original_shape) > max_size:
        stride = -(-max(original_shape) // max_size)
    if stride > 1:
        if verbose:
            print(f"Stride-downsampling by {stride} ...")
        volume = volume[::stride, ::stride, ::stride]

    vol = np.ascontiguousarray(volume, dtype=np.float32)
    if verbose:
        print(f"Working array: {vol.shape}  ({vol.nbytes / 1e6:.1f} MB)")

    # ── Invert ──────────────────────────────────────────────────────
    if invert:
        vol = 1.0 - vol

    # ── Optional smoothing ──────────────────────────────────────────
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        if verbose:
            print(f"Gaussian smoothing (sigma={smooth_sigma}) ...")
        vol = gaussian_filter(vol, sigma=smooth_sigma)

    # ── Render ──────────────────────────────────────────────────────
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background_color)

    render_kwargs = dict(
        color=mesh_color, opacity=mesh_opacity,
        ambient=ambient, diffuse=diffuse, specular=specular,
    )

    if mode == "volume":
        _add_volume_render(plotter, vol, stride, verbose, **render_kwargs)
    elif mode == "surface":
        _add_surface_render(
            plotter, vol, stride, verbose,
            pbr=pbr, metallic=metallic, roughness=roughness,
            **render_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'volume' or 'surface'.")

    if show_axes:
        plotter.show_axes()
    if show_bounds:
        plotter.show_bounds(grid=False)

    # ── Camera ──────────────────────────────────────────────────────
    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        center = np.array(original_shape, dtype=float) / 2.0
        d = max(original_shape) * 2.2
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        cam_pos = np.array([
            center[0] + d * np.cos(el_rad) * np.cos(az_rad),
            center[1] + d * np.cos(el_rad) * np.sin(az_rad),
            center[2] + d * np.sin(el_rad),
        ])

        # Pan: shift both camera and focal point in screen-space
        pan_x, pan_y = pan
        if pan_x != 0 or pan_y != 0:
            view_dir = center - cam_pos
            view_dir /= np.linalg.norm(view_dir)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(view_dir, up)
            right /= np.linalg.norm(right)
            screen_up = np.cross(right, view_dir)
            shift = pan_x * d * right + pan_y * d * screen_up
            cam_pos += shift
            center = center + shift

        plotter.camera_position = [
            tuple(cam_pos), tuple(center), (0, 0, 1),
        ]
        plotter.camera.view_angle = 30.0

    if zoom != 1.0:
        plotter.camera.zoom(zoom)

    output_path = str(Path(output_path).resolve())
    plotter.screenshot(output_path)
    plotter.close()

    # ── Post-render crop ────────────────────────────────────────────
    if crop is not None:
        from PIL import Image
        left, top, right, bottom = crop
        img = Image.open(output_path)
        w, h = img.size
        img = img.crop((left, top, w - right, h - bottom))
        img.save(output_path)
        if verbose:
            print(f"Cropped to {img.size[0]}x{img.size[1]}")

    elapsed = time.time() - t0
    if verbose:
        print(f"Saved to {output_path}  ({elapsed:.1f}s)")
    return output_path


# ── Render backends ─────────────────────────────────────────────────


def _add_volume_render(plotter, vol, stride, verbose, *,
                       color, opacity, ambient, diffuse, specular):
    """GPU volume ray casting — no mesh extraction needed."""
    import pyvista as pv

    if verbose:
        print("GPU volume ray casting ...")
    grid = pv.ImageData(dimensions=vol.shape, spacing=(stride, stride, stride))
    grid.point_data["values"] = vol.ravel(order="F")

    plotter.add_volume(
        grid,
        scalars="values",
        opacity=[0, 0, 0.0, opacity, opacity],
        cmap=[color, color],
        shade=True,
        ambient=ambient,
        diffuse=diffuse,
        specular=specular,
        show_scalar_bar=False,
    )


def _add_surface_render(plotter, vol, stride, verbose, *,
                        color, opacity, ambient, diffuse, specular,
                        pbr, metallic, roughness):
    """VTK FlyingEdges isosurface — solid surface look."""
    import pyvista as pv

    if verbose:
        print("VTK FlyingEdges isosurface ...")
    grid = pv.ImageData(dimensions=vol.shape)
    grid.point_data["values"] = vol.ravel(order="F")
    mesh = grid.contour([0.5], scalars="values", method="flying_edges")
    if verbose:
        print(f"Mesh: {mesh.n_cells:,} faces")

    if stride > 1:
        mesh.points *= stride

    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        smooth_shading=True,
        pbr=pbr,
        metallic=metallic,
        roughness=roughness,
    )
    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.4))


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="Render a binary 3D .npy volume to PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to binary .npy volume")
    p.add_argument("-o", "--output", default=None,
                   help="Output PNG path (default: <input>_render.png)")

    g = p.add_argument_group("phase selection")
    g.add_argument("--invert", action="store_true",
                   help="Render the 0-phase instead of 1-phase")

    g = p.add_argument_group("render mode")
    g.add_argument("--mode", choices=["volume", "surface"], default="volume",
                   help="'volume'=GPU ray cast, 'surface'=isosurface")
    g.add_argument("--max-size", type=int, default=384,
                   help="Downsample so largest dim <= this (0=off)")

    g = p.add_argument_group("camera")
    g.add_argument("--zoom", type=float, default=1.0,
                   help="<1 zooms out, >1 zooms in")
    g.add_argument("--elevation", type=float, default=30.0,
                   help="Vertical tilt in degrees")
    g.add_argument("--azimuth", type=float, default=45.0,
                   help="Horizontal rotation in degrees")
    g.add_argument("--pan", type=float, nargs=2, default=[0, 0],
                   metavar=("PX", "PY"),
                   help="Translate rock in image (fractions of scene size)")

    g = p.add_argument_group("image")
    g.add_argument("--size", type=int, default=None,
                   help="Square image size (shorthand for --width S --height S)")
    g.add_argument("--width", type=int, default=1080, help="Image width px")
    g.add_argument("--height", type=int, default=1080, help="Image height px")
    g.add_argument("--background", default="white", help="Background colour")
    g.add_argument("--crop", type=int, nargs=4, default=None,
                   metavar=("L", "T", "R", "B"),
                   help="Crop margins in pixels: left top right bottom")

    g = p.add_argument_group("appearance")
    g.add_argument("--color", default="#c2956b", help="Mesh/volume colour")
    g.add_argument("--opacity", type=float, default=1.0, help="Opacity [0-1]")
    g.add_argument("--smooth-sigma", type=float, default=0.0,
                   help="Gaussian smooth sigma (0=off, try 0.6-1.0)")
    g.add_argument("--ambient", type=float, default=0.3, help="Ambient light")
    g.add_argument("--diffuse", type=float, default=0.7, help="Diffuse light")
    g.add_argument("--specular", type=float, default=0.2, help="Specular light")
    g.add_argument("--metallic", type=float, default=0.05,
                   help="PBR metallic (surface mode)")
    g.add_argument("--roughness", type=float, default=0.6,
                   help="PBR roughness (surface mode)")

    g = p.add_argument_group("widgets")
    g.add_argument("--axes", action="store_true", help="Show XYZ axes")
    g.add_argument("--bounds", action="store_true", help="Show bounding box")

    args = p.parse_args()

    if args.output is None:
        args.output = str(Path(args.input).stem) + "_render.png"

    if args.size is not None:
        w = h = args.size
    else:
        w, h = args.width, args.height

    render_binary_volume(
        volume=args.input,
        output_path=args.output,
        invert=args.invert,
        mode=args.mode,
        max_size=args.max_size or None,
        zoom=args.zoom,
        elevation=args.elevation,
        azimuth=args.azimuth,
        pan=tuple(args.pan),
        window_size=(w, h),
        background_color=args.background,
        crop=tuple(args.crop) if args.crop else None,
        mesh_color=args.color,
        mesh_opacity=args.opacity,
        smooth_sigma=args.smooth_sigma,
        ambient=args.ambient,
        diffuse=args.diffuse,
        specular=args.specular,
        metallic=args.metallic,
        roughness=args.roughness,
        show_axes=args.axes,
        show_bounds=args.bounds,
    )


if __name__ == "__main__":
    main()

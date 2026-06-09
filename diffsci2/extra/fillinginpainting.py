from typing import Literal, Any
from jaxtyping import Float
from torch import Tensor
import torch
import numpy as np

from diffsci2.torchutils import periodic_getitem, periodic_setitem


def _get_grid_generation_order(grid_map: list[int]) -> tuple[list[tuple[int, int, int]], int]:
    """
    Returns positions in generation order based on coordinate parity patterns.

    The generation order follows 8 patterns based on whether each coordinate is even or odd:
    1. (even, even, even): positions (i*2, j*2, k*2) - all coordinates even
    2. (even, even, odd): positions (i*2, j*2, k*2+1) - x,y even, z odd
    3. (even, odd, even): positions (i*2, j*2+1, k*2) - x,z even, y odd
    4. (even, odd, odd): positions (i*2, j*2+1, k*2+1) - x even, y,z odd
    5. (odd, even, even): positions (i*2+1, j*2, k*2) - x odd, y,z even
    6. (odd, even, odd): positions (i*2+1, j*2, k*2+1) - x,z odd, y even
    7. (odd, odd, even): positions (i*2+1, j*2+1, k*2) - x,y odd, z even
    8. (odd, odd, odd): positions (i*2+1, j*2+1, k*2+1) - all coordinates odd

    Within each pattern, positions are generated in lexicographical order.

    Args:
        grid_map: [nx, ny, nz] - number of grid steps in each direction

    Returns:
        Tuple of:
        - List of (i, j, k) tuples in generation order
        - Integer count of positions in the first pattern (all even) - this is corner_inds_limit
    """
    nx, ny, nz = grid_map
    positions = []

    # Pattern 1: (even, even, even) - (i*2, j*2, k*2)
    pattern1 = []
    for i in range((nx + 1) // 2):  # i*2 < nx, so i < (nx+1)//2
        for j in range((ny + 1) // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2, j * 2, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern1.append(pos)
    pattern1.sort()  # Lexicographical order
    positions.extend(pattern1)
    corner_inds_limit = len(pattern1)

    # Pattern 2: (even, even, odd) - (i*2, j*2, k*2+1)
    pattern2 = []
    for i in range((nx + 1) // 2):
        for j in range((ny + 1) // 2):
            for k in range(nz // 2):  # k*2+1 < nz, so k < (nz-1+1)//2 = nz//2
                pos = (i * 2, j * 2, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern2.append(pos)
    pattern2.sort()
    positions.extend(pattern2)

    # Pattern 3: (even, odd, even) - (i*2, j*2+1, k*2)
    pattern3 = []
    for i in range((nx + 1) // 2):
        for j in range(ny // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2, j * 2 + 1, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern3.append(pos)
    pattern3.sort()
    positions.extend(pattern3)

    # Pattern 4: (even, odd, odd) - (i*2, j*2+1, k*2+1)
    pattern4 = []
    for i in range((nx + 1) // 2):
        for j in range(ny // 2):
            for k in range(nz // 2):
                pos = (i * 2, j * 2 + 1, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern4.append(pos)
    pattern4.sort()
    positions.extend(pattern4)

    # Pattern 5: (odd, even, even) - (i*2+1, j*2, k*2)
    pattern5 = []
    for i in range(nx // 2):
        for j in range((ny + 1) // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2 + 1, j * 2, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern5.append(pos)
    pattern5.sort()
    positions.extend(pattern5)

    # Pattern 6: (odd, even, odd) - (i*2+1, j*2, k*2+1)
    pattern6 = []
    for i in range(nx // 2):
        for j in range((ny + 1) // 2):
            for k in range(nz // 2):
                pos = (i * 2 + 1, j * 2, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern6.append(pos)
    pattern6.sort()
    positions.extend(pattern6)

    # Pattern 7: (odd, odd, even) - (i*2+1, j*2+1, k*2)
    pattern7 = []
    for i in range(nx // 2):
        for j in range(ny // 2):
            for k in range((nz + 1) // 2):
                pos = (i * 2 + 1, j * 2 + 1, k * 2)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern7.append(pos)
    pattern7.sort()
    positions.extend(pattern7)

    # Pattern 8: (odd, odd, odd) - (i*2+1, j*2+1, k*2+1)
    pattern8 = []
    for i in range(nx // 2):
        for j in range(ny // 2):
            for k in range(nz // 2):
                pos = (i * 2 + 1, j * 2 + 1, k * 2 + 1)
                if pos[0] < nx and pos[1] < ny and pos[2] < nz:
                    pattern8.append(pos)
    pattern8.sort()
    positions.extend(pattern8)

    return positions, corner_inds_limit


def _get_cube_spatial_bounds(
    grid_pos: tuple[int, int, int],
    base_shape: list[int],
    overlap_size: int,
    final_shape: list[int],
    periodicity: list[bool] = [False, False, False]
) -> tuple[slice, slice, slice]:
    """
    Computes spatial bounds for a cube at grid position (i, j, k).

    Args:
        grid_pos: (i, j, k) grid position
        base_shape: [channels, dx, dy, dz] - base cube shape
        overlap_size: overlap between cubes
        final_shape: [channels, final_dx, final_dy, final_dz] - final volume shape

    Returns:
        Tuple of 3 slice objects for extracting/placing cubes
    """
    i, j, k = grid_pos
    base_size = base_shape[1:]  # [dx, dy, dz]
    final_size = final_shape[1:]  # [final_dx, final_dy, final_dz]
    overlap_half = overlap_size // 2

    # Compute spatial start positions (with overlap)
    start_x = i * base_size[0] - overlap_half
    start_y = j * base_size[1] - overlap_half
    start_z = k * base_size[2] - overlap_half

    # Compute extended size
    extended_size = [s + overlap_size for s in base_size]

    # Compute end positions
    end_x = start_x + extended_size[0]
    end_y = start_y + extended_size[1]
    end_z = start_z + extended_size[2]

    # Clamp to volume boundaries
    if not periodicity[0]:
        start_x = max(0, start_x)
        end_x = min(final_size[0], end_x)
    else:
        start_x = start_x % final_size[0]
        end_x = end_x % final_size[0]
    if not periodicity[1]:
        start_y = max(0, start_y)
        end_y = min(final_size[1], end_y)
    else:
        start_y = start_y % final_size[1]
        end_y = end_y % final_size[1]
    if not periodicity[2]:
        start_z = max(0, start_z)
        end_z = min(final_size[2], end_z)
    else:
        start_z = start_z % final_size[2]
        end_z = end_z % final_size[2]

    return (slice(start_x, end_x), slice(start_y, end_y), slice(start_z, end_z))


def _build_inpaint_mask(
    grid_pos: tuple[int, int, int],
    generated_positions: set[tuple[int, int, int]],
    base_shape: list[int],
    overlap_size: int,
    final_shape: list[int],
    periodicity: list[bool] = [False, False, False]
) -> torch.Tensor:
    """
    Creates mask for inpainting at current position.
    Mask is 1 where data exists from previously generated cubes.

    Args:
        grid_pos: (i, j, k) current grid position
        generated_positions: set of previously generated (i, j, k) positions
        base_shape: [channels, dx, dy, dz] - base cube shape
        overlap_size: overlap between cubes
        final_shape: [channels, final_dx, final_dy, final_dz] - final volume shape
        periodicity: [periodic_x, periodic_y, periodic_z] - whether each dimension wraps around

    Returns:
        Mask tensor of shape [channels, extended_size, extended_size, extended_size]
        where 1 indicates known data, 0 indicates unknown
    """
    # Get spatial bounds for current cube
    current_bounds = _get_cube_spatial_bounds(grid_pos, base_shape, overlap_size, final_shape, periodicity)
    sx, sy, sz = current_bounds

    # Extended size is base_size + overlap_size (not computed from bounds which may wrap)
    base_size = base_shape[1:]
    extended_size = [s + overlap_size for s in base_size]

    # Create a temporary volume to mark all previously generated regions
    # This approach handles periodicity correctly by using periodic_setitem
    # Create on CPU - will be moved to device by caller if needed
    temp_volume = torch.zeros(final_shape)

    # Mark all previously generated regions as 1 in temp_volume
    for prev_pos in generated_positions:
        prev_bounds = _get_cube_spatial_bounds(prev_pos, base_shape, overlap_size, final_shape, periodicity)
        psx, psy, psz = prev_bounds
        
        # Create a cube of ones with the same shape as the previous cube
        # Extended size is base_size + overlap_size (same for all cubes)
        ones_cube = torch.ones((base_shape[0], extended_size[0], extended_size[1], extended_size[2]))
        
        # Mark this region in temp_volume using periodic_setitem
        periodic_setitem(temp_volume, ones_cube, slice(None), psx, psy, psz)

    # Extract the mask for current cube from temp_volume using periodic_getitem
    mask = periodic_getitem(temp_volume, slice(None), sx, sy, sz)
    
    # Clamp mask to [0, 1] in case of overlaps (should already be 0 or 1, but just in case)
    mask = torch.clamp(mask, 0, 1)

    return mask


def _extract_wrapped_index(
    tensor: torch.Tensor,
    spatial_bounds: tuple[slice, slice, slice],
) -> torch.Tensor:
    """
    Extracts a slice from tensor with wrapping support for periodic dimensions.
    Generalization of: subvolume = cube[:, sx, sy, sz]

    Args:
        tensor: Input tensor of shape [channels, dx, dy, dz]
        spatial_bounds: Tuple of 3 slice objects (sx, sy, sz)
        final_shape: [channels, final_dx, final_dy, final_dz] - shape of the full volume
        periodicity: [periodic_x, periodic_y, periodic_z] - whether each dimension wraps around

    Returns:
        Extracted slice tensor with wrapping applied where needed
    """
    # TODO: Use torchutils.periodic_getitem
    sx, sy, sz = spatial_bounds
    return periodic_getitem(tensor, slice(None), sx, sy, sz)


def _combine_cube_into_volume(
    volume: torch.Tensor,
    cube: torch.Tensor,
    spatial_bounds: tuple[slice, slice, slice],
    blend_mode: str = 'latest'
) -> torch.Tensor:
    """
    Places generated cube into final volume.

    Args:
        volume: Final volume tensor to update
        cube: Generated cube to place
        spatial_bounds: Tuple of 3 slice objects for placement
        blend_mode: 'latest' to overwrite, 'cosine' to blend

    Returns:
        Updated volume tensor
    """
    sx, sy, sz = spatial_bounds

    if blend_mode == 'latest':
        # Simply overwrite
        periodic_setitem(volume, cube, slice(None), sx, sy, sz)
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")

    return volume


def sample_grid_volume(
    flow_module,
    grid_map: list[int],
    base_shape: list[int],
    overlap_size: int,
    y: None | dict[str, torch.Tensor] | np.ndarray = None,
    guidance: float = 1.0,
    nsteps: int = 30,
    blend_mode: Literal['latest', 'cosine'] = 'latest',
    periodicity: list[bool] = [False, False, False],
    inpaint_method: Literal['inpaint', 'inpaint_dps', 'inpaint_lanpaint'] = 'inpaint_dps',
    inpaint_kwargs: dict | None = None,
) -> Float[Tensor, "batch *final_shape"]:
    """
    Generate large volumes by tiling smaller cubes in a grid pattern using inpainting.

    Args:
        flow_module: SIModule instance for generation
        grid_map: [nx, ny, nz] - number of grid steps in each direction
        base_shape: [channels, dx, dy, dz] - base cube shape (e.g., [4, 32, 32, 32])
        overlap_size: int - overlap between cubes (e.g., 16)
        y: Optional condition tensor
        guidance: Guidance scale for conditional generation
        nsteps: Number of integration steps
        blend_mode: How to handle overlaps ('latest' or 'cosine')
        periodicity: [periodic_x, periodic_y, periodic_z] - whether each dimension wraps
        inpaint_method: Which inpainting algorithm to use:
            - 'inpaint': Original replacement-based
            - 'inpaint_dps': Diffusion Posterior Sampling (default)
            - 'inpaint_lanpaint': Langevin dynamics (experimental)
        inpaint_kwargs: Additional kwargs passed to the inpainting function

    Returns:
        Generated volume tensor of shape [1, channels, final_dx, final_dy, final_dz]
    """
    if isinstance(y, dict) or y is None:
        total_grid_map = np.prod(grid_map)
        y = np.array([y for _ in range(total_grid_map)]).reshape(grid_map)
    # Compute final volume shape
    final_shape = [
        base_shape[0],
        base_shape[1] * grid_map[0],
        base_shape[2] * grid_map[1],
        base_shape[3] * grid_map[2]
    ]

    # Check whether true periodic dimensions are matched with even grid maps
    for i in range(3):
        if periodicity[i] and grid_map[i] % 2 != 0:
            raise ValueError(f"Grid map for dimension {i} is not even, but periodicity is True")

    # Generate big noise cube of final shape
    device = flow_module.device
    noise_cube = torch.randn(1, *final_shape).to(device)

    # Initialize empty volume tensor
    volume = torch.zeros(1, *final_shape, device=device)

    # Get generation order based on parity patterns
    generation_order, corner_inds_limit = _get_grid_generation_order(grid_map)

    # Track generated positions
    generated_positions = set()

    # Generate cubes in order
    for grid_ind, grid_pos in enumerate(generation_order):
        # Compute spatial bounds for this cube
        spatial_bounds = _get_cube_spatial_bounds(grid_pos, base_shape, overlap_size, final_shape, periodicity)
        sx, sy, sz = spatial_bounds

        # Extract noise slice from big noise cube
        noise_slice = _extract_wrapped_index(noise_cube[0], spatial_bounds)  # Remove batch dim for extraction
        noise_slice = noise_slice.unsqueeze(0)  # Add batch dim back

        # Get extended cube shape
        extended_shape = list(noise_slice.shape[1:])
        cube_shape = [base_shape[0]] + extended_shape

        # Check if this is a "corner" (first pattern: all even coordinates)
        is_corner = grid_ind < corner_inds_limit

        if is_corner:
            # Use independent sampling for corners
            generated_cube = flow_module.sample(
                nsamples=1,
                shape=extended_shape,
                y=y[grid_pos[0], grid_pos[1], grid_pos[2]],
                guidance=guidance,
                nsteps=nsteps,
                is_latent_shape=True,
                orig_noise=noise_slice,
                return_latents=True,
            )
        else:
            # Use inpainting for edges/faces/centers
            # Build mask from previously generated cubes
            # continue  # TODO: Remove, obviously, this is a test

            mask = _build_inpaint_mask(
                grid_pos, generated_positions, base_shape, overlap_size, final_shape, periodicity
            )
            mask = mask.to(device)

            # Extract known regions from volume into x_orig
            x_orig = periodic_getitem(volume[0], slice(None), sx, sy, sz)

            # Use inpainting
            inpaint_fn = getattr(flow_module, inpaint_method)
            generated_cube = inpaint_fn(
                x_orig=x_orig,
                mask=mask,
                nsamples=1,
                y=y[grid_pos[0], grid_pos[1], grid_pos[2]],
                guidance=guidance,
                nsteps=nsteps,
                **(inpaint_kwargs or {}),
            )

        # inpaint returns [batch, *shape], remove batch dim if needed
        # if generated_cube.dim() == len(cube_shape) + 1 and generated_cube.shape[0] == 1:
            # generated_cube = generated_cube[0]
        generated_cube = generated_cube[0]
        # Combine generated cube into volume
        volume = _combine_cube_into_volume(volume[0], generated_cube, spatial_bounds, blend_mode)
        volume = volume.unsqueeze(0)  # Add batch dim back

        # Add position to generated_positions set
        generated_positions.add(grid_pos)

    return volume

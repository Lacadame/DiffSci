import matplotlib.pyplot as plt


def plot_volume_slices(x_cpu, n_slices=5, figsize=(15, 12), title_prefix="S", axis_labels=("D0", "D1", "D2")):
    """
    Plot evenly spaced slices from a 3D volume.

    Args:
        x_cpu (np.ndarray or torch.Tensor): Input 3D array of shape [H, W, D].
        n_slices (int): Number of slices to plot per axis.
        figsize (tuple): Figure size for matplotlib.
        title_prefix (str): Prefix for slice titles.
        axis_labels (tuple): Labels for each axis.
    """
    import numpy as np
    if hasattr(x_cpu, "detach"):  # torch.Tensor
        x_cpu = x_cpu.cpu().detach()
        if hasattr(x_cpu, "numpy"):
            x_cpu = x_cpu.numpy()
    if x_cpu.ndim != 3:
        raise ValueError(f"x_cpu must be 3D (got shape {x_cpu.shape})")

    fig, axes = plt.subplots(3, n_slices, figsize=figsize)

    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])

    # D0 slices (x direction, varying along axis 0)
    
    slide_idx = np.linspace(0, x_cpu.shape[0] - 1, n_slices).astype(int)
    for j, slice_idx in enumerate(slide_idx):
        ax = axes[0, j]
        ax.imshow(x_cpu[slice_idx, :, :], cmap="gray")
        ax.axis("off")
        ax.set_title(f"{title_prefix}{slice_idx}, {axis_labels[0]}")

    # D1 slices (y direction, varying along axis 1)
    slide_idx = np.linspace(0, x_cpu.shape[1] - 1, n_slices).astype(int)
    for j, slice_idx in enumerate(slide_idx):
        ax = axes[1, j]
        ax.imshow(x_cpu[:, slice_idx, :], cmap="gray")
        ax.axis("off")
        ax.set_title(f"{title_prefix}{slice_idx}, {axis_labels[1]}")

    # D2 slices (z direction, varying along axis 2)
    slide_idx = np.linspace(0, x_cpu.shape[2] - 1, n_slices).astype(int)
    for j, slice_idx in enumerate(slide_idx):
        ax = axes[2, j]
        ax.imshow(x_cpu[:, :, slice_idx], cmap="gray")
        ax.axis("off")
        ax.set_title(f"{title_prefix}{slice_idx}, {axis_labels[2]}")

    plt.tight_layout()
    plt.show()


def plot_latent_volume_slices(
    x_latent,
    n_slices=5,
    figsize=(15, 12),
    axis='x',
    colorbar=False,
):
    """
    Plot evenly spaced slices from a 3D latent volume.

    Args:
        x_latent (np.ndarray or torch.Tensor): Inputavg_calculated_porosity 4D array of shape [C, H, W, D].
        n_slices (int or list): Number of slices to plot along the chosen axis, or list of specific slice indices.
                                If int=1, picks the middle slice. If list, plots those specific slices.
        figsize (tuple): Figure size for matplotlib.
        axis (str): Which axis to slice along ('x' for dim=1, 'y' for dim=2, 'z' for dim=3).
        colorbar (bool): Whether to add a colorbar after the figures.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if hasattr(x_latent, "detach"):  # torch.Tensor
        x_latent = x_latent.cpu().detach()
        if hasattr(x_latent, "numpy"):
            x_latent = x_latent.numpy()
    if x_latent.ndim != 4:
        raise ValueError(f"x_latent must be 4D (got shape {x_latent.shape})")

    # axis_map: user axis ('x','y','z') -> (dim in np, axis label)
    axis_map = {
        'x': (1, 'x'),
        'y': (2, 'y'),
        'z': (3, 'z'),
    }
    if axis not in axis_map:
        raise ValueError(f"axis must be one of {list(axis_map.keys())}")
    axis_dim, axis_label = axis_map[axis]
    C = x_latent.shape[0]
    axis_len = x_latent.shape[axis_dim]

    # Compute the slice indices
    if isinstance(n_slices, list):
        slice_indices = n_slices
    else:
        assert (n_slices >= 1)
        if n_slices == 1:
            slice_indices = [axis_len // 2]
        else:
            slice_indices = [
                int(round(j * (axis_len - 1) / (n_slices - 1)))
                for j in range(n_slices - 1)
            ]
            slice_indices.append(axis_len - 1)

    num_slices = len(slice_indices)
    fig, axes = plt.subplots(C, num_slices, figsize=figsize, squeeze=False)
    vmin = x_latent.min()
    vmax = x_latent.max()
    images = []

    for c in range(C):
        for j, slice_idx in enumerate(slice_indices):
            ax = axes[c, j]
            if axis_dim == 1:  # slicing along x
                img = x_latent[c, slice_idx, :, :]
            elif axis_dim == 2:  # slicing along y
                img = x_latent[c, :, slice_idx, :]
            elif axis_dim == 3:  # slicing along z
                img = x_latent[c, :, :, slice_idx]
            im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            if c == 0 and j == 0:
                first_im = im  # Grab a reference for colorbar.
            images.append(im)
            ax.axis("off")
            ax.set_title(f"ch{c}, {axis_label}={slice_idx}")
    plt.tight_layout()
    if colorbar:
        # Place colorbar on the side of the entire figure
        fig.colorbar(first_im, ax=axes.ravel().tolist(), shrink=0.75, aspect=35, pad=0.02)
    plt.show()


def plot_latent_volume_slice_surfaces(
    x_latent,
    n_slices=5,
    figsize=(15, 12),
    axis='x',
    colorbar=False,
    cmap='viridis',
    elevation=30,
    azimuth=45,
    alpha=0.9,
    legends=None,
    title=None,
):
    """
    Plot evenly spaced slices from a 3D latent volume as 3D surface plots.
    Each slice is displayed as a surface where the height represents pixel values.

    Args:
        x_latent (np.ndarray, torch.Tensor, or list): Input 4D array of shape [C, H, W, D],
            or a list of such arrays for comparison. If a list, all arrays must have the same shape.
        n_slices (int or list): Number of slices to plot along the chosen axis, or list of specific slice indices.
                                If int=1, picks the middle slice. If list, plots those specific slices.
        figsize (tuple): Figure size for matplotlib.
        axis (str): Which axis to slice along ('x' for dim=1, 'y' for dim=2, 'z' for dim=3).
        colorbar (bool): Whether to add a colorbar after the figures.
        cmap (str or list): Colormap to use for the surface(s). If x_latent is a list and cmap is a string,
            uses different colormaps for each. If cmap is a list, uses those colormaps.
        elevation (float): Elevation angle for 3D view.
        azimuth (float): Azimuth angle for 3D view.
        alpha (float or list): Transparency for surfaces. Single value or list per latent.
        legends (list of str, optional): Legend labels for each latent space. Defaults to 'S1', 'S2', etc.
        title (str, optional): Overall figure title.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Handle single or multiple latent spaces
    if isinstance(x_latent, list):
        latent_list = x_latent
        multi_latent = True
    else:
        latent_list = [x_latent]
        multi_latent = False

    # Convert all to numpy
    converted_latents = []
    for lat in latent_list:
        if hasattr(lat, "detach"):  # torch.Tensor
            lat = lat.cpu().detach()
            if hasattr(lat, "numpy"):
                lat = lat.numpy()
        if lat.ndim != 4:
            raise ValueError(f"x_latent must be 4D (got shape {lat.shape})")
        converted_latents.append(lat)
    latent_list = converted_latents

    num_latents = len(latent_list)

    # Set up legends
    if legends is None:
        legends = [f'S{i+1}' for i in range(num_latents)]
    elif len(legends) != num_latents:
        raise ValueError(f"legends must have {num_latents} entries, got {len(legends)}")

    # Set up colormaps
    default_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'RdYlBu', 'Spectral']
    if isinstance(cmap, str):
        if multi_latent:
            cmaps = [default_cmaps[i % len(default_cmaps)] for i in range(num_latents)]
        else:
            cmaps = [cmap]
    else:
        cmaps = cmap

    # Set up alphas
    if isinstance(alpha, (int, float)):
        alphas = [alpha] * num_latents
    else:
        alphas = alpha

    # axis_map: user axis ('x','y','z') -> (dim in np, axis label)
    axis_map = {
        'x': (1, 'x'),
        'y': (2, 'y'),
        'z': (3, 'z'),
    }
    if axis not in axis_map:
        raise ValueError(f"axis must be one of {list(axis_map.keys())}")
    axis_dim, axis_label = axis_map[axis]

    # Use first latent for shape info (all assumed same shape)
    C = latent_list[0].shape[0]
    axis_len = latent_list[0].shape[axis_dim]

    # Compute the slice indices
    if isinstance(n_slices, list):
        slice_indices = n_slices
    else:
        assert (n_slices >= 1)
        if n_slices == 1:
            slice_indices = [axis_len // 2]
        else:
            slice_indices = [
                int(round(j * (axis_len - 1) / (n_slices - 1)))
                for j in range(n_slices - 1)
            ]
            slice_indices.append(axis_len - 1)

    num_slices = len(slice_indices)
    fig = plt.figure(figsize=figsize)

    # Compute global vmin/vmax across all latents
    vmin = min(lat.min() for lat in latent_list)
    vmax = max(lat.max() for lat in latent_list)
    surfaces = []

    for c in range(C):
        for j, slice_idx in enumerate(slice_indices):
            ax = fig.add_subplot(C, num_slices, c * num_slices + j + 1, projection='3d')

            # Plot each latent space as a surface
            for lat_idx, x_lat in enumerate(latent_list):
                # Extract the 2D slice
                if axis_dim == 1:  # slicing along x
                    img = x_lat[c, slice_idx, :, :]
                elif axis_dim == 2:  # slicing along y
                    img = x_lat[c, :, slice_idx, :]
                elif axis_dim == 3:  # slicing along z
                    img = x_lat[c, :, :, slice_idx]

                # Create meshgrid for surface plot
                height, width = img.shape
                X, Y = np.meshgrid(np.arange(width), np.arange(height))
                Z = img

                # Plot surface
                surf = ax.plot_surface(X, Y, Z, cmap=cmaps[lat_idx], vmin=vmin, vmax=vmax,
                                       linewidth=0, antialiased=True, alpha=alphas[lat_idx],
                                       label=legends[lat_idx])
                surfaces.append(surf)

            # Set view angle
            ax.view_init(elev=elevation, azim=azimuth)

            # Labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Value')
            ax.set_title(f"ch{c}, {axis_label}={slice_idx}")

            # Set z-limits for consistent scaling
            ax.set_zlim(vmin, vmax)

    # Add legend if multiple latents
    if multi_latent:
        # Create proxy artists for legend since 3D surfaces don't support labels directly
        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=plt.cm.get_cmap(cmaps[i])(0.5), label=legends[i])
                          for i in range(num_latents)]
        fig.legend(handles=legend_patches, loc='upper right')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if colorbar:
        # Add a colorbar
        fig.colorbar(surfaces[0], ax=fig.get_axes(), shrink=0.5, aspect=10, pad=0.05)

    plt.show()

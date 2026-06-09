"""
Dataset for sampling random subvolumes or subslices from 3D volume data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Union, Callable
from pathlib import Path

from .data_symmetries import SquareSymmetry, CubeSymmetry


class VolumeSubvolumeDataset(Dataset):
    """
    Dataset that samples random subvolumes from 3D volume data.

    Args:
        volumes: Either a single numpy array, a list of numpy arrays, or a list of paths to .raw files
        dataset_size: Arbitrary dataset size (defines __len__)
        subvolume_size: Size of subvolumes to extract. Can be int (cubic) or list[int] for [D, H, W]
        cube_symmetry: Optional CubeSymmetry instance for applying random symmetries
        dtype: Data type of the raw files if loading from paths (default: np.uint8)
        volume_shapes: Required if loading from paths - list of tuples with (D, H, W) for each volume
        extractor: Optional callable that extracts features from the subvolume tensor
        return_as_dict: If True, always return {'x': tensor, 'y': condition_dict}
    """

    def __init__(
        self,
        volumes: Union[np.ndarray, List[np.ndarray], List[str], List[Path]],
        dataset_size: int,
        subvolume_size: Union[int, List[int]] = 256,
        cube_symmetry=None,
        dtype: np.dtype = np.uint8,
        volume_shapes: List[tuple] = None,
        extractor: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
        return_as_dict: bool = False
    ):
        self.dataset_size = dataset_size
        self.extractor = extractor
        self.return_as_dict = return_as_dict
        # Handle subvolume size
        if isinstance(subvolume_size, int):
            self.subvolume_size = [subvolume_size, subvolume_size, subvolume_size]
        else:
            assert len(subvolume_size) == 3, "subvolume_size must be int or list of 3 ints"
            self.subvolume_size = subvolume_size

        # Load or store volumes
        if isinstance(volumes, np.ndarray):
            # Single volume
            self.volumes = [volumes]
        elif isinstance(volumes, list):
            if len(volumes) == 0:
                raise ValueError("volumes list cannot be empty")

            # Check if list of paths or numpy arrays
            if isinstance(volumes[0], (str, Path)):
                # Load from paths
                if volume_shapes is None:
                    raise ValueError("volume_shapes must be provided when loading from paths")
                if len(volume_shapes) != len(volumes):
                    raise ValueError("volume_shapes must have same length as volumes paths")

                self.volumes = []
                for path, shape in zip(volumes, volume_shapes):
                    volume = np.fromfile(path, dtype=dtype).reshape(shape)
                    self.volumes.append(volume)
            else:
                # Already numpy arrays
                self.volumes = volumes
        else:
            raise ValueError("volumes must be numpy array, list of arrays, or list of paths")

        # Validate volume sizes
        for i, volume in enumerate(self.volumes):
            if any(s < sv for s, sv in zip(volume.shape, self.subvolume_size)):
                raise ValueError(
                    f"Volume {i} with shape {volume.shape} is smaller than "
                    f"subvolume_size {self.subvolume_size}"
                )

        # Store cube symmetry
        self.cube_symmetry = cube_symmetry

        # Random number generator for reproducibility if needed
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Sample a random subvolume from a random volume.

        Returns:
            torch.Tensor: Float tensor of shape [1, D, H, W]
        """
        # 1. Choose one of the volumes randomly
        volume_idx = self.rng.integers(0, len(self.volumes))
        volume = self.volumes[volume_idx]

        # 2. Choose a random subvolume
        max_d = volume.shape[0] - self.subvolume_size[0]
        max_h = volume.shape[1] - self.subvolume_size[1]
        max_w = volume.shape[2] - self.subvolume_size[2]

        start_d = self.rng.integers(0, max_d + 1) if max_d > 0 else 0
        start_h = self.rng.integers(0, max_h + 1) if max_h > 0 else 0
        start_w = self.rng.integers(0, max_w + 1) if max_w > 0 else 0

        subvolume = volume[
            start_d:start_d + self.subvolume_size[0],
            start_h:start_h + self.subvolume_size[1],
            start_w:start_w + self.subvolume_size[2]
        ]

        # 3. Convert to tensor
        subvolume_tensor = torch.from_numpy(subvolume.copy())

        # 4. Apply random symmetry if provided
        if self.cube_symmetry is not None:
            subvolume_tensor = self.cube_symmetry.apply_random_symmetry(subvolume_tensor)

        # 5. Add channel dimension (unsqueeze)
        subvolume_tensor = subvolume_tensor.unsqueeze(0)

        # 6. Convert to float
        subvolume_tensor = subvolume_tensor.float()

        # 7. Extract features if provided
        if self.extractor is not None:
            condition_dict = self.extractor(subvolume_tensor)
        else:
            condition_dict = {}

        # 8. Return as dict if requested
        if self.return_as_dict:
            return {'x': subvolume_tensor, 'y': condition_dict}
        else:
            if len(condition_dict) > 0:
                return subvolume_tensor, condition_dict
            else:
                return subvolume_tensor

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)


class ConditionalVolumeSubvolumeDataset(VolumeSubvolumeDataset):
    """
    Extended dataset that also returns conditional information.

    Returns both the subvolume and a dictionary with conditional information.
    """

    def __init__(self, *args, compute_porosity: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_porosity = compute_porosity

    def __getitem__(self, idx):
        """
        Sample a random subvolume with conditional information.

        Returns:
            dict: {'x': subvolume_tensor, 'y': condition_dict}
                - subvolume_tensor: Float tensor of shape [1, D, H, W]
                - condition_dict: Dictionary with conditional information
        """
        subvolume = super().__getitem__(idx)

        condition_dict = {}

        if self.compute_porosity:
            # Compute porosity (assuming 0 = pore, 1 = solid)
            porosity = 1.0 - subvolume.mean().item()
            condition_dict['porosity'] = torch.tensor([porosity], dtype=torch.float32)

        return {'x': subvolume, 'y': condition_dict}


# Aliases for backward compatibility
StoneSubvolumeDataset = VolumeSubvolumeDataset
ConditionalStoneSubvolumeDataset = ConditionalVolumeSubvolumeDataset


class VolumeSubsliceDataset(Dataset):
    """
    Dataset that samples random 2D subslices from 3D volume data.

    The extraction process:
    1. Select a random volume from the dataset
    2. Select a random slice along the depth axis
    3. Extract a random subslice of size (H, W) from that slice
    4. Optionally apply square symmetry (D4 group: rotations + reflections)

    Args:
        volumes: Either a single numpy array, a list of numpy arrays, or a list of paths to .raw files
        dataset_size: Arbitrary dataset size (defines __len__)
        subslice_size: Size of subslices to extract. Can be int (square) or list[int] for [H, W]
        square_symmetry: Optional SquareSymmetry instance for applying random symmetries
        dtype: Data type of the raw files if loading from paths (default: np.uint8)
        volume_shapes: Required if loading from paths - list of tuples with (D, H, W) for each volume
        slice_axis: Axis along which to take slices (0, 1, or 2). Default is 0 (depth).
        extractor: Optional callable that extracts features from the subslice tensor
        return_as_dict: If True, always return {'x': tensor, 'y': condition_dict}
    """

    def __init__(
        self,
        volumes: Union[np.ndarray, List[np.ndarray], List[str], List[Path]],
        dataset_size: int,
        subslice_size: Union[int, List[int]] = 256,
        square_symmetry=None,
        dtype: np.dtype = np.uint8,
        volume_shapes: List[tuple] = None,
        slice_axis: int = 0,
        extractor: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
        return_as_dict: bool = False
    ):
        self.dataset_size = dataset_size
        self.extractor = extractor
        self.return_as_dict = return_as_dict
        self.slice_axis = slice_axis

        # Handle subslice size
        if isinstance(subslice_size, int):
            self.subslice_size = [subslice_size, subslice_size]
        else:
            assert len(subslice_size) == 2, "subslice_size must be int or list of 2 ints"
            self.subslice_size = subslice_size

        # Load or store volumes
        if isinstance(volumes, np.ndarray):
            # Single volume
            self.volumes = [volumes]
        elif isinstance(volumes, list):
            if len(volumes) == 0:
                raise ValueError("volumes list cannot be empty")

            # Check if list of paths or numpy arrays
            if isinstance(volumes[0], (str, Path)):
                # Load from paths
                if volume_shapes is None:
                    raise ValueError("volume_shapes must be provided when loading from paths")
                if len(volume_shapes) != len(volumes):
                    raise ValueError("volume_shapes must have same length as volumes paths")

                self.volumes = []
                for path, shape in zip(volumes, volume_shapes):
                    volume = np.fromfile(path, dtype=dtype).reshape(shape)
                    self.volumes.append(volume)
            else:
                # Already numpy arrays
                self.volumes = volumes
        else:
            raise ValueError("volumes must be numpy array, list of arrays, or list of paths")

        # Validate volume sizes (check the 2D slice dimensions)
        for i, volume in enumerate(self.volumes):
            slice_shape = self._get_slice_shape(volume.shape)
            if any(s < ss for s, ss in zip(slice_shape, self.subslice_size)):
                raise ValueError(
                    f"Volume {i} slice shape {slice_shape} (from volume shape {volume.shape}) "
                    f"is smaller than subslice_size {self.subslice_size}"
                )

        # Store square symmetry
        self.square_symmetry = square_symmetry

        # Random number generator for reproducibility if needed
        self.rng = np.random.default_rng()

    def _get_slice_shape(self, volume_shape: tuple) -> tuple:
        """Get the 2D shape of a slice along the slice axis."""
        if self.slice_axis == 0:
            return (volume_shape[1], volume_shape[2])
        elif self.slice_axis == 1:
            return (volume_shape[0], volume_shape[2])
        else:  # slice_axis == 2
            return (volume_shape[0], volume_shape[1])

    def _extract_slice(self, volume: np.ndarray, slice_idx: int) -> np.ndarray:
        """Extract a 2D slice from the volume along the slice axis."""
        if self.slice_axis == 0:
            return volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            return volume[:, slice_idx, :]
        else:  # slice_axis == 2
            return volume[:, :, slice_idx]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Sample a random 2D subslice from a random volume.

        Returns:
            torch.Tensor: Float tensor of shape [1, H, W]
        """
        # 1. Choose one of the volumes randomly
        volume_idx = self.rng.integers(0, len(self.volumes))
        volume = self.volumes[volume_idx]

        # 2. Choose a random slice along the slice axis
        num_slices = volume.shape[self.slice_axis]
        slice_idx = self.rng.integers(0, num_slices)
        slice_2d = self._extract_slice(volume, slice_idx)

        # 3. Choose a random subslice within the 2D slice
        max_h = slice_2d.shape[0] - self.subslice_size[0]
        max_w = slice_2d.shape[1] - self.subslice_size[1]

        start_h = self.rng.integers(0, max_h + 1) if max_h > 0 else 0
        start_w = self.rng.integers(0, max_w + 1) if max_w > 0 else 0

        subslice = slice_2d[
            start_h:start_h + self.subslice_size[0],
            start_w:start_w + self.subslice_size[1]
        ]

        # 4. Convert to tensor
        subslice_tensor = torch.from_numpy(subslice.copy())

        # 5. Apply random symmetry if provided
        if self.square_symmetry is not None:
            subslice_tensor = self.square_symmetry.apply_random_symmetry(subslice_tensor)

        # 6. Add channel dimension (unsqueeze)
        subslice_tensor = subslice_tensor.unsqueeze(0)

        # 7. Convert to float
        subslice_tensor = subslice_tensor.float()

        # 8. Extract features if provided
        if self.extractor is not None:
            condition_dict = self.extractor(subslice_tensor)
        else:
            condition_dict = {}

        # 9. Return as dict if requested
        if self.return_as_dict:
            return {'x': subslice_tensor, 'y': condition_dict}
        else:
            if len(condition_dict) > 0:
                return subslice_tensor, condition_dict
            else:
                return subslice_tensor

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)


class ConditionalVolumeSubsliceDataset(VolumeSubsliceDataset):
    """
    Extended dataset that also returns conditional information.

    Returns both the subslice and a dictionary with conditional information.
    """

    def __init__(self, *args, compute_porosity: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_porosity = compute_porosity

    def __getitem__(self, idx):
        """
        Sample a random subslice with conditional information.

        Returns:
            dict: {'x': subslice_tensor, 'y': condition_dict}
                - subslice_tensor: Float tensor of shape [1, H, W]
                - condition_dict: Dictionary with conditional information
        """
        subslice = super().__getitem__(idx)

        condition_dict = {}

        if self.compute_porosity:
            # Compute porosity (assuming 0 = pore, 1 = solid)
            porosity = 1.0 - subslice.mean().item()
            condition_dict['porosity'] = torch.tensor([porosity], dtype=torch.float32)

        return {'x': subslice, 'y': condition_dict}


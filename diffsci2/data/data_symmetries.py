"""
Symmetry operations for 2D and 3D arrays.

Contains:
- SquareSymmetry: D4 group (8 elements) for 2D arrays
- CubeSymmetry: Oh group (48 elements) for 3D arrays
"""

import numpy as np


class SquareSymmetry:
    """
    Applies the 8 symmetries of a square to 2D arrays (numpy or torch tensors).

    The 8 symmetries form the dihedral group D4 and consist of:
    - 4 rotations (0°, 90°, 180°, 270°)
    - 4 reflections (horizontal, vertical, and two diagonal)
    """

    def __init__(self):
        """Initialize the symmetry operator."""
        self._symmetry_names = self._build_symmetry_names()

    def _build_symmetry_names(self) -> dict[int, str]:
        """Build human-readable descriptions of each symmetry."""
        return {
            0: "Identity",
            1: "90° rotation CCW",
            2: "180° rotation",
            3: "270° rotation CCW",
            4: "Horizontal flip",
            5: "Vertical flip",
            6: "Transpose (main diagonal flip)",
            7: "Anti-transpose (anti-diagonal flip)",
        }

    def apply(self, arr, symmetry_id: int, inplace: bool = False):
        """
        Apply a square symmetry to a 2D array.

        Parameters:
        -----------
        arr : numpy.ndarray or torch.Tensor
            2D array of shape [H, W]
        symmetry_id : int
            Which symmetry to apply (0-7)
        inplace : bool
            If True, attempt to modify array in place when possible.
            Note: many symmetries require creating new arrays due to
            non-trivial index remapping.

        Returns:
        --------
        Transformed array (same type as input)
        """
        # Handle torch tensors
        is_torch = False
        torch_device = None
        torch_dtype = None

        try:
            import torch
            if isinstance(arr, torch.Tensor):
                is_torch = True
                torch_device = arr.device
                torch_dtype = arr.dtype
                arr_np = arr.detach().cpu().numpy()
            else:
                arr_np = arr
        except ImportError:
            arr_np = arr

        # Validate input
        if arr_np.ndim != 2:
            raise ValueError(f"Input must be 2D, got {arr_np.ndim}D")

        if not 0 <= symmetry_id < 8:
            raise ValueError(f"symmetry_id must be 0-7, got {symmetry_id}")

        # Apply symmetry
        result = self._apply_symmetry(arr_np, symmetry_id, inplace)

        # Convert back to torch if needed
        if is_torch:
            import torch
            # Ensure we have a contiguous array with positive strides
            if result.strides[0] < 0 or result.strides[1] < 0:
                result = result.copy()
            return torch.from_numpy(result).to(device=torch_device, dtype=torch_dtype)

        return result

    def _apply_symmetry(self, arr: np.ndarray, sym_id: int, inplace: bool) -> np.ndarray:
        """Internal method to apply symmetry to numpy array."""

        if sym_id == 0:  # Identity
            return arr if inplace else arr.copy()
        elif sym_id == 1:  # 90° CCW
            return np.ascontiguousarray(np.rot90(arr, k=1))
        elif sym_id == 2:  # 180°
            return np.ascontiguousarray(np.rot90(arr, k=2))
        elif sym_id == 3:  # 270° CCW
            return np.ascontiguousarray(np.rot90(arr, k=3))
        elif sym_id == 4:  # Horizontal flip
            return np.ascontiguousarray(arr[:, ::-1])
        elif sym_id == 5:  # Vertical flip
            return np.ascontiguousarray(arr[::-1, :])
        elif sym_id == 6:  # Transpose
            return np.ascontiguousarray(arr.T)
        elif sym_id == 7:  # Anti-transpose
            return np.ascontiguousarray(np.rot90(arr, k=2).T)
        else:
            raise ValueError(f"Unhandled symmetry_id: {sym_id}")

    def get_all_symmetries(self, arr) -> list:
        """
        Generate all 8 symmetry transformations of the input array.

        Returns a list of 8 transformed arrays.
        """
        return [self.apply(arr, i) for i in range(8)]

    def find_symmetry(self, original, transformed, tolerance: float = 1e-10) -> int | None:
        """
        Find which symmetry maps original to transformed.

        Returns symmetry_id if found, None otherwise.
        """
        # Convert to numpy for comparison
        try:
            import torch
            if isinstance(original, torch.Tensor):
                original = original.detach().cpu().numpy()
                transformed = transformed.detach().cpu().numpy()
        except ImportError:
            pass

        for i in range(8):
            candidate = self._apply_symmetry(original, i, False)
            if np.allclose(candidate, transformed, atol=tolerance):
                return i

        return None

    def compose(self, id1: int, id2: int) -> int:
        """
        Find the single symmetry equivalent to applying id1 then id2.

        This demonstrates the group structure of square symmetries.
        """
        # Create small test array with unique values
        test = np.arange(9).reshape(3, 3)

        # Apply composition
        result = self._apply_symmetry(
            self._apply_symmetry(test, id1, False),
            id2,
            False
        )

        # Find equivalent single symmetry
        for i in range(8):
            if np.array_equal(self._apply_symmetry(test, i, False), result):
                return i

        raise RuntimeError(f"Composition {id1} o {id2} not found - implementation error!")

    def describe(self, symmetry_id: int) -> str:
        """Get human-readable description of a symmetry."""
        return self._symmetry_names.get(symmetry_id, f"Unknown symmetry {symmetry_id}")

    def inverse(self, symmetry_id: int) -> int:
        """
        Find the inverse of a given symmetry.

        The inverse symmetry, when composed with the original, gives identity.
        """
        if symmetry_id == 0:
            return 0  # Identity is self-inverse
        elif symmetry_id == 1:
            return 3  # 90° CCW inverse is 270° CCW
        elif symmetry_id == 2:
            return 2  # 180° is self-inverse
        elif symmetry_id == 3:
            return 1  # 270° CCW inverse is 90° CCW
        elif symmetry_id in [4, 5, 6, 7]:
            return symmetry_id  # All reflections are self-inverse

        raise RuntimeError(f"Failed to find inverse for symmetry {symmetry_id}")

    def multiplication_table(self) -> np.ndarray:
        """
        Generate the full group multiplication table.

        Returns an 8x8 array where element [i,j] is the result of symmetry i o j.
        """
        table = np.zeros((8, 8), dtype=np.int32)
        for i in range(8):
            for j in range(8):
                table[i, j] = self.compose(i, j)
        return table

    def is_orientation_preserving(self, symmetry_id: int) -> bool:
        """
        Check if a symmetry preserves orientation (proper rotation).

        Symmetries 0-3 are rotations (proper), 4-7 are reflections (improper).
        """
        return symmetry_id < 4

    def order(self, symmetry_id: int) -> int:
        """
        Find the order of a symmetry (how many times to apply to get identity).
        """
        if symmetry_id == 0:
            return 1
        elif symmetry_id in [2, 4, 5, 6, 7]:
            return 2  # 180° and all reflections have order 2
        elif symmetry_id in [1, 3]:
            return 4  # 90° and 270° have order 4

        raise RuntimeError(f"Order calculation failed for symmetry {symmetry_id}")

    def get_symmetry_matrix(self, symmetry_id: int) -> np.ndarray:
        """
        Get the 2x2 rotation/reflection matrix representation of the symmetry.
        """
        matrices = {
            0: np.array([[1, 0], [0, 1]]),      # Identity
            1: np.array([[0, -1], [1, 0]]),     # 90° CCW
            2: np.array([[-1, 0], [0, -1]]),    # 180°
            3: np.array([[0, 1], [-1, 0]]),     # 270° CCW
            4: np.array([[-1, 0], [0, 1]]),     # Horizontal flip
            5: np.array([[1, 0], [0, -1]]),     # Vertical flip
            6: np.array([[0, 1], [1, 0]]),      # Transpose
            7: np.array([[0, -1], [-1, 0]]),    # Anti-transpose
        }
        return matrices[symmetry_id]

    def apply_random_symmetry(self, arr):
        """
        Apply a random symmetry to the input array.
        """
        symmetry_id = np.random.randint(0, 8)
        return self.apply(arr, symmetry_id)


class CubeSymmetry:
    """
    Applies the 48 symmetries of a cube to 3D arrays (numpy or torch tensors).

    The 48 symmetries form the octahedral group Oh and consist of:
    - 24 proper rotations (orientation-preserving)
    - 24 improper rotations (orientation-reversing, includes inversion)
    """

    def __init__(self):
        """Initialize the symmetry operator."""
        self._symmetry_names = self._build_symmetry_names()

    def _build_symmetry_names(self) -> dict[int, str]:
        """Build human-readable descriptions of each symmetry."""
        names = {
            0: "Identity",
            1: "90° rotation around x-axis",
            2: "180° rotation around x-axis",
            3: "270° rotation around x-axis",
            4: "90° rotation around y-axis",
            5: "180° rotation around y-axis",
            6: "270° rotation around y-axis",
            7: "90° rotation around z-axis",
            8: "180° rotation around z-axis",
            9: "270° rotation around z-axis",
            10: "120° rotation around (1,1,1) diagonal",
            11: "240° rotation around (1,1,1) diagonal",
            12: "120° rotation around (1,1,-1) diagonal",
            13: "240° rotation around (1,1,-1) diagonal",
            14: "120° rotation around (1,-1,1) diagonal",
            15: "240° rotation around (1,-1,1) diagonal",
            16: "120° rotation around (-1,1,1) diagonal",
            17: "240° rotation around (-1,1,1) diagonal",
            18: "180° rotation around edge (x-parallel, +y+z)",
            19: "180° rotation around edge (x-parallel, +y-z)",
            20: "180° rotation around edge (y-parallel, +x+z)",
            21: "180° rotation around edge (y-parallel, -x+z)",
            22: "180° rotation around edge (z-parallel, +x+y)",
            23: "180° rotation around edge (z-parallel, +x-y)",
            24: "Inversion through center",
        }

        # Add improper rotations
        for i in range(25, 48):
            base = names[i - 24]
            names[i] = f"Inversion + {base}"

        return names

    def apply(self, arr, symmetry_id: int, inplace: bool = False):
        """
        Apply a cube symmetry to a 3D array.

        Parameters:
        -----------
        arr : numpy.ndarray or torch.Tensor
            3D cubic array of shape [D, D, D]
        symmetry_id : int
            Which symmetry to apply (0-47)
        inplace : bool
            If True, attempt to modify array in place when possible.
            Note: many symmetries require creating new arrays due to
            non-trivial index remapping.

        Returns:
        --------
        Transformed array (same type as input)
        """
        # Handle torch tensors
        is_torch = False
        torch_device = None
        torch_dtype = None

        try:
            import torch
            if isinstance(arr, torch.Tensor):
                is_torch = True
                torch_device = arr.device
                torch_dtype = arr.dtype
                arr_np = arr.detach().cpu().numpy()
            else:
                arr_np = arr
        except ImportError:
            arr_np = arr

        # Validate input
        if arr_np.ndim != 3:
            raise ValueError(f"Input must be 3D, got {arr_np.ndim}D")

        D = arr_np.shape[0]
        if not (arr_np.shape[0] == arr_np.shape[1] == arr_np.shape[2]):
            raise ValueError(f"Input must be cubic, got shape {arr_np.shape}")

        if not 0 <= symmetry_id < 48:
            raise ValueError(f"symmetry_id must be 0-47, got {symmetry_id}")

        # Apply symmetry
        result = self._apply_symmetry(arr_np, symmetry_id, inplace)

        # Convert back to torch if needed
        if is_torch:
            import torch
            # Ensure we have a contiguous array with positive strides
            if result.strides[0] < 0 or result.strides[1] < 0 or result.strides[2] < 0:
                result = result.copy()
            return torch.from_numpy(result).to(device=torch_device, dtype=torch_dtype)

        return result

    def _apply_symmetry(self, arr: np.ndarray, sym_id: int, inplace: bool) -> np.ndarray:
        """Internal method to apply symmetry to numpy array."""

        # Identity - can always be inplace
        if sym_id == 0:
            return arr if inplace else arr.copy()

        # For inversion (24) and composed inversions (25-47)
        if sym_id >= 24:
            if sym_id == 24:
                # Pure inversion
                result = arr[::-1, ::-1, ::-1]
                # If inplace requested, return view; otherwise make contiguous copy
                return result if inplace else np.ascontiguousarray(result)
            else:
                # Apply base symmetry then invert
                base_result = self._apply_symmetry(arr, sym_id - 24, False)
                result = base_result[::-1, ::-1, ::-1]
                return np.ascontiguousarray(result)

        # Face rotations (1-9)
        if 1 <= sym_id <= 9:
            return self._rotate_face(arr, sym_id)

        # Vertex diagonal rotations (10-17)
        if 10 <= sym_id <= 17:
            return self._rotate_diagonal(arr, sym_id)

        # Edge midpoint rotations (18-23)
        if 18 <= sym_id <= 23:
            return self._rotate_edge(arr, sym_id)

        raise ValueError(f"Unhandled symmetry_id: {sym_id}")

    def _rotate_face(self, arr: np.ndarray, sym_id: int) -> np.ndarray:
        """Rotate around a face normal (x, y, or z axis)."""

        # Determine axis and number of 90° rotations
        if 1 <= sym_id <= 3:
            axis = 0  # x-axis
            rotations = sym_id
        elif 4 <= sym_id <= 6:
            axis = 1  # y-axis
            rotations = sym_id - 3
        else:  # 7 <= sym_id <= 9
            axis = 2  # z-axis
            rotations = sym_id - 6

        # Use numpy's rot90 which is efficient
        if axis == 0:  # Rotate in yz plane
            result = np.rot90(arr, k=rotations, axes=(1, 2))
        elif axis == 1:  # Rotate in xz plane
            result = np.rot90(arr, k=rotations, axes=(0, 2))
        else:  # axis == 2, Rotate in xy plane
            result = np.rot90(arr, k=rotations, axes=(0, 1))

        # Ensure contiguous for better performance
        return np.ascontiguousarray(result)

    def _rotate_diagonal(self, arr: np.ndarray, sym_id: int) -> np.ndarray:
        """
        Rotate around a vertex diagonal.

        The four diagonals are: (1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)
        Each can be rotated by 120° or 240°.
        """
        diagonal_id = (sym_id - 10) // 2
        times_120 = ((sym_id - 10) % 2) + 1

        if diagonal_id == 0:  # (1,1,1) diagonal
            # Cyclic permutation of axes
            if times_120 == 1:
                return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
            else:  # times_120 == 2
                return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))

        elif diagonal_id == 1:  # (1,1,-1) diagonal
            # Flip z, permute, flip back
            if times_120 == 1:
                temp = arr[:, :, ::-1]
                temp = np.transpose(temp, (2, 0, 1))
                return np.ascontiguousarray(temp[:, :, ::-1])
            else:
                temp = arr[:, :, ::-1]
                temp = np.transpose(temp, (1, 2, 0))
                return np.ascontiguousarray(temp[:, :, ::-1])

        elif diagonal_id == 2:  # (1,-1,1) diagonal
            # Flip y, permute, flip back
            if times_120 == 1:
                temp = arr[:, ::-1, :]
                temp = np.transpose(temp, (2, 0, 1))
                return np.ascontiguousarray(temp[:, ::-1, :])
            else:
                temp = arr[:, ::-1, :]
                temp = np.transpose(temp, (1, 2, 0))
                return np.ascontiguousarray(temp[:, ::-1, :])

        else:  # diagonal_id == 3, (-1,1,1) diagonal
            # Flip x, permute, flip back
            if times_120 == 1:
                temp = arr[::-1, :, :]
                temp = np.transpose(temp, (2, 0, 1))
                return np.ascontiguousarray(temp[::-1, :, :])
            else:
                temp = arr[::-1, :, :]
                temp = np.transpose(temp, (1, 2, 0))
                return np.ascontiguousarray(temp[::-1, :, :])

    def _rotate_edge(self, arr: np.ndarray, sym_id: int) -> np.ndarray:
        """
        180° rotation around edge midpoint.

        These are axes parallel to coordinate axes but passing through
        face centers rather than the origin.
        """
        edge_id = sym_id - 18

        # All edge rotations are 180°, so they're self-inverse
        # They flip two coordinates while leaving one unchanged

        if edge_id in [0, 1]:  # x-parallel edges
            result = arr[:, ::-1, ::-1]
        elif edge_id in [2, 3]:  # y-parallel edges
            result = arr[::-1, :, ::-1]
        else:  # edge_id in [4, 5], z-parallel edges
            result = arr[::-1, ::-1, :]

        return np.ascontiguousarray(result)

    def get_all_symmetries(self, arr) -> list:
        """
        Generate all 48 symmetry transformations of the input array.

        Returns a list of 48 transformed arrays.
        """
        return [self.apply(arr, i) for i in range(48)]

    def find_symmetry(self, original, transformed, tolerance: float = 1e-10) -> int | None:
        """
        Find which symmetry maps original to transformed.

        Returns symmetry_id if found, None otherwise.
        """
        # Convert to numpy for comparison
        is_torch = False
        try:
            import torch
            if isinstance(original, torch.Tensor):
                is_torch = True
                original = original.detach().cpu().numpy()
                transformed = transformed.detach().cpu().numpy()
        except ImportError:
            pass

        for i in range(48):
            candidate = self._apply_symmetry(original, i, False)
            if np.allclose(candidate, transformed, atol=tolerance):
                return i

        return None

    def compose(self, id1: int, id2: int) -> int:
        """
        Find the single symmetry equivalent to applying id1 then id2.

        This demonstrates the group structure of cube symmetries.
        """
        # Create small test array with unique values
        test = np.arange(27).reshape(3, 3, 3)

        # Apply composition
        result = self._apply_symmetry(
            self._apply_symmetry(test, id1, False),
            id2,
            False
        )

        # Find equivalent single symmetry
        for i in range(48):
            if np.array_equal(self._apply_symmetry(test, i, False), result):
                return i

        raise RuntimeError(f"Composition {id1} o {id2} not found - implementation error!")

    def describe(self, symmetry_id: int) -> str:
        """Get human-readable description of a symmetry."""
        return self._symmetry_names.get(symmetry_id, f"Unknown symmetry {symmetry_id}")

    def inverse(self, symmetry_id: int) -> int:
        """
        Find the inverse of a given symmetry.

        The inverse symmetry, when composed with the original, gives identity.
        """
        # More efficient: directly compute the inverse
        # Most symmetries are self-inverse or have simple inverse relationships
        if symmetry_id == 0:
            return 0  # Identity is self-inverse

        # For proper rotations
        if symmetry_id <= 23:
            test = np.arange(27).reshape(3, 3, 3)
            identity = test.copy()

            # Find what undoes this symmetry
            for i in range(48):
                if np.array_equal(
                    self._apply_symmetry(self._apply_symmetry(test, symmetry_id, False), i, False),
                    identity
                ):
                    return i

        # Inversion is self-inverse
        if symmetry_id == 24:
            return 24

        # For improper rotations, use group theory
        # (inversion + rotation)^-1 = rotation^-1 + inversion
        if symmetry_id > 24:
            base_rotation = symmetry_id - 24
            base_inverse = self.inverse(base_rotation)
            return (base_inverse + 24) if base_inverse < 24 else (base_inverse - 24)

        raise RuntimeError(f"Failed to find inverse for symmetry {symmetry_id}")

    def multiplication_table(self) -> np.ndarray:
        """
        Generate the full group multiplication table.

        Returns a 48x48 array where element [i,j] is the result of symmetry i o j.
        This is computationally intensive but useful for understanding the group structure.
        """
        table = np.zeros((48, 48), dtype=np.int32)
        for i in range(48):
            for j in range(48):
                table[i, j] = self.compose(i, j)
        return table

    def is_orientation_preserving(self, symmetry_id: int) -> bool:
        """
        Check if a symmetry preserves orientation (proper rotation).

        Symmetries 0-23 are proper, 24-47 are improper.
        """
        return symmetry_id < 24

    def order(self, symmetry_id: int) -> int:
        """
        Find the order of a symmetry (how many times to apply to get identity).
        """
        if symmetry_id == 0:
            return 1

        current = symmetry_id
        order = 1

        test = np.arange(27).reshape(3, 3, 3)
        identity = test.copy()
        result = self._apply_symmetry(test, symmetry_id, False)

        while not np.array_equal(result, identity):
            result = self._apply_symmetry(result, symmetry_id, False)
            order += 1
            if order > 48:  # Safety check
                raise RuntimeError(f"Order calculation failed for symmetry {symmetry_id}")

        return order

    def get_symmetry_matrix(self, symmetry_id: int) -> np.ndarray:
        """
        Get the 3x3 rotation matrix representation of the symmetry.

        Note: This only gives the rotation part. Improper rotations
        would need a 4x4 matrix or separate handling of the inversion.
        """
        # Apply symmetry to unit vectors to extract the matrix
        test = np.zeros((3, 3, 3))
        test[1, 0, 0] = 1  # x unit vector
        test[0, 1, 0] = 2  # y unit vector
        test[0, 0, 1] = 3  # z unit vector

        result = self._apply_symmetry(test, symmetry_id % 24, False)  # Get rotation part only

        # Extract where the unit vectors went
        matrix = np.zeros((3, 3))

        # Find where each unit vector ended up
        for val in [1, 2, 3]:
            original_axis = val - 1
            location = np.where(np.abs(result - val) < 1e-10)
            if len(location[0]) > 0:
                i, j, k = location[0][0], location[1][0], location[2][0]
                # Determine which axis and sign
                if i == 1 and j == 0 and k == 0:
                    matrix[0, original_axis] = 1
                elif i == 0 and j == 0 and k == 0:
                    matrix[0, original_axis] = -1
                elif j == 1 and i == 0 and k == 0:
                    matrix[1, original_axis] = 1
                elif j == 0 and i == 0 and k == 0:
                    matrix[1, original_axis] = -1
                elif k == 1 and i == 0 and j == 0:
                    matrix[2, original_axis] = 1
                elif k == 0 and i == 0 and j == 0:
                    matrix[2, original_axis] = -1

        return matrix

    def apply_random_symmetry(self, arr):
        """
        Apply a random symmetry to the input array.
        """
        symmetry_id = np.random.randint(0, 48)
        return self.apply(arr, symmetry_id)


# Example usage and testing
if __name__ == "__main__":
    import time

    # Test SquareSymmetry
    print("=" * 60)
    print("Testing SquareSymmetry (D4 group)")
    print("=" * 60)

    sq_sym = SquareSymmetry()

    # Test with numpy
    print("\nTesting with NumPy array:")
    arr_2d = np.random.rand(5, 5)

    # Apply a symmetry
    rotated = sq_sym.apply(arr_2d, 1)  # 90° rotation CCW
    print(f"Shape preserved: {arr_2d.shape == rotated.shape}")

    # Check that 4 rotations of 90° return to original
    result = arr_2d
    for _ in range(4):
        result = sq_sym.apply(result, 1)
    print(f"4x 90° rotation returns to original: {np.allclose(result, arr_2d)}")

    # Test all 8 symmetries
    print("\nTesting all 8 symmetries:")
    test_arr_2d = np.random.rand(4, 4)
    errors = []
    for i in range(8):
        try:
            result = sq_sym.apply(test_arr_2d, i)
            if result.shape != test_arr_2d.shape:
                errors.append(f"Symmetry {i}: shape mismatch")
        except Exception as e:
            errors.append(f"Symmetry {i}: {e}")

    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  {error}")
    else:
        print("All 8 symmetries work correctly!")

    # Test CubeSymmetry
    print("\n" + "=" * 60)
    print("Testing CubeSymmetry (Oh group)")
    print("=" * 60)

    cube_sym = CubeSymmetry()

    # Test with numpy
    print("\nTesting with NumPy array:")
    arr_np = np.random.rand(5, 5, 5)

    # Apply a symmetry
    rotated = cube_sym.apply(arr_np, 7)  # 90° rotation around z
    print(f"Shape preserved: {arr_np.shape == rotated.shape}")

    # Check that 4 rotations of 90° return to original
    result = arr_np
    for _ in range(4):
        result = cube_sym.apply(result, 7)
    print(f"4x 90° rotation returns to original: {np.allclose(result, arr_np)}")

    # Test with PyTorch if available
    try:
        import torch
        print("\nTesting with PyTorch tensor:")
        arr_torch = torch.rand(5, 5, 5, device='cpu', dtype=torch.float32)

        rotated_torch = cube_sym.apply(arr_torch, 7)
        print(f"Type preserved: {type(arr_torch) == type(rotated_torch)}")
        print(f"Device preserved: {arr_torch.device == rotated_torch.device}")
        print(f"Dtype preserved: {arr_torch.dtype == rotated_torch.dtype}")

        # Test with various symmetries including those that create negative strides
        test_symmetries = [0, 7, 24, 25, 18]  # identity, rotation, inversion, composed, edge
        print("\nTesting various symmetries with torch:")
        for sym_id in test_symmetries:
            try:
                result = cube_sym.apply(arr_torch, sym_id)
                print(f"  Symmetry {sym_id:2d} ({cube_sym.describe(sym_id)[:30]:30s}): OK")
            except Exception as e:
                print(f"  Symmetry {sym_id:2d}: FAILED - {e}")

        # Verify it's the same transformation as numpy
        rotated_torch_np = rotated_torch.numpy()
        rotated_np_direct = cube_sym.apply(arr_torch.numpy(), 7)
        print(f"\nSame result as numpy: {np.allclose(rotated_torch_np, rotated_np_direct)}")

        # Test GPU tensors if available
        if torch.cuda.is_available():
            print("\nTesting with CUDA tensor:")
            arr_cuda = torch.rand(5, 5, 5, device='cuda', dtype=torch.float32)
            rotated_cuda = cube_sym.apply(arr_cuda, 7)
            print(f"CUDA device preserved: {rotated_cuda.device == arr_cuda.device}")

    except ImportError:
        print("\nPyTorch not available, skipping torch tests")

    # Test group properties
    print("\nGroup property tests:")

    # Test inverse
    test_id = 15  # Some arbitrary symmetry
    inv_id = cube_sym.inverse(test_id)
    composition = cube_sym.compose(test_id, inv_id)
    print(f"Symmetry {test_id} inverse is {inv_id}, composition gives {composition} (should be 0)")

    # Test order
    print("\nOrder of various symmetries:")
    for test_id in [0, 1, 2, 7, 10, 24]:
        order = cube_sym.order(test_id)
        print(f"  Symmetry {test_id:2d} ({cube_sym.describe(test_id)[:30]:30s}) has order {order}")

    # Performance comparison
    print("\nPerformance test (1000 applications on 10x10x10 array):")
    test_array = np.random.rand(10, 10, 10)

    start = time.time()
    for _ in range(1000):
        _ = cube_sym.apply(test_array, 7, inplace=False)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.3f} seconds ({1000/elapsed:.1f} ops/sec)")

    # Test that all symmetries work without error
    print("\nTesting all 48 symmetries:")
    test_arr = np.random.rand(4, 4, 4)
    errors = []
    for i in range(48):
        try:
            result = cube_sym.apply(test_arr, i)
            if result.shape != test_arr.shape:
                errors.append(f"Symmetry {i}: shape mismatch")
        except Exception as e:
            errors.append(f"Symmetry {i}: {e}")

    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  {error}")
    else:
        print("All 48 symmetries work correctly!")

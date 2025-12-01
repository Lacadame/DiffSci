import torch
from diffsci.torchutils import periodic_getitem


def test_periodic_getitem_basic():
    """Test basic slicing within bounds."""
    a = torch.arange(5)  # [0, 1, 2, 3, 4]

    # Normal slice
    result = periodic_getitem(a, slice(1, 3))
    assert result.tolist() == [1, 2]

    # Full slice
    result = periodic_getitem(a, slice(0, 5))
    assert result.tolist() == [0, 1, 2, 3, 4]


def test_periodic_getitem_wrap_around():
    """Test wrap-around within one period."""
    a = torch.arange(5)  # [0, 1, 2, 3, 4]

    # Wrap from end to beginning
    result = periodic_getitem(a, slice(3, 2))
    assert result.tolist() == [3, 4, 0, 1]

    # Start from negative
    result = periodic_getitem(a, slice(-2, 2))
    assert result.tolist() == [3, 4, 0, 1]


def test_periodic_getitem_negative_start():
    """Test negative start indices."""
    a = torch.arange(5)  # [0, 1, 2, 3, 4]

    # -1 should be index 4
    result = periodic_getitem(a, slice(-1, 2))
    assert result.tolist() == [4, 0, 1]

    # -3 should be index 2
    result = periodic_getitem(a, slice(-3, 1))
    assert result.tolist() == [2, 3, 4, 0]


def test_periodic_getitem_larger_than_size():
    """Test slices larger than tensor size (multi-period)."""
    a = torch.arange(3)  # [0, 1, 2]

    # Slice from -2 to 7 (9 elements)
    result = periodic_getitem(a, slice(-2, 7))
    assert result.tolist() == [1, 2, 0, 1, 2, 0, 1, 2, 0]

    # Slice from 0 to 6 (2 full periods)
    result = periodic_getitem(a, slice(0, 6))
    assert result.tolist() == [0, 1, 2, 0, 1, 2]

    # Slice from 0 to 7 (2 full periods + 1)
    result = periodic_getitem(a, slice(0, 7))
    assert result.tolist() == [0, 1, 2, 0, 1, 2, 0]

    # Slice from 1 to 8 (start offset + 2 full periods + partial)
    result = periodic_getitem(a, slice(1, 8))
    assert result.tolist() == [1, 2, 0, 1, 2, 0, 1]


def test_periodic_getitem_very_large_slice():
    """Test with slice much larger than size."""
    a = torch.arange(3)  # [0, 1, 2]

    # 10 elements starting from 0
    result = periodic_getitem(a, slice(0, 10))
    assert result.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

    # Like the original error case: slice(-5, 37) on size 32
    b = torch.arange(32)
    result = periodic_getitem(b, slice(-5, 37))
    assert len(result) == 42
    # First 5 elements should be 27, 28, 29, 30, 31
    assert result[:5].tolist() == [27, 28, 29, 30, 31]
    # Next 32 elements should be 0..31
    assert result[5:37].tolist() == list(range(32))
    # Last 5 elements should be 0, 1, 2, 3, 4
    assert result[37:].tolist() == [0, 1, 2, 3, 4]


def test_periodic_getitem_multidimensional():
    """Test with multi-dimensional tensors."""
    # 2D tensor
    a = torch.arange(12).reshape(3, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    #  [8, 9, 10, 11]]

    # Large slice on first dim only
    result = periodic_getitem(a, slice(0, 6), slice(None))
    assert result.shape == (6, 4)
    assert result[0].tolist() == [0, 1, 2, 3]
    assert result[3].tolist() == [0, 1, 2, 3]  # Wrapped

    # Large slice on second dim only
    result = periodic_getitem(a, slice(None), slice(-1, 6))
    assert result.shape == (3, 7)
    assert result[0].tolist() == [3, 0, 1, 2, 3, 0, 1]


def test_periodic_getitem_3d():
    """Test with 3D tensor (like the chunk_decode use case)."""
    # Shape: (2, 4, 3)
    a = torch.arange(24).reshape(2, 4, 3)

    # Large slice on middle dimension
    result = periodic_getitem(a, slice(None), slice(-1, 6), slice(None))
    assert result.shape == (2, 7, 3)

    # Large slice on last dimension
    result = periodic_getitem(a, slice(None), slice(None), slice(0, 6))
    assert result.shape == (2, 4, 6)


def test_periodic_getitem_empty_slice():
    """Test empty slices."""
    a = torch.arange(5)

    result = periodic_getitem(a, slice(2, 2))
    assert result.shape == (0,)


def test_periodic_getitem_exact_size():
    """Test slice exactly equal to size."""
    a = torch.arange(5)

    # Starting from 0
    result = periodic_getitem(a, slice(0, 5))
    assert result.tolist() == [0, 1, 2, 3, 4]

    # Starting from 2 (wrap around)
    result = periodic_getitem(a, slice(2, 7))
    assert result.tolist() == [2, 3, 4, 0, 1]


if __name__ == '__main__':
    test_periodic_getitem_basic()
    print('test_periodic_getitem_basic passed')

    test_periodic_getitem_wrap_around()
    print('test_periodic_getitem_wrap_around passed')

    test_periodic_getitem_negative_start()
    print('test_periodic_getitem_negative_start passed')

    test_periodic_getitem_larger_than_size()
    print('test_periodic_getitem_larger_than_size passed')

    test_periodic_getitem_very_large_slice()
    print('test_periodic_getitem_very_large_slice passed')

    test_periodic_getitem_multidimensional()
    print('test_periodic_getitem_multidimensional passed')

    test_periodic_getitem_3d()
    print('test_periodic_getitem_3d passed')

    test_periodic_getitem_empty_slice()
    print('test_periodic_getitem_empty_slice passed')

    test_periodic_getitem_exact_size()
    print('test_periodic_getitem_exact_size passed')

    print('\nAll tests passed!')

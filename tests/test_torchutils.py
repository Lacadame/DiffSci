import torch
from diffsci.torchutils import periodic_getitem, periodic_getitem_extended


# ============================================================================
# Tests for periodic_getitem (original, unchanged behavior)
# ============================================================================

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


def test_periodic_getitem_empty_slice():
    """Test empty slices."""
    a = torch.arange(5)

    result = periodic_getitem(a, slice(2, 2))
    assert result.shape == (0,)


def test_periodic_getitem_rejects_large_slices():
    """Test that periodic_getitem rejects slices larger than size."""
    a = torch.arange(5)

    try:
        periodic_getitem(a, slice(0, 10))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "too large" in str(e)


# ============================================================================
# Tests for periodic_getitem_extended (new, supports multi-period)
# ============================================================================

def test_extended_basic():
    """Test that extended works for basic cases too."""
    a = torch.arange(5)

    # Normal slice
    result = periodic_getitem_extended(a, slice(1, 3))
    assert result.tolist() == [1, 2]

    # Full slice
    result = periodic_getitem_extended(a, slice(0, 5))
    assert result.tolist() == [0, 1, 2, 3, 4]


def test_extended_wrap_within_period():
    """Test wrap-around within one period works in extended."""
    a = torch.arange(5)

    # Start from negative, wrap within period
    result = periodic_getitem_extended(a, slice(-2, 2))
    assert result.tolist() == [3, 4, 0, 1]

    result = periodic_getitem_extended(a, slice(-1, 2))
    assert result.tolist() == [4, 0, 1]


def test_extended_larger_than_size():
    """Test slices larger than tensor size (multi-period)."""
    a = torch.arange(3)  # [0, 1, 2]

    # Slice from -2 to 7 (9 elements)
    result = periodic_getitem_extended(a, slice(-2, 7))
    assert result.tolist() == [1, 2, 0, 1, 2, 0, 1, 2, 0]

    # Slice from 0 to 6 (2 full periods)
    result = periodic_getitem_extended(a, slice(0, 6))
    assert result.tolist() == [0, 1, 2, 0, 1, 2]

    # Slice from 0 to 7 (2 full periods + 1)
    result = periodic_getitem_extended(a, slice(0, 7))
    assert result.tolist() == [0, 1, 2, 0, 1, 2, 0]

    # Slice from 1 to 8 (start offset + 2 full periods + partial)
    result = periodic_getitem_extended(a, slice(1, 8))
    assert result.tolist() == [1, 2, 0, 1, 2, 0, 1]


def test_extended_very_large_slice():
    """Test with slice much larger than size."""
    a = torch.arange(3)  # [0, 1, 2]

    # 10 elements starting from 0
    result = periodic_getitem_extended(a, slice(0, 10))
    assert result.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

    # The original error case: slice(-5, 37) on size 32
    b = torch.arange(32)
    result = periodic_getitem_extended(b, slice(-5, 37))
    assert len(result) == 42
    # First 5 elements should be 27, 28, 29, 30, 31
    assert result[:5].tolist() == [27, 28, 29, 30, 31]
    # Next 32 elements should be 0..31
    assert result[5:37].tolist() == list(range(32))
    # Last 5 elements should be 0, 1, 2, 3, 4
    assert result[37:].tolist() == [0, 1, 2, 3, 4]


def test_extended_multidimensional():
    """Test with multi-dimensional tensors."""
    # 2D tensor
    a = torch.arange(12).reshape(3, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    #  [8, 9, 10, 11]]

    # Large slice on first dim only
    result = periodic_getitem_extended(a, slice(0, 6), slice(None))
    assert result.shape == (6, 4)
    assert result[0].tolist() == [0, 1, 2, 3]
    assert result[3].tolist() == [0, 1, 2, 3]  # Wrapped

    # Large slice on second dim only
    result = periodic_getitem_extended(a, slice(None), slice(-1, 6))
    assert result.shape == (3, 7)
    assert result[0].tolist() == [3, 0, 1, 2, 3, 0, 1]


def test_extended_3d():
    """Test with 3D tensor (like the chunk_decode use case)."""
    # Shape: (2, 4, 3)
    a = torch.arange(24).reshape(2, 4, 3)

    # Large slice on middle dimension
    result = periodic_getitem_extended(a, slice(None), slice(-1, 6), slice(None))
    assert result.shape == (2, 7, 3)

    # Large slice on last dimension
    result = periodic_getitem_extended(a, slice(None), slice(None), slice(0, 6))
    assert result.shape == (2, 4, 6)


def test_extended_empty_slice():
    """Test empty slices."""
    a = torch.arange(5)

    result = periodic_getitem_extended(a, slice(2, 2))
    assert result.shape == (0,)

    # Negative n_elements
    result = periodic_getitem_extended(a, slice(3, 1))
    assert result.shape == (0,)


def test_extended_exact_size():
    """Test slice exactly equal to size."""
    a = torch.arange(5)

    # Starting from 0
    result = periodic_getitem_extended(a, slice(0, 5))
    assert result.tolist() == [0, 1, 2, 3, 4]

    # Starting from 2 (wrap around)
    result = periodic_getitem_extended(a, slice(2, 7))
    assert result.tolist() == [2, 3, 4, 0, 1]


# ============================================================================
# Compatibility: extended matches original for valid original cases
# ============================================================================

def test_extended_matches_original_for_small_slices():
    """Verify extended matches original for slices within size."""
    for size in [3, 5, 7, 10]:
        a = torch.arange(size)
        # Forward slices
        for start in range(size):
            for stop in range(start, size + 1):
                s = slice(start, stop)
                orig = periodic_getitem(a, s)
                ext = periodic_getitem_extended(a, s)
                assert orig.tolist() == ext.tolist(), \
                    f"Mismatch for size={size}, slice({start}, {stop})"


if __name__ == '__main__':
    print('--- periodic_getitem tests ---')
    test_periodic_getitem_basic()
    print('test_periodic_getitem_basic passed')

    test_periodic_getitem_wrap_around()
    print('test_periodic_getitem_wrap_around passed')

    test_periodic_getitem_negative_start()
    print('test_periodic_getitem_negative_start passed')

    test_periodic_getitem_empty_slice()
    print('test_periodic_getitem_empty_slice passed')

    test_periodic_getitem_rejects_large_slices()
    print('test_periodic_getitem_rejects_large_slices passed')

    print('\n--- periodic_getitem_extended tests ---')
    test_extended_basic()
    print('test_extended_basic passed')

    test_extended_wrap_within_period()
    print('test_extended_wrap_within_period passed')

    test_extended_larger_than_size()
    print('test_extended_larger_than_size passed')

    test_extended_very_large_slice()
    print('test_extended_very_large_slice passed')

    test_extended_multidimensional()
    print('test_extended_multidimensional passed')

    test_extended_3d()
    print('test_extended_3d passed')

    test_extended_empty_slice()
    print('test_extended_empty_slice passed')

    test_extended_exact_size()
    print('test_extended_exact_size passed')

    print('\n--- Compatibility tests ---')
    test_extended_matches_original_for_small_slices()
    print('test_extended_matches_original_for_small_slices passed')

    print('\nAll tests passed!')

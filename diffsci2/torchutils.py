import torch


def broadcast_from_below(t: torch.Tensor, x: torch.Tensor):

    """
    This is an important function, so we guessed it deserved a comprehensive
    explanation. The purpose of this function is to grab a lower dimensional
    Tensor (in here named t) and make its shape the same dimension as the shape
    of a upper dimensional vector (in here named x) by appending ones at the
    end of its dimension. The elements of the vector won't change. I think an
    example may enlighten what we mean:

    If t.shape = (1,2,3) and x.shape = (5,6,7,8,9,10) and we run the following
    line of code

    new_t = broadcast_from_below(t, x)

    We should get the following:

    new_t.shape = (1,2,3,1,1,1)

    As you can see, we just kept appending ones at the end of t.shape until it
    matched x.shape so it can be properly bradcasted.

    Parameters
    t : torch.Tensor of shape (nbatch,)
        tensor to be broadcasted by appending dimensions at the end
    x : torch.Tensor of shape (nbatch, ...)
        tensor to be broadcasted
    """
    if x.ndim < t.ndim:
        raise ValueError(
            "The number of dimensions of the x tensor must be greater or" +
            " equal to the number of dimensions of the t tensor"
        )

    newshape = t.shape + (1,)*(x.ndim-t.ndim)
    new_t = t.view(newshape).to(x)
    return new_t


def to_torch_tensor(x, device="cpu"):
    """
    Transform x to torch.Tensor if it is not already a torch.Tensor,
    and move it to device.

    Parameters
    ----------
    x : torch.Tensor or array-like
    device : str
        Device to move the tensor to

    Returns
    -------
    x : torch.Tensor
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x, device=device)


def linear_interpolation(x1, x2, n):
    return torch.stack([x1 + (x2 - x1) * i / (n - 1) for i in range(n)])


def dict_map(func, d):
    if isinstance(d, dict):
        return {k: dict_map(func, v) for k, v in d.items()}
    else:
        return func(d)


def dict_unsqueeze(d, dim):
    f = lambda x: torch.unsqueeze(x, dim)  # noqa: E731
    return dict_map(f, d)


def dict_squeeze(d, dim):
    f = lambda x: torch.squeeze(x, dim)  # noqa: E731
    return dict_map(f, d)


def dict_to(d, device):
    f = lambda x: x.to(device)  # noqa: E731
    return dict_map(f, d)


def load_submodule(model, checkpoint_path, model_name="model"):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    # Create new state dict with modified keys
    new_state_dict = {}
    prefix = model_name + "."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove prefix
            new_state_dict[new_key] = value

    # Load the modified state dict
    model.load_state_dict(new_state_dict)
    return model


def periodic_getitem(tensor, *indices):
    """Extract periodic slice from tensor with dimension-by-dimension wrapping.

    Usage:
        periodic_getitem(a, slice(7, 2), slice(None), slice(3, 1))
        periodic_getitem(a, slice(7, 2))  # for 1D
    """
    if not indices:
        return tensor

    result = tensor
    offset = 0  # track dimension offset as we squeeze integer indices

    for dim_orig, idx in enumerate(indices):
        dim = dim_orig - offset

        if isinstance(idx, slice):
            size = result.shape[dim]
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else size
            step = idx.step if idx.step is not None else 1

            if abs(start - stop) > size:
                raise ValueError(f"Slice {idx} is too large for dimension {dim} with size {size}")
            # Normalize negatives
            start = start % size if start < 0 else start
            stop = stop % size if stop < 0 else stop

            # Normalize greater than size
            start = start % size if start > size else start
            stop = stop % size if stop > size else stop

            if step == 1 and stop < start:
                # Wrap around
                tail = result.narrow(dim, start, size - start)
                head = result.narrow(dim, 0, stop)
                result = torch.cat([tail, head], dim=dim)
            elif step == 1:
                # Normal slice
                result = result.narrow(dim, start, max(0, stop - start))
            else:
                # With step
                raise NotImplementedError("Only step=1 supported in periodic_getitem")
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    return result


# Cleaner version without step support (since you said you don't need it)
def periodic_setitem(tensor, value, *indices):
    """Assign to periodic slice in tensor (in-place). Only supports step=1.

    Args:
        tensor: tensor to modify (in-place)
        indices: tuple of slice objects (step must be None or 1)
        value: values to assign

    Usage:
        periodic_setitem_simple(a, (slice(7, 2),), values)
        periodic_setitem_simple(a, (slice(7, 2), slice(10, 3)), values)
    """
    if not isinstance(indices, tuple):
        indices = (indices,)

    # Analyze each dimension
    dim_info = []
    for dim, idx in enumerate(indices):
        if not isinstance(idx, slice):
            raise TypeError(f"Only slice indexing supported, got {type(idx)}")

        size = tensor.shape[dim]
        start = idx.start if idx.start is not None else 0
        stop = idx.stop if idx.stop is not None else size
        step = idx.step if idx.step is not None else 1

        if abs(start - stop) > size:
            raise ValueError(f"Slice {idx} is too large for dimension {dim} with size {size}")

        if step != 1:
            raise ValueError("Only step=1 supported in simple version")

        # Normalize negatives
        start = start % size if start < 0 else start
        stop = stop % size if stop < 0 else stop

        if stop < start:
            # Wrapping: [start:] + [:stop]
            n_tail = size - start
            n_head = stop
            dim_info.append([
                (slice(start, size), slice(0, n_tail)),
                (slice(0, stop), slice(n_tail, n_tail + n_head))
            ])
        else:
            # Normal
            n_elements = stop - start
            dim_info.append([
                (slice(start, stop), slice(0, n_elements))
            ])

    # Generate all combinations
    _assign_combinations(tensor, value, dim_info, 0, [], [])


def _assign_combinations(tensor, value, dim_info, current_dim, tensor_slices, value_slices):
    """Recursively assign to all region combinations."""
    if current_dim >= len(dim_info):
        # Execute assignment
        tensor[tuple(tensor_slices)] = value[tuple(value_slices)]
        return

    # Iterate through regions for current dimension
    for tensor_slice, value_slice in dim_info[current_dim]:
        _assign_combinations(
            tensor, value, dim_info, current_dim + 1,
            tensor_slices + [tensor_slice],
            value_slices + [value_slice]
        )
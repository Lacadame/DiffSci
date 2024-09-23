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

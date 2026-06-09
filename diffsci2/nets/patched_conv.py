import torch
import gc

def divide_dims(ub, window_size, lb=0):
    """
    Divides a range [lb, ub] into patches of window size window_size

    Args:
        ub (float): Upper bound of the range
        window_size (int): Window size of division
        lb (float): Lower bound of the range (default: 0.0)

    Returns:
        list: List of tensors, each representing a division of the range
    """
    patches = []
    if (ub - lb) % window_size == 0:
        n_patches = (ub - lb) // window_size
    else:
        n_patches = (ub - lb) // window_size + 1
    for i in range(n_patches):
        lbi = lb + i*window_size
        ubi = lb + (i+1)*window_size
        ubi = min(ub, ubi) - 1
        patches.append(
            (lbi, ubi+1)
        )
    return patches


def patch_conv_1d(input_tensor: torch.Tensor,
                  patch_size: int,
                  conv_cls: torch.nn.Module,
                  padding: int):
    # Assume shape is [B, C, D]
    Cin = conv_cls.in_channels
    Cout = conv_cls.out_channels
    B, C = input_tensor.shape[:2]
    assert C == Cin, f"Input channel dimension mismatch: {C} != {Cin}"
    dimensions = list(input_tensor.shape[-1:])
    input_tensor = torch.nn.functional.pad(input_tensor, [padding]*2)
    output_tensor = torch.zeros(*([B, Cout] + dimensions))
    dim_patches = [divide_dims(d, patch_size) for d in dimensions]
    for i in range(len(dim_patches[0])):
        lbx, ubx = dim_patches[0][i]
        lbxe, ubxe = lbx, ubx + 2*padding
        output_tensor[..., lbx:ubx] = conv_cls(
            input_tensor[..., lbxe:ubxe])
    return output_tensor


def patch_conv_2d(input_tensor: torch.Tensor,
                  patch_size: int,
                  conv_cls: torch.nn.Module,
                  padding: int):
    # Assume shape is [B, C, D, H]
    Cin = conv_cls.in_channels
    Cout = conv_cls.out_channels
    B, C = input_tensor.shape[:2]
    assert C == Cin, f"Input channel dimension mismatch: {C} != {Cin}"
    dimensions = list(input_tensor.shape[-2:])
    input_tensor = torch.nn.functional.pad(input_tensor, [padding]*4)
    output_tensor = torch.zeros(*([B, Cout] + dimensions))
    dim_patches = [divide_dims(d, patch_size) for d in dimensions]
    for i in range(len(dim_patches[0])):
        for j in range(len(dim_patches[1])):
            lbx, ubx = dim_patches[0][i]
            lby, uby = dim_patches[1][j]
            lbxe, ubxe = lbx, ubx + 2*padding
            lbye, ubye = lby, uby + 2*padding
            output_tensor[..., lbx:ubx, lby:uby] = conv_cls(
                input_tensor[..., lbxe:ubxe, lbye:ubye])
    return output_tensor


def patch_conv_3d(input_tensor: torch.Tensor,
                  patch_size: int,
                  conv_cls: torch.nn.Module,
                  padding: int):
    # Assume shape is [B, C, D, H, W]
    Cin = conv_cls.in_channels
    Cout = conv_cls.out_channels
    B, C = input_tensor.shape[:2]
    assert C == Cin, f"Input channel dimension mismatch: {C} != {Cin}"
    dimensions = list(input_tensor.shape[-3:])
    input_tensor = torch.nn.functional.pad(input_tensor, [padding]*6)
    output_tensor = torch.zeros(*([B, Cout] + dimensions)).to(input_tensor)
    dim_patches = [divide_dims(d, patch_size) for d in dimensions]
    for i in range(len(dim_patches[0])):
        for j in range(len(dim_patches[1])):
            for k in range(len(dim_patches[2])):
                lbx, ubx = dim_patches[0][i]
                lby, uby = dim_patches[1][j]
                lbz, ubz = dim_patches[2][k]

                lbxe, ubxe = lbx, ubx + 2*padding
                lbye, ubye = lby, uby + 2*padding
                lbze, ubze = lbz, ubz + 2*padding
                output_tensor[..., lbx:ubx, lby:uby, lbz:ubz] = conv_cls(
                    input_tensor[..., lbxe:ubxe, lbye:ubye, lbze:ubze])
    del input_tensor
    torch.cuda.empty_cache()
    gc.collect()
    return output_tensor


def get_patch_conv(dimension: int):
    if dimension == 3:
        return patch_conv_3d
    elif dimension == 2:
        return patch_conv_2d  # TODO: implement patch conv for 2D
    elif dimension == 1:
        return patch_conv_1d  # TODO: implement patch conv for 1D
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

import numpy as np
import torch.fft as tfft
import torch
import math
from typing import Tuple, List
from torch.nn import functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


# NumPy centered FFT
def fftc(
    x: np.ndarray,
    axes: Tuple[int, int] = (-2, -1)
):
    x_shifted = np.fft.ifftshift(x, axes=axes)
    k = np.fft.fft2(x_shifted, axes=axes, norm='ortho')
    k_centered = np.fft.fftshift(k, axes=axes)
    return k_centered


# NumPy inverse centered FFT
def ifftc(
    k: np.ndarray,
    axes: Tuple[int, int] = (-2, -1)
):
    k_shifted = np.fft.ifftshift(k, axes=axes)
    x = np.fft.ifft2(k_shifted, axes=axes, norm='ortho')
    x_centered = np.fft.fftshift(x, axes=axes)
    return x_centered


@torch.jit.script
def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop with a given shape.
    shape: (height, width)
    """
    h_from = (data.shape[-2] - shape[0]) // 2
    w_from = (data.shape[-1] - shape[1]) // 2
    h_to = h_from + shape[0]
    w_to = w_from + shape[1]

    return data[..., h_from:h_to, w_from:w_to]


@torch.jit.script
def zero_pad(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply symmetrical zero padding to match the given shape.
    shape: (height, width)
    Inverse operation of center_crop.
    """
    pad_h = shape[0] - data.shape[-2]
    pad_w = shape[1] - data.shape[-1]

    h_from = pad_h // 2
    h_to = pad_h - h_from
    w_from = pad_w // 2
    w_to = pad_w - w_from

    return F.pad(
        data, (w_from, w_to, h_from, h_to),
        mode='constant', value=0.0
    )


@torch.jit.script
def norm_tensor(
    x: torch.Tensor,
    axes: Tuple[int, int] = (-2, -1)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize a 4 tensor along the last two dims.
    The function can work with complex tensors.
    Returns the normalized tensor, mean and std.
    """
    mu = torch.mean(x, dim=axes, keepdim=True)
    sig = torch.std(x, dim=axes, keepdim=True)
    x = (x - mu) / sig
    return x, mu, sig


@torch.jit.script
def unnorm_tensor(
    x: torch.Tensor,
    mu: torch.Tensor,
    sig: torch.Tensor
) -> torch.Tensor:
    """
    Unnormalize a 4 tensor along the last two dims.
    The function can work with complex tensors.
    mu and sig should be computed using norm_tensor
    """
    return x * sig + mu


# Torch centered FFT
@torch.jit.script
def tfftc(
    x: torch.Tensor,
    axes: Tuple[int, int] = (-2, -1)
) -> torch.Tensor:
    """Centered FFT"""
    x_shifted = tfft.ifftshift(x, dim=axes)
    k = tfft.fft2(x_shifted, dim=axes, norm='ortho')
    k_centered = tfft.fftshift(k, dim=axes)
    return k_centered


# Torch centered inverse FFT
@torch.jit.script
def itfftc(
    k: torch.Tensor,
    axes: Tuple[int, int] = (-2, -1)
) -> torch.Tensor:
    """Inverse centered FFT"""
    k_shifted = tfft.ifftshift(k, dim=axes)
    x = tfft.ifft2(k_shifted, dim=axes, norm='ortho')
    x_centered = tfft.fftshift(x, dim=axes)
    return x_centered


# x -> S_1 x, S_2 x, ..., S_C x
@torch.jit.script
def expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """ Expand coil combined image to multi-coil image"""
    return sens_maps * x


# x_1, ..., x_C -> sum_{i=1}^C conj(S_i) * x_i
@torch.jit.script
def reduce(xc: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """ Reduce multi-coil image to match filter combined image"""
    return torch.sum(torch.conj(sens_maps) * xc, dim=1, keepdim=True)


@torch.jit.script
def forward_op(
    x: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor
) -> torch.Tensor:
    """
    Parallel imaging forward operator
    x -> M F S x
    """
    return mask * tfftc(expand(x, sens_maps))


@torch.jit.script
def adjoint_op(
    k: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor
) -> torch.Tensor:
    """
    Parallel imaging adjoint operator
    k -> S^H F^{-1} M k
    """
    return reduce(itfftc(mask * k), sens_maps)


@torch.jit.script
def adjoint_forward(
    x: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor
) -> torch.Tensor:
    """
    First applies forward operator A, then adjoint operator A^H
    x -> S^H F^{-1} M F S x
    """
    return adjoint_op(forward_op(x, mask, sens_maps), mask, sens_maps)


@torch.jit.script
def conj_grad(
    b: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor,
    delta: torch.Tensor, num_iter: int = 10
):
    """
    Conjugate gradient algorithm to solve Cx = b
    where C = (I + 1 / delta ** 2 A^H A)
    b: input tensor, A: forward operator
    """
    delta_sq_inv = 1.0 / (delta * delta)

    # Initialize x, r, and p
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    # Compute initial norm squared of r
    rs_old = torch.sum(r.conj() * r, dim=(-2, -1), keepdim=True).real

    for _ in range(num_iter):
        # Compute Cp
        Cp = p + delta_sq_inv * adjoint_forward(p, mask, sens_maps)

        # Compute <p, Cp>
        pCp = torch.sum(p.conj() * Cp, dim=(-2, -1), keepdim=True).real

        # Compute step size with numerical stability
        alpha = rs_old / pCp

        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Cp

        # Compute new residual norm squared
        rs_new = torch.sum(r.conj() * r, dim=(-2, -1), keepdim=True).real

        # Compute next search direction
        beta = rs_new / rs_old
        p = r + beta * p

        # Update rs_old for next iteration
        rs_old = rs_new
    return x


@torch.jit.script
def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to channel dimension for real-valued networks."""
    x = torch.view_as_real(x)
    b, c, h, w, two = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


@torch.jit.script
def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert channel dimension back to complex tensor."""
    b, c2, h, w = x.shape
    c = c2 // 2
    out = x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    return torch.view_as_complex(out)


@torch.jit.script
def pad(
    x: torch.Tensor
) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
    _, _, h, w = x.shape
    w_mult = ((w - 1) | 15) + 1
    h_mult = ((h - 1) | 15) + 1
    w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
    h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
    x = F.pad(x, w_pad + h_pad)
    return x, (h_pad, w_pad, h_mult, w_mult)


@torch.jit.script
def unpad(
    x: torch.Tensor,
    h_pad: List[int],
    w_pad: List[int],
    h_mult: int,
    w_mult: int,
) -> torch.Tensor:
    return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]


@torch.jit.script
def per_slice_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each slice in the batch to [0, 1] range.
    Assumes x is of shape (B, 1, H, W)
    eps is used to avoid division by zero.
    """
    batch_size = x.shape[0]
    x_flat = x.view(batch_size, -1)
    min_val = x_flat.min(dim=1, keepdim=True)[0]
    min_val = min_val.view(-1, 1, 1, 1)
    max_val = x_flat.max(dim=1, keepdim=True)[0]
    max_val = max_val.view(-1, 1, 1, 1)
    return (x - min_val) / torch.clamp(max_val - min_val, min=eps)


def batch_ssim(
        pred, target, min_target, max_target, eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes the average SSIM for a batch of reconstructions and targets.
    Targets do not need to be normalized to [0, 1] range.
    """
    B = pred.shape[0]
    # Ensure min_target and max_target have correct shape for broadcasting
    min_target = min_target.view(B, 1, 1, 1)
    max_target = max_target.view(B, 1, 1, 1)

    # Compute data range per slice
    data_ranges = torch.clamp(max_target - min_target, min=eps)

    # Normalize each slice to [0, 1] range using broadcasting
    pred_normalized = (pred - min_target) / data_ranges
    target_normalized = (target - min_target) / data_ranges

    # Calculate SSIM - inputs are already (B, 1, H, W)
    return ssim(
        pred_normalized, target_normalized,
        gaussian_kernel=False,
        kernel_size=7,
        data_range=(0.0, 1.0)
    )  # type: ignore

"""Group-wise regular Hadamard rotation for INT8 W8A8 quantization.

Rotates linear-layer activations and weights into a basis where outliers
are spread across channels, dramatically improving INT8 quantization
accuracy. The rotation is orthogonal, so `(Q x) · (Q W) = x · W` exactly
in full precision — only the quantization noise is reduced.

The construction uses Kronecker products of a 4x4 *regular* Hadamard
matrix (every row + column sums to 2) rather than the standard Sylvester
H_2 ⊗ ... ⊗ H_2, which has an all-1s column that amplifies row-wise
outliers in diffusion-model attention.

References
----------
QuaRot — Outlier-Free 4-Bit Inference in Rotated LLMs
    https://arxiv.org/abs/2404.00456

ConvRot — Group-wise rotation for DiT models
    Theorem 3.3 in the ConvRot paper specifies the H4 base matrix.

The 4-line H4 matrix below is the published construction from the
ConvRot paper — fundamental math, not copyrighted code.
"""
from __future__ import annotations

import math

import torch


# Per-(size, device, dtype) cache: the rotation matrix is a runtime
# constant once built — caching avoids the Kronecker chain cost on
# every linear forward (which happens N_LAYERS × N_STEPS times).
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized regular orthogonal Hadamard matrix.

    `size` must be a power of 4 (4, 16, 64, 256, 1024, ...). The matrix
    is constructed recursively via Kronecker products of the 4x4 base
    matrix `H4`, then normalized by sqrt(size) so `H @ H.T == I`.
    """
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")

    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    h4 = torch.tensor(
        [
            [ 1,  1,  1, -1],
            [ 1,  1, -1,  1],
            [ 1, -1,  1,  1],
            [-1,  1,  1,  1],
        ],
        dtype=dtype,
        device=device,
    )

    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h = h / (size ** 0.5)
    _HADAMARD_CACHE[cache_key] = h
    return h


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate a (out_features, in_features) weight offline: ``W' = W @ H^T``.

    The input dimension is split into contiguous blocks of `group_size`;
    each block is rotated independently. This is the "group-wise" part —
    full-dim Hadamard would be too memory-intensive for hidden sizes in
    the thousands.

    Must be paired with a corresponding `rotate_activation(x, h, group_size)`
    at inference time so the matmul `x @ W^T` produces the same result as
    the un-rotated equivalent (orthogonality of H).
    """
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
    n_groups = in_features // group_size
    grouped = weight.view(out_features, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    rotated = torch.matmul(grouped, h_t)
    return rotated.reshape(out_features, in_features)


def rotate_activation(x: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate an activation online: ``x' = x @ H`` (last dim).

    Shape is preserved; the last dimension is grouped and each group
    rotated by `h`. Called inside `LinearInt8ConvRot.forward` before
    quantizing `x` to INT8 — the rotation pushes outliers away from any
    single channel, so the per-row max-abs scale captures less of an
    outlier "ceiling" and the INT8 grid uses its 256 levels efficiently.
    """
    *prefix, features = x.shape
    if features % group_size != 0:
        raise ValueError(
            f"features ({features}) must be divisible by group_size ({group_size})"
        )
    n_groups = features // group_size
    grouped = x.view(*prefix, n_groups, group_size)
    h_dev = h.to(dtype=x.dtype, device=x.device)
    rotated = torch.matmul(grouped, h_dev)
    return rotated.view(*prefix, features)

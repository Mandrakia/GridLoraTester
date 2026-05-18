"""INT8 W8A8 + ConvRot linear layer — `nn.Linear` drop-in replacement.

Replaces a standard `nn.Linear` with:

  Storage:
    weight       int8 (out, in)   — ConvRot-rotated then per-row quantized
    weight_scale fp32 (out, 1)    — per-row max-abs / 127
    H            fp32 (G, G)      — Hadamard rotation matrix (cached globally)
    bias         bf16 (out,)      — kept high-precision

  Forward (B x ... x in -> B x ... x out):
    x' = rotate_activation(x, H, G)             # online Hadamard
    x_int8, x_scale = quantize_axiswise(x', -1) # per-row activation quant
    y_int32 = torch._int_mm(x_int8, weight.T)   # native cuBLASLt IMMA path
    y = y_int32.to(fp32) * x_scale * weight_scale.T  # broadcast dequant
    y = y.to(bf16) + bias

`torch._int_mm` is PyTorch's native INT8 GEMM. On Ampere (sm80/sm86) it
dispatches to cuBLASLt INT8 IMMA tensor cores — the path that gives the
~2x speedup vs BF16. No custom Triton kernel needed for the headline
speedup; Triton fusion is a future optimization (saves ~5-10% by fusing
quantize_rowwise + matmul + dequant into one kernel).

Per-row activation quantization is essential here: outliers tend to
appear in specific tokens (not specific channels), so a per-token scale
captures them without inflating the scale for every other token.
ConvRot's rotation then further reduces the per-token outlier magnitude.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .hadamard import build_hadamard


# Group size for the Hadamard rotation. 256 = 4^4 is a sweet spot:
# - Small enough that the rotation matrix fits in L2 cache (256 * 256 *
#   4 bytes = 256 KB)
# - Large enough that the outlier spreading is meaningful
# - Common GCD of Flux2 hidden sizes (4096, 12288, 16384, 24576, 36864
#   are all divisible by 256)
CONVROT_GROUP_SIZE = 256


# ---- Quantization utilities ----------------------------------------------


def quantize_int8_per_row(x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row symmetric INT8 quantization.

    Returns ``(x_int8, scale)`` where ``x_int8`` is ``(..., N)`` int8 and
    ``scale`` is ``(...,)`` float32 (same leading shape as x without the
    last dim, then unsqueezed back to broadcast).

    Symmetric (no zero-point): each row uses ``scale = max(abs(row))/127``,
    so values are mapped into the ``[-127, 127]`` range with -128 clamped.
    1e-30 floor avoids div-by-zero for all-zero rows (rare but real for
    masked or zero-initialized layers during warm-up).
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (x.float() / scale).round().clamp_(-128, 127).to(torch.int8)
    return q, scale.squeeze(-1)


def quantize_weight_per_row(w: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row weight quantization. ``w`` is ``(out, in)``; returns
    ``(w_int8, scale)`` with scale shape ``(out, 1)`` for broadcast on
    the matmul output.
    """
    abs_max = w.abs().amax(dim=1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (w.float() / scale).round().clamp_(-128, 127).to(torch.int8)
    return q, scale


def dequantize_weight(w_int8: Tensor, scale: Tensor) -> Tensor:
    """Reverse `quantize_weight_per_row` to recover a bf16 weight (for
    LoRA bake-in: dequant → add ΔW → rotate → re-quant).
    """
    return (w_int8.float() * scale).to(torch.bfloat16)


# ---- LinearInt8ConvRot module --------------------------------------------


class LinearInt8ConvRot(nn.Module):
    """`nn.Linear` replacement with W8A8 INT8 + group-wise Hadamard rotation.

    Constructed via `LinearInt8ConvRot.from_bf16_linear(linear)` which
    rotates the weight, quantizes it to int8, and packs everything
    inline. The original bf16 weight is NOT kept around — to bake a LoRA
    delta later, `lora.bake_lora_into` re-derives the bf16 source via
    `dequantize_weight(self.weight, self.weight_scale)`.
    """

    # Tag so external code can `isinstance(m, LinearInt8ConvRot)` filter
    # quickly without importing the class everywhere. Equivalent to
    # `_is_quantized = True` flag.
    _is_int8_convrot = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = CONVROT_GROUP_SIZE,
        bias: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size "
                f"({group_size}) for ConvRot — fall back to a plain nn.Linear "
                f"for this layer (see quantize.py's exclusion logic)."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Buffers (not Parameters): we don't train these and don't want
        # them shown by `.parameters()` to optimizers.
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros((out_features, 1), dtype=torch.float32, device=device),
        )
        # Hadamard rotation matrix as a non-persistent buffer.
        # Why a buffer (not a Python-dict cache lookup inside forward):
        # torch.compile with mode='reduce-overhead' captures the forward
        # as CUDA graphs. A tensor returned from a Python-side cache is
        # treated by cudagraphs as a recycled output of a prior call —
        # cudagraphs then rejects re-using it as an input with
        # "accessing tensor output of CUDAGraphs that has been
        # overwritten by a subsequent run". A registered buffer is a
        # known-static module attribute, so cudagraphs treats it as a
        # fixed pointer it can reference safely.
        # bf16 storage matches the activation dtype throughout Klein —
        # no per-call dtype cast inside forward, no allocation churn.
        # persistent=False excludes it from state_dict (the cache file
        # would otherwise store 256x256 bf16 × 149 modules = 19 MB of
        # fully-reconstructible data). 256x256 bf16 = 128 KB per layer
        # × 149 layers ≈ 19 MB of VRAM total — acceptable.
        self.register_buffer(
            "hadamard_h",
            build_hadamard(group_size, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        if bias:
            # Bias stays bf16 — it's tiny and biases are sensitive to
            # quantization noise (~5% PSNR loss observed when quantized).
            self.bias = nn.Parameter(
                torch.zeros((out_features,), dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
        else:
            self.bias = None

    # ---- Constructors ----

    @classmethod
    def from_bf16_linear(
        cls,
        linear: nn.Linear,
        group_size: int = CONVROT_GROUP_SIZE,
    ) -> "LinearInt8ConvRot":
        """Build an INT8 ConvRot layer from a plain bf16 nn.Linear.

        Performs the offline weight rotation and per-row quantization.
        The compute happens on whatever device the input linear's weight
        lives on (CPU or GPU), then the resulting buffers stay there.
        """
        from .hadamard import rotate_weight

        if linear.in_features % group_size != 0:
            raise ValueError(
                f"Linear({linear.in_features}, {linear.out_features}) has "
                f"in_features not divisible by group_size {group_size}"
            )
        instance = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )
        # Rotation requires a float type; cast through fp32 to avoid
        # accuracy loss in the Kronecker matmul, then quantize.
        w = linear.weight.detach().to(torch.float32)
        h = build_hadamard(group_size, device=w.device, dtype=w.dtype)
        w_rot = rotate_weight(w, h, group_size=group_size)
        q, scale = quantize_weight_per_row(w_rot)
        instance.weight = q
        instance.weight_scale = scale
        if linear.bias is not None:
            instance.bias = nn.Parameter(
                linear.bias.detach().to(torch.bfloat16), requires_grad=False
            )
        return instance

    # ---- Forward ----

    def forward(self, x: Tensor) -> Tensor:
        # cuBLASLt INT8 IMMA requires the activation tensor to be 2D
        # (M, K). Reshape, run, then unflatten back to the input's
        # original leading shape.
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])

        # Online activation rotation. Inlined here (instead of calling
        # `rotate_activation`) so torch.compile sees a single
        # graph node — no function-call boundary, no per-call dtype
        # cast (`self.hadamard_h` is pre-stored in bf16 to match x).
        h = self.hadamard_h
        if h.dtype != x.dtype:
            h = h.to(x.dtype)
        gs = self.group_size
        x_grouped = x_2d.reshape(x_2d.shape[0], x_2d.shape[1] // gs, gs)
        x_rot = (x_grouped @ h).reshape(x_2d.shape)

        # Small-batch fallback: torch._int_mm has overhead that beats
        # F.linear only above ~16 rows. For very narrow batches (the
        # initial timestep embedding pass, where M=1), dequant the
        # weight and use a plain bf16 matmul. The branch lives in
        # forward — it doesn't change the graph topology so compile
        # specializes on shape, not values, and the swap is silent.
        if x_2d.shape[0] > 16:
            # Per-row activation quant: scale shape (M, 1) for broadcast.
            x_int8, x_scale = quantize_int8_per_row(x_rot)
            x_scale = x_scale.unsqueeze(-1)
            y_int32 = torch._int_mm(x_int8, self.weight.T)
            # res = (y_int32 * x_scale * weight_scale^T) cast to bf16.
            # The dequant multiplication uses fp32 to avoid bf16 overflow
            # on the int32 intermediates.
            y = (
                y_int32.to(torch.float32)
                .mul_(x_scale)
                .mul_(self.weight_scale.T)
                .to(torch.bfloat16)
            )
        else:
            w_bf16 = (self.weight.float() * self.weight_scale).to(x.dtype)
            y = torch.nn.functional.linear(x_rot.to(x.dtype), w_bf16, None)

        if self.bias is not None:
            y = y + self.bias.to(y.dtype)

        return y.reshape(*x_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}, "
            f"quant=INT8_ConvRot"
        )

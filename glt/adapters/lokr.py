"""LoKr (Low-rank Kronecker) adapter support.

LoKr is an alternative parameter-efficient adapter to LoRA: instead of
`delta_W = lora_B @ lora_A` (rank-r factorization), it stores
`delta_W = scale * kron(w1, w2)` where w1, w2 are small dense matrices.
Diffusers/peft only handle LoRA, so we route LoKr through a parallel code
path: detect → load → register forward hooks on the target Linears that add
`strength * scale * kron(w1, w2) @ x` to each output. The Kronecker product
is applied without materializing the full delta_W (which can reach ~40 GB
for a full FLUX.2 LoKr) — two small matmuls per forward suffice.
"""
from __future__ import annotations

import re


# Maps a LoKr source-key suffix within 'diffusion_model.double_blocks.<N>.X'
# (or single_blocks) to (diffusers_submodule_paths, split_factor).
# split_factor=3 means LoKr stores fused QKV but diffusers exposes 3 separate
# Linears (slice the kron output along dim 0 into 3 equal chunks).
_LOKR_DOUBLE_MAP = {
    "img_attn.proj": (["transformer_blocks.{i}.attn.to_out.0"], 1),
    "img_attn.qkv":  (["transformer_blocks.{i}.attn.to_q",
                       "transformer_blocks.{i}.attn.to_k",
                       "transformer_blocks.{i}.attn.to_v"], 3),
    "img_mlp.0":     (["transformer_blocks.{i}.ff.linear_in"], 1),
    "img_mlp.2":     (["transformer_blocks.{i}.ff.linear_out"], 1),
    "txt_attn.proj": (["transformer_blocks.{i}.attn.to_add_out"], 1),
    "txt_attn.qkv":  (["transformer_blocks.{i}.attn.add_q_proj",
                       "transformer_blocks.{i}.attn.add_k_proj",
                       "transformer_blocks.{i}.attn.add_v_proj"], 3),
    "txt_mlp.0":     (["transformer_blocks.{i}.ff_context.linear_in"], 1),
    "txt_mlp.2":     (["transformer_blocks.{i}.ff_context.linear_out"], 1),
}
_LOKR_SINGLE_MAP = {
    "linear1":       (["single_transformer_blocks.{i}.attn.to_qkv_mlp_proj"], 1),
    "linear2":       (["single_transformer_blocks.{i}.attn.to_out"], 1),
}


def is_lokr_file(path) -> bool:
    """`True` if the `.safetensors` at `path` contains LoKr keys (`.lokr_w1`
    or `.lokr_w2`). Reads safetensors metadata only — cheap."""
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if k.endswith(".lokr_w1") or k.endswith(".lokr_w2"):
                    return True
    except Exception:
        return False
    return False


def _map_lokr_key_to_diffusers(module_key: str, total_out_dim: int):
    """Map a LoKr module key (`'double_blocks.0.img_attn.qkv'`, etc.) to a
    list of `(diffusers_submodule_path, slice_or_None)`. Fused-QKV LoKrs
    return 3 entries each with a `(start, end)` slice along the kron output
    dim."""
    m = re.match(r"double_blocks\.(\d+)\.(.+)$", module_key)
    if m:
        i, sub = int(m.group(1)), m.group(2)
        spec = _LOKR_DOUBLE_MAP.get(sub)
    else:
        m = re.match(r"single_blocks\.(\d+)\.(.+)$", module_key)
        if not m:
            raise KeyError(f"unrecognized LoKr module key: {module_key}")
        i, sub = int(m.group(1)), m.group(2)
        spec = _LOKR_SINGLE_MAP.get(sub)
    if spec is None:
        raise KeyError(f"unknown LoKr submodule for block {i}: {sub}")
    paths, split = spec
    if split == 1:
        return [(paths[0].format(i=i), None)]
    chunk = total_out_dim // split
    return [(p.format(i=i), (j * chunk, (j + 1) * chunk)) for j, p in enumerate(paths)]


def load_lokr_data(path, device="cuda", dtype=None):
    """Load a LoKr safetensors file and return a dict mapping each diffusers
    submodule path to `(w1, w2, scale, slice_or_None)`.

    Raises `NotImplementedError` for decomposed LoKr (`lokr_w1_a/b`,
    `lokr_w2_a/b`, `lokr_t2`) — most current trainers (ai-toolkit default)
    save direct w1/w2 only.
    """
    from safetensors.torch import load_file
    sd = load_file(str(path), device="cpu")
    modules: dict[str, dict] = {}
    for k, v in sd.items():
        kk = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k
        for suffix in (".lokr_w1", ".lokr_w2", ".alpha",
                       ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a",
                       ".lokr_w2_b", ".lokr_t2", ".dora_scale"):
            if kk.endswith(suffix):
                modules.setdefault(kk[: -len(suffix)], {})[suffix[1:]] = v
                break
        else:
            raise ValueError(f"unrecognized LoKr key: {k}")

    result: dict[str, tuple] = {}
    for mod_key, parts in modules.items():
        if "lokr_w1" not in parts or "lokr_w2" not in parts:
            raise NotImplementedError(
                f"decomposed LoKr (lokr_w*_a/b or lokr_t2) not supported yet for "
                f"module '{mod_key}'. File: {path}"
            )
        w1 = parts["lokr_w1"]
        w2 = parts["lokr_w2"]
        # Direct w1/w2 convention (ai-toolkit + ComfyUI both agree): scale=1.0.
        # The stored `.alpha` equals lora_dim by construction in this case.
        scale = 1.0
        if dtype is None:
            dtype = w1.dtype
        w1 = w1.to(device=device, dtype=dtype)
        w2 = w2.to(device=device, dtype=dtype)
        total_out = w1.shape[0] * w2.shape[0]
        for diffusers_path, slc in _map_lokr_key_to_diffusers(mod_key, total_out):
            result[diffusers_path] = (w1, w2, scale, slc)
    return result


def _lokr_apply_kron(x, w1, w2, slc):
    """Compute `kron(w1, w2) @ x_last_dim` without materializing the kron.

    With `w1=[m,n]`, `w2=[p,q]` and `x` last dim = `n*q`, the output last dim
    is `m*p` (or its slice).

    Math: `Y[..., m, p] = sum_n w1[m, n] * z[..., n, p]`
          where `z[..., n, p] = sum_q w2[p, q] * x_view[..., n, q]`
    so `z = x_view @ w2.T`, `Y = w1 @ z`  (matmul broadcasts leading dims).
    `Y` is contiguous in `[..., m, p]` layout → reshape to `[..., m*p]` is
    free, which matches the kron flatten order (kron iterates m outer, p
    inner). Allocates only TWO transient tensors of size `[..., n, p]` and
    `[..., m, p]`; at batch=7, T=4096, n=8, p=1536 that's ~700 MB each.
    """
    *batch, _ = x.shape
    m, n = w1.shape
    p, q = w2.shape
    xr = x.reshape(*batch, n, q)
    z = xr @ w2.t()                        # [..., n, p]
    y = w1 @ z                             # [..., m, p] in contiguous order
    del z
    y = y.reshape(*batch, m * p)           # view, no realloc
    if slc is not None:
        a, b = slc
        y = y[..., a:b]                    # view
    return y


class LoKrHookManager:
    """Owns the lifecycle of LoKr forward hooks on a diffusers transformer.

    `apply()` at the start of a row; `remove()` at the end. The hooks add a
    `strength * scale * kron(w1, w2) @ x` delta on top of each target
    Linear's output — composes cleanly with peft LoRA wrappers (sticky base
    LoRAs) since both are additive on the output side.
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self.handles: list = []

    def apply(self, lokr_data: dict, strength: float = 1.0) -> int:
        for path, (w1, w2, scale, slc) in lokr_data.items():
            try:
                module = self.transformer.get_submodule(path)
            except AttributeError:
                print(f"  [lokr][warn] no submodule '{path}' on transformer — skipping")
                continue
            # Sanity check: expected output dim must match this Linear's out_features.
            expected_out = (slc[1] - slc[0]) if slc is not None else (w1.shape[0] * w2.shape[0])
            actual_out = getattr(module, "out_features", None)
            if actual_out is None:
                w = getattr(module, "weight", None)
                actual_out = w.shape[0] if w is not None else None
            if actual_out is not None and actual_out != expected_out:
                print(f"  [lokr][warn] shape mismatch at '{path}': expected "
                      f"out={expected_out} but module has {actual_out} — skipping")
                continue
            effective = scale * float(strength)
            self.handles.append(module.register_forward_hook(
                self._make_hook(w1, w2, effective, slc)
            ))
        return len(self.handles)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    @staticmethod
    def _make_hook(w1, w2, scale: float, slc):
        def _hook(_module, args, output):
            x = args[0]
            delta = _lokr_apply_kron(x, w1, w2, slc)
            # In-place fused add (output += scale * delta). Saves three
            # allocations vs `output + (delta * scale).to(output.dtype)`:
            # the `delta * scale`, the `.to(...)`, and the `+` result. Safe
            # because Linear/LoRA wrappers return a fresh tensor on each
            # forward call; we're in torch.no_grad() at inference.
            if delta.dtype != output.dtype:
                delta = delta.to(output.dtype)
            output.add_(delta, alpha=scale)
            return output
        return _hook

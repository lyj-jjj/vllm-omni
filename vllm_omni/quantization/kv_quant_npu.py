# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).
"""

from __future__ import annotations

import math

import torch

# Hadamard rotation matrix for QuaRot-style preprocessing (head_dim must match).
_ROT_MATRIX: torch.Tensor | None = None

_FP8_KV_LABELS = frozenset({"fp8", "fp8_e4m3", "fp8_e4m3fn"})


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """True if config requests FP8-style KV / QKV quantization for the NPU FA path."""
    return kv_cache_dtype in _FP8_KV_LABELS


def fp8_rotate_quant_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str = "BNSD",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run NPU fused attention with dynamic FP8 Q/K/V and optional QuaRot preprocess.

    Args:
        query, key, value: Tensors in ``layout`` order (default BNSD: batch, heads, seq, dim).
        layout: ``BNSD`` or ``BSND`` for ``npu_fused_infer_attention_score_v2``.
        softmax_scale: If None, uses ``1 / sqrt(head_dim)``.

    Returns:
        Attention output in the same layout as inputs.
    """
    try:
        import torch_npu
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess
        from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode, create_rot
    except ImportError as e:
        raise ImportError(
            "fp8_rotate_quant_fa requires torch_npu, MindIE-SD (mindiesd), and MSModelSlim. "
            "See https://gitcode.com/Ascend/MindIE-SD and https://gitcode.com/Ascend/msmodelslim"
        ) from e

    global _ROT_MATRIX

    out_dtype = query.dtype
    device = query.device

    if layout == "BNSD":
        _, n, s, d = query.shape
    elif layout == "BSND":
        _, s, n, d = query.shape
    else:
        raise ValueError(f"fp8_rotate_quant_fa: unsupported layout {layout!r}, expected BNSD or BSND")

    if _ROT_MATRIX is None or _ROT_MATRIX.shape[0] != d:
        _ROT_MATRIX = create_rot(QuaRotMode.HADAMARD, d, seed=425500)
    if _ROT_MATRIX.device != device:
        _ROT_MATRIX = _ROT_MATRIX.to(device)
    rot = _ROT_MATRIX

    q_f = torch.matmul(query, rot)
    k_f = torch.matmul(key, rot)

    q, q_scale = fa_block_quant_preprocess(
        q_f, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    k, k_scale = fa_block_quant_preprocess(
        k_f, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    v, v_scale = fa_block_quant_preprocess(
        value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    out = torch_npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        input_layout=layout,
        num_query_heads=n,
        softmax_scale=scale,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        query_quant_mode=7,
        key_quant_mode=7,
        value_quant_mode=7,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=out_dtype,
    )[0]

    if out.shape[2] != s:
        if layout == "BNSD":
            out = out[:, :, :s, :]
        elif layout == "BSND":
            out = out[:, :s, :, :]

    return out

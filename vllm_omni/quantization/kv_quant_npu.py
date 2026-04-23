# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FP8 quantization utilities for diffusion attention tensors.
Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).
"""
import math
import logging
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

FP8_KV_LABELS = frozenset({"fp8", "fp8_e4m3", "fp8_e4m3fn"})
DEFAULT_QUANT_MODE = 7
DEFAULT_Q_BLOCK_SIZE = 128
DEFAULT_KV_BLOCK_SIZE = 256
MAX_SEQ_LEN = 2147483647
ROT_MATRIX_SEED = 425500

_ROT_MATRIX_CACHE = {}
_IMPORTED_FLAG = False


_FA_BLOCK_QUANT = None
_QUAROT_MODE = None
_CREATE_ROT = None
_TORCH_NPU = None


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    return kv_cache_dtype in FP8_KV_LABELS


def _lazy_import():
    global _IMPORTED_FLAG, _FA_BLOCK_QUANT, _QUAROT_MODE, _CREATE_ROT, _TORCH_NPU
    if _IMPORTED_FLAG:
        return

    try:
        import torch_npu
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess
        from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode, create_rot

        _TORCH_NPU = torch_npu
        _FA_BLOCK_QUANT = fa_block_quant_preprocess
        _QUAROT_MODE = QuaRotMode
        _CREATE_ROT = create_rot
        _IMPORTED_FLAG = True

    except ImportError as e:
        raise ImportError(
            f"Failed to load NPU quantization dependencies: {str(e)}\n"
            "Required: torch_npu, mindiesd, msmdelslim"
        ) from e


def _get_rot_matrix(dim: int, device: torch.device):
    key = (dim, device)
    if key in _ROT_MATRIX_CACHE:
        return _ROT_MATRIX_CACHE[key]

    rot = _CREATE_ROT(_QUAROT_MODE.HADAMARD, dim, seed=ROT_MATRIX_SEED)
    rot = rot.to(device).contiguous()
    _ROT_MATRIX_CACHE[key] = rot
    return rot

def fp8_rotate_quant_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str = "BNSD",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    _lazy_import()
    torch_npu = _TORCH_NPU

    if layout == "BNSD":
        n_heads = query.size(1)
        seq_len = query.size(2)
        head_dim = query.size(3)
    elif layout == "BSND":
        seq_len = query.size(1)
        n_heads = query.size(2)
        head_dim = query.size(3)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    device = query.device
    rot_matrix = _get_rot_matrix(head_dim, device)

    q_rot = torch.matmul(query, rot_matrix)
    k_rot = torch.matmul(key, rot_matrix)

    q, q_scale = _FA_BLOCK_QUANT(
        q_rot, block_size=DEFAULT_Q_BLOCK_SIZE,
        dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    k, k_scale = _FA_BLOCK_QUANT(
        k_rot, block_size=DEFAULT_KV_BLOCK_SIZE,
        dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    v, v_scale = _FA_BLOCK_QUANT(
        value, block_size=DEFAULT_KV_BLOCK_SIZE,
        dst_type=torch_npu.float8_e4m3fn, layout=layout
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    out = torch_npu.npu_fused_infer_attention_score_v2(
        q, k, v,
        input_layout=layout,
        num_query_heads=n_heads,
        softmax_scale=scale,
        pre_tokens=MAX_SEQ_LEN,
        next_tokens=MAX_SEQ_LEN,
        query_quant_mode=DEFAULT_QUANT_MODE,
        key_quant_mode=DEFAULT_QUANT_MODE,
        value_quant_mode=DEFAULT_QUANT_MODE,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=query.dtype,
    )[0]

    if layout == "BNSD" and out.size(2) != seq_len:
        out = out[:, :, :seq_len]
    elif layout == "BSND" and out.size(1) != seq_len:
        out = out[:, :seq_len]

    return out
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Used by Hop1 (GPU worker <-> scheduler) to avoid pickling large video tensors
through the MessageQueue. Tensors above ``_SHM_TENSOR_THRESHOLD`` are copied
into a named shared-memory segment; only a lightweight metadata dict is
serialised through the queue.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput

_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB


def _tensor_to_shm(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy a tensor into POSIX shared memory and return a metadata handle.

    The shared memory segment remains alive after this call (the local fd is
    closed, but the segment persists until ``_tensor_from_shm`` unlinks it).
    """
    from multiprocessing import shared_memory

    import numpy as np

    tensor = tensor.detach().cpu().contiguous()
<<<<<<< HEAD
    if tensor.dtype == torch.bfloat16:
        # NumPy cannot materialize BF16 tensors directly, so store the raw
        # 16-bit payload and reconstruct the original dtype on unpack.
        tensor_for_storage = tensor.view(torch.uint16)
        arr = tensor_for_storage.numpy()
    else:
        arr = tensor.numpy()
=======
    original_dtype = tensor.dtype
    # NumPy does not support bfloat16; promote to float32 for the SHM
    # transfer and record the original dtype so _tensor_from_shm can
    # convert back.  The round-trip is lossless for bfloat16 values.
    if original_dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    arr = tensor.numpy()
>>>>>>> 95a07f7732900974d9e608b39f36e5b2e6518442
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:nbytes])
    np.copyto(shm_arr, arr)
    handle = {
        "__tensor_shm__": True,
        "name": shm.name,
        "shape": list(tensor.shape),
        "torch_dtype": str(original_dtype),
        "numpy_dtype": str(arr.dtype),
        "nbytes": nbytes,
    }
    shm.close()
    return handle


def _tensor_from_shm(handle: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from a shared-memory handle and free the segment."""
    from multiprocessing import shared_memory

    import numpy as np

    shm = shared_memory.SharedMemory(name=handle["name"])
    try:
        np_dtype = np.dtype(handle["numpy_dtype"])
        arr = np.ndarray(handle["shape"], dtype=np_dtype, buffer=shm.buf[: handle["nbytes"]])
        tensor = torch.from_numpy(arr.copy())
<<<<<<< HEAD
        if handle["torch_dtype"] == str(torch.bfloat16):
            tensor = tensor.view(torch.bfloat16)
=======
        # Restore the original dtype if it differs from the numpy-compatible
        # dtype used for the SHM transfer (e.g. bfloat16 → float32 → bfloat16).
        torch_dtype_str = handle.get("torch_dtype", "")
        if torch_dtype_str:
            original_dtype = getattr(torch, torch_dtype_str.replace("torch.", ""), None)
            if original_dtype is not None and tensor.dtype != original_dtype:
                tensor = tensor.to(original_dtype)
>>>>>>> 95a07f7732900974d9e608b39f36e5b2e6518442
    finally:
        shm.close()
        shm.unlink()
    return tensor


def _pack_tensor_if_large(val: torch.Tensor) -> torch.Tensor | dict:
    """Replace a tensor with an SHM handle if it exceeds the threshold."""
    if val.nelement() * val.element_size() > _SHM_TENSOR_THRESHOLD:
        return _tensor_to_shm(val)
    return val


def _unpack_if_shm_handle(val: object) -> object:
    """Reconstruct a tensor from an SHM handle dict, or return as-is."""
    if isinstance(val, dict) and val.get("__tensor_shm__"):
        return _tensor_from_shm(val)
    return val


def _pack_diffusion_fields(output: DiffusionOutput) -> DiffusionOutput:
    if output.output is not None and isinstance(output.output, torch.Tensor):
        output.output = _pack_tensor_if_large(output.output)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
<<<<<<< HEAD
        if output.trajectory_latents.nelement() * output.trajectory_latents.element_size() > _SHM_TENSOR_THRESHOLD:
            output.trajectory_latents = _tensor_to_shm(output.trajectory_latents)
    if output.trajectory_log_probs is not None and isinstance(output.trajectory_log_probs, torch.Tensor):
        if output.trajectory_log_probs.nelement() * output.trajectory_log_probs.element_size() > _SHM_TENSOR_THRESHOLD:
            output.trajectory_log_probs = _tensor_to_shm(output.trajectory_log_probs)
=======
        output.trajectory_latents = _pack_tensor_if_large(output.trajectory_latents)
    if output.trajectory_timesteps is not None and isinstance(output.trajectory_timesteps, torch.Tensor):
        output.trajectory_timesteps = _pack_tensor_if_large(output.trajectory_timesteps)
    if output.trajectory_log_probs is not None and isinstance(output.trajectory_log_probs, torch.Tensor):
        output.trajectory_log_probs = _pack_tensor_if_large(output.trajectory_log_probs)
>>>>>>> 95a07f7732900974d9e608b39f36e5b2e6518442
    return output


def pack_diffusion_output_shm(output: object) -> object:
    """Replace large tensors in diffusion worker outputs with SHM handles.

    Supports either a bare ``DiffusionOutput`` or a wrapper object carrying one
    in ``.result`` (for example ``RunnerOutput``).
    """
    if isinstance(output, DiffusionOutput):
        return _pack_diffusion_fields(output)

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _pack_diffusion_fields(result)
    return output


def _unpack_diffusion_fields(output: DiffusionOutput) -> DiffusionOutput:
<<<<<<< HEAD
    if isinstance(output.output, dict) and output.output.get("__tensor_shm__"):
        output.output = _tensor_from_shm(output.output)
    if isinstance(output.trajectory_latents, dict) and output.trajectory_latents.get("__tensor_shm__"):
        output.trajectory_latents = _tensor_from_shm(output.trajectory_latents)
    if isinstance(output.trajectory_log_probs, dict) and output.trajectory_log_probs.get("__tensor_shm__"):
        output.trajectory_log_probs = _tensor_from_shm(output.trajectory_log_probs)
=======
    output.output = _unpack_if_shm_handle(output.output)
    output.trajectory_latents = _unpack_if_shm_handle(output.trajectory_latents)
    output.trajectory_timesteps = _unpack_if_shm_handle(output.trajectory_timesteps)
    output.trajectory_log_probs = _unpack_if_shm_handle(output.trajectory_log_probs)
>>>>>>> 95a07f7732900974d9e608b39f36e5b2e6518442
    return output


def unpack_diffusion_output_shm(output: object) -> object:
    """Reconstruct tensors from SHM handles in diffusion worker outputs."""
    if isinstance(output, DiffusionOutput):
        return _unpack_diffusion_fields(output)

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _unpack_diffusion_fields(result)
    return output

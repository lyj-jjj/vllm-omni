# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations
import torch

from torch import nn
from vllm_omni.platforms import current_omni_platform

def patch_wan_rms_norm():
    '''Replace small operators with fused operators'''

    if not current_omni_platform.is_npu():
        return

    import torch_npu
    class WanRMS_norm(nn.Module):
        def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
            super().__init__()
            broadcastable_dims = (1, 1, 1) if not images else (1, 1)
            shape = (dim, *broadcastable_dims) if channel_first else (dim,)
            self.channel_first = channel_first
            self.scale = dim ** 0.5
            self.gamma = nn.Parameter(torch.ones(shape))
            self.gamma_new = None
            self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

        def forward(self, x):
            x = x.transpose(1, -1)
            if self.gamma_new is None:
                self.gamma_new = self.gamma.transpose(0, -1).reshape(-1)
            x_out = torch_npu.npu_rms_norm(x, self.gamma_new, epsilon=1e-6)
            x_out = x_out[0].transpose(1, -1)
            return x_out

    import sys
    for module_name, module in sys.modules.items():
        if hasattr(module, 'WanRMS_norm'):
            setattr(module, 'WanRMS_norm', WanRMS_norm)
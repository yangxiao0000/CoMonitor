"""
Utils for Colossal-LLaMA
"""
import torch_npu

import torch
import torch.distributed as dist

from colossalai.booster import Plugin
import os
DEVICE_ID= 0
if os.getenv('DEVICE_ID') and str.isdigit(os.getenv('DEVICE_ID')):
    DEVICE_ID= int(os.getenv('DEVICE_ID'))


def all_reduce_mean(tensor: torch.Tensor, plugin: Plugin = None) -> torch.Tensor:
    if plugin is not None:
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, group=plugin.dp_group)
        tensor.div_(plugin.dp_size)
    else:
        dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


def get_model_numel(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

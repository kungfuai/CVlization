from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def clip_grad_agc_(
        parameters: _tensor_or_tensors,
        clip: float,
        pmin: float,
        foreach: Optional[bool] = None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    params = []
    grads = []
    for p in parameters:
        if p.grad is not None:
            params.append(p)
            grads.append(p.grad)

    if len(grads) == 0:
        return
    grouped: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [params, grads]
    )  # type: ignore[assignment]

    for (device, _), ([device_params, device_grads], _) in grouped.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            pnorm = torch._foreach_norm(device_params, ord=2)
            gnorm = torch._foreach_norm(device_grads, ord=2)
            upper = torch._foreach_mul(torch._foreach_maximum(pnorm, pmin), clip)
            scale = torch._foreach_reciprocal(torch._foreach_maximum(torch._foreach_div(gnorm, upper), 1.0))
            torch._foreach_mul_(device_grads, scale)
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            for p, g in zip(device_params, device_grads):
                pnorm = torch.norm(p, p=2)
                gnorm = torch.norm(g, p=2)
                upper = torch.tensor(clip) * torch.maximum(torch.tensor(pmin), pnorm)
                scale = 1 / torch.maximum(torch.tensor(1.0), gnorm / upper)
                g.detach().mul_(scale)

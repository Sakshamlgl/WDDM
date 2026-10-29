"""
Multi-GPU forward pass that keeps both domains on every GPU.

nn.DataParallel splits the batch into contiguous chunks, so a
[source; target] batch would give GPU 0 only source images and GPU 1 only
target images, and each replica's BatchNorm would normalize a single domain.
This wrapper splits EACH input separately (e.g. 16 source + 16 target ->
8 + 8 per GPU), runs the replicas in parallel and returns the outputs on the
module's device in the original order.

Otherwise it behaves like nn.DataParallel: parameters are broadcast on every
forward, gradients are summed back into the original module, and BatchNorm
running statistics are updated from the first GPU's replica only. BatchNorm
batch statistics are computed per GPU (half the batch each), not synchronized.

With device_ids=None (or a single id) it just runs the module on its own
device, so callers can use it unconditionally.
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.parallel import parallel_apply, replicate


class DomainBalancedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, device_ids: Optional[Sequence[int]] = None):
        super().__init__()
        self.module = module
        self.device_ids = list(device_ids) if device_ids and len(device_ids) > 1 else []

    def forward(self, *parts: torch.Tensor):
        device = next(self.module.parameters()).device
        sizes = [p.size(0) for p in parts]
        if not self.device_ids:
            out = self.module(torch.cat([p.to(device, non_blocking=True) for p in parts]))
            outs = out.split(sizes)
            return outs[0] if len(outs) == 1 else tuple(outs)

        n = len(self.device_ids)
        splits = [p.tensor_split(n) for p in parts]
        inputs, chunk_sizes, devices = [], [], []
        for i, dev in enumerate(self.device_ids):
            pieces = [s[i] for s in splits]
            cs = [x.size(0) for x in pieces]
            if sum(cs) == 0:
                continue
            target = torch.device("cuda", dev)
            inputs.append((torch.cat([x.to(target, non_blocking=True) for x in pieces]),))
            chunk_sizes.append(cs)
            devices.append(dev)

        replicas = replicate(self.module, devices, not torch.is_grad_enabled())
        outputs = parallel_apply(replicas, inputs, devices=devices)
        results = [[] for _ in parts]
        for out, cs in zip(outputs, chunk_sizes):
            for j, piece in enumerate(out.split(cs)):
                results[j].append(piece.to(device))
        outs = [torch.cat(r) for r in results]
        return outs[0] if len(outs) == 1 else tuple(outs)

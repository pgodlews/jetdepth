"""Upstream's streaming step with its temporal memory as explicit inputs and outputs.

TensorRT engines hold no state between runs, so each step receives the previous memory and returns
the new memory. Import this module only after ``dp2trt.upstream.use_upstream``.
"""
from __future__ import annotations

import torch
from demo.export.stream import CausalStep


class StatefulStep(CausalStep):
    """``CausalStep`` whose caches are passed in and returned instead of stored in buffers."""

    def forward(self, frame: torch.Tensor, *state: torch.Tensor):
        self.state = dict(zip(self.cache_names, state))
        self.next_state = {}
        depth = super().forward(frame)
        return (depth, *[self.next_state[name] for name in self.cache_names])

    def _push(self, name: str, latest: torch.Tensor, dim: int = 0) -> torch.Tensor:
        cache = self.state[name]
        size = latest.shape[dim]
        updated = torch.cat([cache.narrow(dim, size, cache.shape[dim] - size), latest], dim=dim)
        self.next_state[name] = updated
        return updated

    def initial_state(self) -> list[torch.Tensor]:
        return [torch.zeros_like(getattr(self, name)) for name in self.cache_names]

    def io_names(self) -> tuple[list[str], list[str]]:
        inputs = ["frame"] + [f"state_{name}" for name in self.cache_names]
        outputs = ["inverse_depth"] + [f"next_{name}" for name in self.cache_names]
        return inputs, outputs

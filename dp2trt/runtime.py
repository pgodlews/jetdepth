"""Minimal TensorRT runtime for single-frame and temporal Depth Pro 2 engines."""
from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch


class DepthEngine:
    """Run a compiled engine, keeping the temporal memory between calls.

    The engine reads and writes raw GPU memory. Each step reads the memory in ``current`` and
    writes the updated memory to ``spare``; the two sets are then swapped, without copying.
    The tensor returned by ``execute`` is reused by the next step: copy it to keep it.
    """

    def __init__(self, path: str | Path):
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not load the engine")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.frame = self._allocate("frame")
        self.depth = self._allocate("inverse_depth")
        self.context.set_tensor_address("frame", int(self.frame.data_ptr()))
        self.context.set_tensor_address("inverse_depth", int(self.depth.data_ptr()))
        self.keys = [name[len("state_"):] for name in names if name.startswith("state_")]
        self.current = {key: self._allocate(f"state_{key}") for key in self.keys}
        self.spare = {key: self._allocate(f"next_{key}") for key in self.keys}
        self.stream = torch.cuda.Stream()
        self.height, self.width = tuple(self.frame.shape[-2:])

    def _allocate(self, name: str) -> torch.Tensor:
        dtype = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}[
            self.engine.get_tensor_dtype(name)
        ]
        return torch.zeros(tuple(self.engine.get_tensor_shape(name)), dtype=dtype, device="cuda")

    @property
    def is_temporal(self) -> bool:
        return bool(self.keys)

    def reset(self) -> None:
        """Forget all previous frames. Call this when the video or camera changes."""
        for key in self.keys:
            self.current[key].zero_()
        torch.cuda.synchronize()

    def upload(self, frame: torch.Tensor) -> None:
        """Copy a contiguous ``(1, 3, H, W)`` float32 RGB frame in [0, 1] into the engine input."""
        if frame.shape != self.frame.shape:
            raise ValueError(f"expected {tuple(self.frame.shape)}, got {tuple(frame.shape)}")
        if not frame.is_contiguous():
            raise ValueError("TensorRT input must be contiguous; call .contiguous() first")
        self.frame.copy_(frame)
        torch.cuda.synchronize()

    def execute(self) -> torch.Tensor:
        """Run one step on the uploaded frame and return inverse depth ``(1, H, W)``."""
        for key in self.keys:
            self.context.set_tensor_address(f"state_{key}", int(self.current[key].data_ptr()))
            self.context.set_tensor_address(f"next_{key}", int(self.spare[key].data_ptr()))
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        self.current, self.spare = self.spare, self.current
        return self.depth

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        self.upload(frame)
        return self.execute()

"""Convert Depth Pro 2 to TensorRT on NVIDIA Jetson Orin, and verify the result."""

PINNED_COMMIT = "aa389cbb0b74341664ed663a0966c859905cd6c0"
MODELS = {"s": "depthpro2-s", "b": "depthpro2-b", "l": "depthpro2-l", "h": "depthpro2-h"}

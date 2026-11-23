"""Exact ONNX graph rewrites needed to compile the temporal model with TensorRT."""
from __future__ import annotations

import onnx
from onnx import helper, numpy_helper
import numpy as np


def _concat_axis(node: onnx.NodeProto) -> int | None:
    for attribute in node.attribute:
        if attribute.name == "axis":
            return helper.get_attribute_value(attribute)
    return None


def split_cross_frame_matmul(model: onnx.ModelProto, prefix: int, patches: int, window: int) -> int:
    """Replace each cross-frame ``weights @ values`` with one product per slot, then a sum.

    ``W @ concat(V_0, ..., V_n) == sum(W[..., slot_i] @ V_i)``, so the result is unchanged. TensorRT
    10.3 segfaults while compiling the single large product with its FP16 flag on.
    Returns the number of products rewritten.
    """
    graph = model.graph
    producers = {out: node for node in graph.node for out in node.output}
    sizes = [prefix] + [patches] * window
    total = sum(sizes)
    sizes_name = "dp2trt_slot_sizes"
    rewritten, nodes = 0, []
    for node in graph.node:
        first = producers.get(node.input[0]) if node.op_type == "MatMul" else None
        second = producers.get(node.input[1]) if node.op_type == "MatMul" else None
        matches = (first is not None and second is not None and first.op_type == "Softmax"
                   and second.op_type == "Concat" and _concat_axis(second) in (2, -2))
        if not matches:
            nodes.append(node)
            continue
        tag = f"dp2trt_slot{rewritten}"
        weight_parts = [f"{tag}_w{i}" for i in range(len(sizes))]
        value_parts = [f"{tag}_v{i}" for i in range(len(sizes))]
        nodes.append(helper.make_node("Split", [node.input[0], sizes_name], weight_parts, axis=-1, name=f"{tag}_split_w"))
        nodes.append(helper.make_node("Split", [node.input[1], sizes_name], value_parts, axis=2, name=f"{tag}_split_v"))
        products = []
        for i, (w, v) in enumerate(zip(weight_parts, value_parts)):
            products.append(f"{tag}_p{i}")
            nodes.append(helper.make_node("MatMul", [w, v], [products[-1]], name=f"{tag}_matmul{i}"))
        running = products[0]
        for i, product in enumerate(products[1:], 1):
            out = node.output[0] if i == len(products) - 1 else f"{tag}_sum{i}"
            nodes.append(helper.make_node("Add", [running, product], [out], name=f"{tag}_add{i}"))
            running = out
        rewritten += 1
    if rewritten:
        graph.initializer.append(numpy_helper.from_array(np.array(sizes, dtype=np.int64), sizes_name))
        del graph.node[:]
        graph.node.extend(nodes)
    if total <= 0:
        raise ValueError("invalid slot sizes")
    return rewritten

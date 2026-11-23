#!/usr/bin/env python3
"""Convert Depth Pro 2 to a verified TensorRT engine.

Stages run in separate processes, so memory is returned between them and a build killed by the
operating system is reported with its exit status and peak memory. The engine keeps the suffix
``.unverified`` until it matches PyTorch within the stated limits, with no NaN or Inf.
Exit status: 0 verified, 2 failed verification, 3 a stage failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dp2trt import MODELS, PINNED_COMMIT
from dp2trt.metrics import Thresholds, compare, gate
from dp2trt.telemetry import mem_available_mb


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="s")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--window", type=int, choices=[1, 4], default=1)
    ap.add_argument("--precision", choices=["fp16", "fp32"],
                    help="S: fp16 (default) or fp32. B, L and H+: fp32 only")
    ap.add_argument("--upstream", default="work/ml-depth-pro-2", help="pinned upstream checkout")
    ap.add_argument("--out", default="work/engines")
    ap.add_argument("--device", default="cuda", help="device for the PyTorch reference and 16-bit fit")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify-frames", type=int, default=16)
    ap.add_argument("--workspace-gb", type=float, default=3.0, help="TensorRT build workspace limit")
    ap.add_argument("--max-median-rel", type=float, default=Thresholds.max_median_rel)
    ap.add_argument("--max-p99-rel", type=float, default=Thresholds.max_p99_rel)
    ap.add_argument("--min-correlation", type=float, default=Thresholds.min_correlation)
    ap.add_argument("--skip-onnx-check", action="store_true", help="skip the ONNX Runtime CPU check")
    ap.add_argument("--trt-verbose", action="store_true", help="write verbose TensorRT logs")
    ap.add_argument("--force", action="store_true", help="redo stages whose outputs exist")
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--stage", choices=["reference", "export", "build", "verify"], help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.precision is None:
        a.precision = "fp16" if a.model == "s" else "fp32"
    if a.precision == "fp16" and a.model != "s":
        ap.error("16-bit B, L and H+ engines failed verification; use --precision fp32")
    return a


class Paths:
    def __init__(self, a: argparse.Namespace):
        self.stem = f"{a.model}_{a.width}x{a.height}_w{a.window}_{a.precision}"
        self.out = Path(a.out).expanduser().resolve()
        self.refs = self.out / f"{self.stem}.refs.npz"
        self.onnx_dir = self.out / f"{self.stem}.onnx.d"
        self.onnx = self.onnx_dir / "model.onnx"
        self.unverified = self.out / f"{self.stem}.engine.unverified"
        self.engine = self.out / f"{self.stem}.engine"
        self.report = self.out / f"{self.stem}.report.json"
        self.log = self.out / f"{self.stem}.log"

    def stage_result(self, stage: str) -> Path:
        return self.out / f"{self.stem}.{stage}.json"


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def stage_reference(a: argparse.Namespace, p: Paths) -> None:
    import numpy as np
    import torch

    from dp2trt import frames
    from dp2trt.upstream import use_upstream

    root = use_upstream(a.upstream, a.allow_unpinned)
    from depth_pro_2 import create_model_and_transforms

    model, transform = create_model_and_transforms(MODELS[a.model], device=a.device, precision=torch.float32)

    def run(sequence):
        # f_px only affects metric depth; inverse_depth is focal-normalised and does not use it.
        stream = model.stream(window=a.window, overlap=0, f_px=float(a.width), mode="unbuffered")
        with torch.inference_mode():
            return [stream.step(transform(rgb))[-1]["inverse_depth"].float().cpu().numpy()[None] for rgb in sequence]

    natural = run(frames.eval_frames(root, a.width, a.height, a.verify_frames))
    stress = run(frames.stress_frames(a.width, a.height))
    np.savez(p.refs, natural=np.stack(natural), stress=np.stack(stress))
    write_json(p.stage_result("reference"), {"natural_frames": len(natural), "stress_frames": len(stress)})


def stage_export(a: argparse.Namespace, p: Paths) -> None:
    import numpy as np
    import onnx
    import torch

    from dp2trt import frames
    from dp2trt.upstream import use_upstream

    root = use_upstream(a.upstream, a.allow_unpinned)
    from demo.export.fp16 import FP16_TARGET_PEAK, fit_attention_fp16, fit_fp16
    from depth_pro_2 import create_model_and_transforms

    from dp2trt.stateful import StatefulStep

    model, _ = create_model_and_transforms(MODELS[a.model], device=a.device, precision=torch.float32)
    residual_divisor, logit_divisors = None, None
    if a.precision == "fp16":
        fit = torch.cat([frames.to_tensor(f, a.device) for f in frames.fit_frames(root, a.width, a.height)])
        residual_divisor = fit_fp16(model, fit, FP16_TARGET_PEAK)
        logit_divisors = fit_attention_fp16(model, fit, a.window, FP16_TARGET_PEAK)
        del fit
    step = StatefulStep(model, a.window, a.height, a.width, logit_divisors).to(a.device).eval()

    refs = np.load(p.refs)
    natural_rgb = frames.eval_frames(root, a.width, a.height, a.verify_frames)
    state, wrapped = step.initial_state(), []
    with torch.inference_mode():
        for rgb in natural_rgb:
            depth, *state = step(frames.to_tensor(rgb, a.device), *state)
            wrapped.append(depth.float().cpu().numpy())
    wrapper = compare(list(refs["natural"]), wrapped)
    del state, wrapped

    step = step.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inputs, outputs = step.io_names()
    if p.onnx_dir.exists():
        shutil.rmtree(p.onnx_dir)
    p.onnx_dir.mkdir(parents=True)
    dummy = (torch.rand(1, 3, a.height, a.width), *step.initial_state())
    with torch.no_grad():
        torch.onnx.export(step, dummy, str(p.onnx), input_names=inputs, output_names=outputs,
                          opset_version=a.opset, do_constant_folding=True, dynamo=False)
    rewritten = 0
    if a.precision == "fp16" and a.window > 1:
        from dp2trt.onnx_rewrite import split_cross_frame_matmul

        # Only the graph changes, so weights stored in external files stay where they are.
        graph = onnx.load(str(p.onnx), load_external_data=False)
        rewritten = split_cross_frame_matmul(graph, model.encoder.num_prefix_tokens, step.num_patches, a.window)
        if rewritten != len(model.encoder.cross_frame_blocks):
            raise SystemExit(f"Rewrote {rewritten} attention products, expected {len(model.encoder.cross_frame_blocks)}")
        temporary = p.onnx.with_name("model.onnx.rewritten")
        temporary.write_bytes(graph.SerializeToString())
        temporary.replace(p.onnx)
        del graph
    onnx.checker.check_model(str(p.onnx), full_check=False)

    runtime_check = None
    if not a.skip_onnx_check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(p.onnx), providers=["CPUExecutionProvider"])
        ort_state, ort_out = [s.numpy() for s in step.initial_state()], []
        count = min(6, len(natural_rgb))
        for rgb in natural_rgb[:count]:
            result = session.run(None, dict(zip(inputs, [frames.to_tensor(rgb).numpy(), *ort_state])))
            ort_out.append(result[0])
            ort_state = result[1:]
        runtime_check = compare(list(refs["natural"][:count]), ort_out)
    size = sum(f.stat().st_size for f in p.onnx_dir.iterdir())
    write_json(p.stage_result("export"), {
        "residual_divisor": residual_divisor,
        "attention_divisors": {str(k): v for k, v in (logit_divisors or {}).items()},
        "memory_tensors": {n: list(s.shape) for n, s in zip(step.cache_names, step.initial_state())},
        "explicit_memory_pytorch_vs_reference": wrapper,
        "onnx_runtime_vs_reference": runtime_check,
        "attention_products_split_for_fp16": rewritten,
        "onnx_mb": size / 2**20,
    })


def stage_build(a: argparse.Namespace, p: Paths) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if a.trt_verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(p.onnx)):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i), flush=True)
        raise SystemExit(3)
    if a.precision == "fp16":
        for i in range(network.num_inputs):
            if network.get_input(i).name != "frame":
                network.get_input(i).dtype = trt.DataType.HALF
        for i in range(network.num_outputs):
            if network.get_output(i).name != "inverse_depth":
                network.get_output(i).dtype = trt.DataType.HALF
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * 2**30))
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(tensor.shape)
        profile.set_shape(tensor.name, shape, shape, shape)
    config.add_optimization_profile(profile)
    if a.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("TensorRT returned no engine", flush=True)
        raise SystemExit(3)
    p.unverified.write_bytes(bytes(serialized))
    write_json(p.stage_result("build"), {"engine_mb": p.unverified.stat().st_size / 2**20})


def stage_verify(a: argparse.Namespace, p: Paths) -> None:
    import numpy as np
    import torch

    from dp2trt import frames
    from dp2trt.runtime import DepthEngine
    from dp2trt.upstream import use_upstream

    root = use_upstream(a.upstream, a.allow_unpinned)
    refs = np.load(p.refs)
    engine = DepthEngine(p.unverified)
    natural_gpu = [frames.to_tensor(f, "cuda") for f in frames.eval_frames(root, a.width, a.height, a.verify_frames)]
    stress_gpu = [frames.to_tensor(f, "cuda") for f in frames.stress_frames(a.width, a.height)]

    def run(sequence):
        engine.reset()
        return [engine(f).cpu().numpy().copy() for f in sequence]

    with torch.inference_mode():
        first = run(natural_gpu)
        second = run(natural_gpu)
        stress = run(stress_gpu)
    repeat = max(float(np.nanmax(np.abs(x - y))) for x, y in zip(first, second))
    write_json(p.stage_result("verify"), {
        "natural": compare(list(refs["natural"]), first),
        "stress": compare(list(refs["stress"]), stress),
        "repeat_after_reset_max_abs": repeat,
    })


STAGES = {"reference": stage_reference, "export": stage_export, "build": stage_build, "verify": stage_verify}


def run_stage(name: str, p: Paths) -> dict:
    """Run one stage in a child process, logging everything and tracking its memory use."""
    forwarded = [x for x in sys.argv[1:]]
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", name, *forwarded]
    if shutil.which("stdbuf"):
        command = ["stdbuf", "-oL", "-eL", *command]
    start = time.time()
    floor = [mem_available_mb() or 0.0]
    done = threading.Event()

    def watch():
        while not done.wait(0.5):
            value = mem_available_mb()
            if value is not None:
                floor[0] = min(floor[0], value)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    with p.log.open("ab") as log:
        log.write(f"\n===== stage {name} =====\n".encode())
        log.flush()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(child.pid, 0)
    child.returncode = os.waitstatus_to_exitcode(status)
    done.set()
    watcher.join(timeout=2)
    code = child.returncode
    info = {"exit_status": code, "seconds": round(time.time() - start, 1),
            "peak_rss_mb": round(usage.ru_maxrss / 1024, 1), "min_mem_available_mb": round(floor[0], 1)}
    if code < 0:
        info["signal"] = signal.Signals(-code).name
    return info


def main() -> int:
    a = parse_args()
    p = Paths(a)
    p.out.mkdir(parents=True, exist_ok=True)
    if a.stage:
        STAGES[a.stage](a, p)
        return 0

    limits = Thresholds(a.max_median_rel, a.max_p99_rel, a.min_correlation)
    report = {"name": p.stem, "model": MODELS[a.model], "width": a.width, "height": a.height,
              "window": a.window, "precision": a.precision, "upstream_commit": PINNED_COMMIT,
              "thresholds": limits.as_dict(), "stages": {}, "passed": False, "failures": []}
    if a.force:
        p.engine.unlink(missing_ok=True)
        p.unverified.unlink(missing_ok=True)
    elif p.engine.exists():
        p.engine.replace(p.unverified)  # An existing engine is verified again before it is trusted.
    needed = {"reference": p.refs, "export": p.onnx, "build": p.unverified, "verify": None}
    for name, output in needed.items():
        if output is not None and output.exists() and not a.force:
            print(f"[{p.stem}] {name}: reusing existing output", flush=True)
        else:
            print(f"[{p.stem}] {name}: running", flush=True)
            info = run_stage(name, p)
            report["stages"][name] = info
            print(f"[{p.stem}] {name}: {info}", flush=True)
            if info["exit_status"] != 0:
                reason = info.get("signal", f"exit status {info['exit_status']}")
                report["failures"].append(f"stage {name} failed ({reason}); see {p.log.name}")
                write_json(p.report, report)
                print(f"[{p.stem}] FAILED: {report['failures'][-1]}", flush=True)
                return 3
        result = p.stage_result(name)
        if result.exists():
            report["stages"].setdefault(name, {})["result"] = json.loads(result.read_text())

    verify = report["stages"]["verify"]["result"]
    report["failures"] = gate(verify["natural"], verify["stress"], limits)
    if a.window > 1 and verify["repeat_after_reset_max_abs"] > 1e-3:
        report["failures"].append("output changed after resetting the temporal memory")
    report["passed"] = not report["failures"]
    if report["passed"]:
        digest = hashlib.sha256(p.unverified.read_bytes()).hexdigest()
        report["engine_sha256"] = digest
        p.unverified.replace(p.engine)
        print(f"[{p.stem}] VERIFIED: {verify['natural']}", flush=True)
    else:
        print(f"[{p.stem}] NOT VERIFIED: {report['failures']}", flush=True)
    write_json(p.report, report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

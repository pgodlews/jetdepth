#!/usr/bin/env python3
"""Benchmark a verified engine and write one machine-readable JSON result.

Timing covers the model only: TensorRT execution on a frame already in GPU memory. Camera, resize,
colour conversion and display are excluded. The script only reads device state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from dp2trt import PINNED_COMMIT, frames, telemetry
from dp2trt.runtime import DepthEngine
from dp2trt.upstream import use_upstream

BUSY_GPU_PERCENT = 5.0
BUSY_OTHER_CPU_SECONDS = 1.0


def summarize(values: list[float]) -> dict | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {"median": float(np.median(clean)), "p95": float(np.percentile(clean, 95)),
            "min": float(min(clean)), "max": float(max(clean))}


def summarize_samples(samples: list[dict]) -> dict:
    rails = sorted({k for s in samples for k in s["power"]})
    zones = sorted({k for s in samples for k in s["temps"]})
    return {
        "samples": len(samples),
        "gpu_load_percent": summarize([s["gpu_load"] for s in samples]),
        "gpu_freq_mhz": summarize([s["gpu_mhz"] for s in samples]),
        "power_w": {r: summarize([s["power"].get(r) for s in samples]) for r in rails},
        "temperature_c": {z: summarize([s["temps"].get(z) for s in samples]) for z in zones},
        "mem_available_mb": summarize([s["mem_available_mb"] for s in samples]),
    }


def sample_for(seconds: float) -> list[dict]:
    sampler = telemetry.Sampler()
    sampler.start()
    time.sleep(seconds)
    return sampler.stop()


def versions() -> dict:
    import onnx
    import onnxruntime

    return {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__, "numpy": np.__version__, "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__, "l4t": telemetry.l4t_release(),
            "upstream_commit": PINNED_COMMIT}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="verified .engine with its .report.json beside it")
    ap.add_argument("--device-label", required=True, help='for example "Jetson Orin Nano Super 8 GB"')
    ap.add_argument("--upstream", default="work/ml-depth-pro-2")
    ap.add_argument("--output", required=True, help="result JSON path")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--idle-seconds", type=float, default=5.0)
    ap.add_argument("--watch-process", action="append", default=[],
                    help="command name of another GPU client; its CPU use marks the run as contended")
    ap.add_argument("--allow-unpinned", action="store_true")
    a = ap.parse_args()

    engine_path = Path(a.engine).expanduser().resolve()
    report_path = engine_path.with_suffix(".report.json")
    report = json.loads(report_path.read_text())
    if not report.get("passed"):
        raise SystemExit("Refusing to benchmark an engine that did not pass verification.")
    digest = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if digest != report.get("engine_sha256"):
        raise SystemExit("Engine does not match its verification report.")
    root = use_upstream(a.upstream, a.allow_unpinned)

    idle = sample_for(a.idle_seconds)
    mem_before = telemetry.mem_available_mb()
    engine = DepthEngine(engine_path)
    clip = [frames.to_tensor(f, "cuda") for f in frames.eval_frames(root, engine.width, engine.height, 16)]
    torch.cuda.synchronize()
    mem_loaded = telemetry.mem_available_mb()

    with torch.inference_mode():
        for i in range(a.warmup):
            engine(clip[i % len(clip)])
        watched_before = telemetry.process_cpu_seconds(a.watch_process) if a.watch_process else 0.0
        sampler = telemetry.Sampler()
        sampler.start()
        times, count, start = [], 0, time.perf_counter()
        while time.perf_counter() - start < a.seconds:
            engine.upload(clip[count % len(clip)])
            begin = time.perf_counter()
            engine.execute()
            times.append((time.perf_counter() - begin) * 1000)
            count += 1
        elapsed = time.perf_counter() - start
        run_samples = sampler.stop()
    watched_cpu = (telemetry.process_cpu_seconds(a.watch_process) - watched_before) if a.watch_process else 0.0

    idle_summary, run_summary = summarize_samples(idle), summarize_samples(run_samples)
    idle_load = (idle_summary["gpu_load_percent"] or {}).get("median")
    reasons = []
    if idle_load is not None and idle_load > BUSY_GPU_PERCENT:
        reasons.append(f"GPU load {idle_load:.1f}% before the run")
    if watched_cpu > BUSY_OTHER_CPU_SECONDS:
        reasons.append(f"watched processes used {watched_cpu:.1f} CPU seconds during the run")

    median_ms = statistics.median(times)
    result = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": a.device_label,
        "device_tree_model": telemetry.device_model(),
        "power_mode": telemetry.power_mode(),
        "model": report["model"], "width": report["width"], "height": report["height"],
        "window": report["window"], "precision": report["precision"],
        "engine_sha256": digest, "engine_mb": engine_path.stat().st_size / 2**20,
        "frames_source": "upstream assets/example.mp4, frames 32-47, resized",
        "timing": {
            "scope": "TensorRT execution only, frame already in GPU memory",
            "frames": count, "seconds": elapsed, "warmup_frames": a.warmup,
            "latency_ms_median": median_ms, "latency_ms_p95": float(np.percentile(times, 95)),
            "latency_ms_p99": float(np.percentile(times, 99)),
            "fps_from_median_latency": 1000 / median_ms,
            "fps_from_p95_latency": 1000 / float(np.percentile(times, 95)),
            "fps_sustained_including_upload": count / elapsed,
        },
        "memory": {
            "note": "Orin GPU memory is shared with the CPU; nvgpu reports no per-process GPU memory.",
            "system_mem_drop_on_load_mb": (mem_before - mem_loaded) if mem_before and mem_loaded else None,
            "process_peak_rss_mb": telemetry.process_peak_rss_mb(),
            "torch_allocated_mb": torch.cuda.memory_allocated() / 2**20,
            "system_mem_available_min_during_run_mb": (run_summary["mem_available_mb"] or {}).get("min"),
        },
        "idle": idle_summary,
        "during_run": run_summary,
        "contention": {"gpu_busy_or_shared": bool(reasons), "reasons": reasons,
                       "watched_process_count": len(a.watch_process), "watched_cpu_seconds": watched_cpu},
        "accuracy_vs_pytorch": report["stages"]["verify"]["result"],
        "thresholds": report["thresholds"],
        "versions": versions(),
    }
    output = Path(a.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    timing = result["timing"]
    print(f"{report['name']}: {timing['fps_from_median_latency']:.1f} FPS, median {median_ms:.1f} ms, "
          f"p95 {timing['latency_ms_p95']:.1f} ms, contended={bool(reasons)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

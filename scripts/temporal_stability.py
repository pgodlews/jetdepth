#!/usr/bin/env python3
"""Compare frame-to-frame stability of a single-frame and a temporal engine on the sample video.

Both engines see the same consecutive frames. The camera moves in this video, so part of each
change is real. The figures are useful for comparing the two engines, not as absolute flicker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp2trt import PINNED_COMMIT, frames  # noqa: E402
from dp2trt.metrics import REL_EPS  # noqa: E402
from dp2trt.runtime import DepthEngine  # noqa: E402
from dp2trt.upstream import use_upstream  # noqa: E402


def load_report(engine: Path) -> dict:
    report = json.loads(engine.with_suffix(".report.json").read_text())
    if not report.get("passed"):
        raise SystemExit(f"{engine.name} did not pass verification.")
    return report


def run_engine(path: Path, clip: list[np.ndarray]) -> list[np.ndarray]:
    engine = DepthEngine(path)
    engine.reset()
    with torch.inference_mode():
        outputs = [engine(frames.to_tensor(rgb, "cuda"))[0].cpu().numpy().copy() for rgb in clip]
    del engine
    torch.cuda.empty_cache()
    return outputs


def stability(outputs: list[np.ndarray], warmup: int) -> dict:
    kept = outputs[warmup:]
    if any(not np.isfinite(o).all() for o in kept):
        raise SystemExit("Non-finite output; stability is not meaningful.")
    changes, scale_jumps = [], []
    for previous, current in zip(kept, kept[1:]):
        changes.append((np.abs(current - previous) / (np.abs(previous) + REL_EPS)).ravel())
        scale_jumps.append(abs(float(np.median(current)) / float(np.median(previous)) - 1))
    pooled = np.concatenate(changes)
    return {
        "frame_pairs": len(scale_jumps),
        "relative_change_median": float(np.median(pooled)),
        "relative_change_p95": float(np.percentile(pooled, 95)),
        "scale_jump_median": float(np.median(scale_jumps)),
        "scale_jump_p95": float(np.percentile(scale_jumps, 95)),
        "scale_jump_max": float(np.max(scale_jumps)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-w1", required=True)
    ap.add_argument("--engine-w4", required=True)
    ap.add_argument("--device-label", required=True)
    ap.add_argument("--upstream", default="work/ml-depth-pro-2")
    ap.add_argument("--output", required=True)
    ap.add_argument("--warmup", type=int, default=8, help="initial frames left out of the statistics")
    ap.add_argument("--allow-unpinned", action="store_true")
    a = ap.parse_args()

    path_w1, path_w4 = Path(a.engine_w1).expanduser().resolve(), Path(a.engine_w4).expanduser().resolve()
    report_w1, report_w4 = load_report(path_w1), load_report(path_w4)
    for key in ("model", "width", "height", "precision"):
        if report_w1[key] != report_w4[key]:
            raise SystemExit(f"Engines differ in {key}.")
    if (report_w1["window"], report_w4["window"]) != (1, 4):
        raise SystemExit("Expected a window-1 engine and a window-4 engine.")
    root = use_upstream(a.upstream, a.allow_unpinned)
    clip = frames.consecutive_frames(root, report_w1["width"], report_w1["height"])

    w1, w4 = stability(run_engine(path_w1, clip), a.warmup), stability(run_engine(path_w4, clip), a.warmup)
    result = {
        "schema": 1, "device": a.device_label, "model": report_w1["model"], "width": report_w1["width"],
        "height": report_w1["height"], "precision": report_w1["precision"],
        "frames_source": f"upstream assets/example.mp4, {len(clip)} consecutive decoded frames, resized",
        "warmup_frames_excluded": a.warmup, "upstream_commit": PINNED_COMMIT,
        "definition": "relative change = |d[t] - d[t-1]| / (|d[t-1]| + 0.001), pooled over pixels and frame pairs; "
                      "scale jump = |median(d[t]) / median(d[t-1]) - 1|; d is inverse depth",
        "caveat": "The camera moves, so these include real scene change.",
        "window_1": w1, "window_4": w4,
        "window_1_over_window_4": {k: w1[k] / w4[k] for k in w1 if k != "frame_pairs" and w4[k] > 0},
    }
    output = Path(a.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"size": f"{result['width']}x{result['height']}", "w1": w1, "w4": w4}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

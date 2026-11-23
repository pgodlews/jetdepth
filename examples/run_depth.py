#!/usr/bin/env python3
"""Minimal runtime example: run a verified engine over a video file.

The output is relative-looking inverse depth. Converting it to metres needs the camera's focal
length, and even then the result is a model estimate rather than a calibrated measurement.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dp2trt.frames import to_tensor  # noqa: E402
from dp2trt.runtime import DepthEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--f-px", type=float, help="focal length in pixels of the resized frame")
    a = ap.parse_args()

    engine = DepthEngine(a.engine)
    engine.reset()  # Start with empty temporal memory. Call again whenever the video changes.
    video = cv2.VideoCapture(a.video)
    with torch.inference_mode():
        for index in range(a.frames):
            ok, bgr = video.read()
            if not ok:
                break
            small = cv2.resize(bgr, (engine.width, engine.height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            # to_tensor returns contiguous memory; TensorRT would misread a permuted view.
            inverse_depth = engine(to_tensor(rgb, "cuda"))[0].cpu().numpy().copy()
            line = f"frame {index}: inverse depth median {np.median(inverse_depth):.4f}"
            if a.f_px:
                depth = a.f_px / (engine.width * np.clip(inverse_depth, 1e-6, None))
                line += f", estimated depth median {np.median(depth):.2f} m (uncalibrated)"
            print(line)
    video.release()


if __name__ == "__main__":
    main()

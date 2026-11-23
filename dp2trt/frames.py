"""Test frames: Apple's public sample video from the upstream checkout, and synthetic patterns."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# Fitting and scoring use separate segments of the sample video.
FIT_INDICES = tuple(range(0, 24, 3))
EVAL_START = 32


def _read_video(upstream: Path, wanted: set[int], width: int, height: int) -> dict[int, np.ndarray]:
    path = upstream / "assets" / "example.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path.name} in the upstream checkout")
    out, index = {}, 0
    last = max(wanted)
    while index <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if index in wanted:
            small = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            out[index] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        index += 1
    cap.release()
    missing = wanted - set(out)
    if missing:
        raise SystemExit(f"Sample video ended early; missing frames {sorted(missing)[:5]}")
    return out


def fit_frames(upstream: Path, width: int, height: int) -> list[np.ndarray]:
    """RGB uint8 frames used only to fit the 16-bit rescaling."""
    frames = _read_video(upstream, set(FIT_INDICES), width, height)
    return [frames[i] for i in FIT_INDICES]


def eval_frames(upstream: Path, width: int, height: int, count: int) -> list[np.ndarray]:
    """Consecutive RGB uint8 frames used only for scoring, so temporal memory is exercised."""
    indices = list(range(EVAL_START, EVAL_START + count))
    frames = _read_video(upstream, set(indices), width, height)
    return [frames[i] for i in indices]


def consecutive_frames(upstream: Path, width: int, height: int) -> list[np.ndarray]:
    """Every RGB uint8 frame the decoder returns from the sample video, in order."""
    cap = cv2.VideoCapture(str(upstream / "assets" / "example.mp4"))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        small = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(out) < 32:
        raise SystemExit(f"Sample video gave only {len(out)} frames")
    return out


def stress_frames(width: int, height: int) -> list[np.ndarray]:
    """Synthetic extremes, used to catch 16-bit overflow on inputs unlike the fitting frames."""
    y, x = np.mgrid[0:height, 0:width]
    black = np.zeros((height, width, 3), np.uint8)
    white = np.full((height, width, 3), 255, np.uint8)
    checker = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
    gradient = (x * 255 // max(width - 1, 1)).astype(np.uint8)
    noise = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return [black, white, np.dstack([checker] * 3), np.dstack([gradient] * 3), noise]


def to_tensor(rgb: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    """(1, 3, H, W) float32 in [0, 1]. Contiguous, because TensorRT reads raw memory."""
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255
    tensor = torch.from_numpy(np.ascontiguousarray(chw))[None].to(device)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor

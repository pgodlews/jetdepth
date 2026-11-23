"""Comparison metrics and the pass/fail gate. Depends only on NumPy so it can be unit-tested."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

REL_EPS = 1e-3


@dataclass(frozen=True)
class Thresholds:
    """Limits an engine must meet against the PyTorch float32 reference."""

    max_median_rel: float = 0.01
    max_p99_rel: float = 0.05
    min_correlation: float = 0.999

    def as_dict(self) -> dict:
        return asdict(self)


def compare(references: list[np.ndarray], outputs: list[np.ndarray]) -> dict:
    """Per-pixel relative difference over all frames, plus the worst frame's figures."""
    if len(references) != len(outputs) or not references:
        raise ValueError("references and outputs must be non-empty and the same length")
    nonfinite = sum(int((~np.isfinite(o)).sum()) for o in outputs)
    pixels = sum(int(o.size) for o in outputs)
    if nonfinite:
        return {"frames": len(outputs), "pixels": pixels, "nonfinite": nonfinite,
                "median_rel": None, "p99_rel": None, "worst_frame_p99_rel": None,
                "min_correlation": None}
    rels, corrs, frame_p99 = [], [], []
    for ref, out in zip(references, outputs):
        if ref.shape != out.shape:
            raise ValueError(f"shape mismatch: {ref.shape} vs {out.shape}")
        rel = np.abs(out - ref) / (np.abs(ref) + REL_EPS)
        rels.append(rel.ravel())
        frame_p99.append(float(np.percentile(rel, 99)))
        same = np.array_equal(ref, out)
        flat = np.ptp(ref) == 0 or np.ptp(out) == 0
        corrs.append(1.0 if same else 0.0 if flat else float(np.corrcoef(ref.ravel(), out.ravel())[0, 1]))
    joined = np.concatenate(rels)
    return {"frames": len(outputs), "pixels": pixels, "nonfinite": 0,
            "median_rel": float(np.median(joined)), "p99_rel": float(np.percentile(joined, 99)),
            "worst_frame_p99_rel": max(frame_p99), "min_correlation": min(corrs)}


def gate(natural: dict, stress: dict | None, limits: Thresholds) -> list[str]:
    """Return the reasons an engine fails. An empty list means it passes."""
    failures = []
    if natural["nonfinite"]:
        failures.append(f"{natural['nonfinite']} NaN/Inf values on natural frames")
    else:
        if natural["median_rel"] > limits.max_median_rel:
            failures.append(f"median difference {natural['median_rel']:.4%} > {limits.max_median_rel:.4%}")
        if natural["p99_rel"] > limits.max_p99_rel:
            failures.append(f"p99 difference {natural['p99_rel']:.4%} > {limits.max_p99_rel:.4%}")
        if natural["min_correlation"] < limits.min_correlation:
            failures.append(f"correlation {natural['min_correlation']:.6f} < {limits.min_correlation}")
    if stress is not None and stress["nonfinite"]:
        failures.append(f"{stress['nonfinite']} NaN/Inf values on synthetic stress frames")
    return failures

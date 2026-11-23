"""The verification gate must refuse bad engines. Needs only NumPy: python tests/test_gate.py"""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "metrics", Path(__file__).resolve().parents[1] / "dp2trt" / "metrics.py"
)
metrics = importlib.util.module_from_spec(spec)
import sys  # noqa: E402

sys.modules["metrics"] = metrics
spec.loader.exec_module(metrics)

LIMITS = metrics.Thresholds()
REF = [np.random.default_rng(1).uniform(0.1, 1.0, (1, 48, 64)).astype(np.float32) for _ in range(3)]


def failures(outputs, stress=None):
    return metrics.gate(metrics.compare(REF, outputs), stress, LIMITS)


def test_identical_passes():
    assert failures([r.copy() for r in REF]) == []


def test_small_difference_passes():
    assert failures([r * 1.002 for r in REF]) == []


def test_nan_fails():
    bad = [r.copy() for r in REF]
    bad[1][0, 3, 3] = np.nan
    assert any("NaN/Inf" in f for f in failures(bad))


def test_inf_fails():
    bad = [r.copy() for r in REF]
    bad[2][0, 0, 0] = np.inf
    assert any("NaN/Inf" in f for f in failures(bad))


def test_scale_error_fails():
    assert any("median" in f for f in failures([r * 1.05 for r in REF]))


def test_scrambled_input_fails():
    rng = np.random.default_rng(2)
    bad = [rng.permutation(r.ravel()).reshape(r.shape) for r in REF]
    assert failures(bad)


def test_stress_nan_fails():
    stress = metrics.compare([np.ones((1, 4, 4))], [np.full((1, 4, 4), np.nan)])
    assert any("stress" in f for f in failures([r.copy() for r in REF], stress))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")

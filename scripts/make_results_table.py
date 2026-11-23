#!/usr/bin/env python3
"""Build results/RESULTS.md from the JSON files, and update the table in README.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
START, END = "<!-- results:start -->", "<!-- results:end -->"
MODEL_ORDER = {"depthpro2-s": (0, "S"), "depthpro2-b": (1, "B"), "depthpro2-l": (2, "L"), "depthpro2-h": (3, "H+")}
HEADER = ("| Model | Input | Frames | Precision | FPS | Latency median / p95 (ms) | Difference from PyTorch, "
          "median / p99 | Power (W) | Peak GPU temp (C) | Engine (MB) | Status |\n"
          "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")


def power_text(run: dict) -> str:
    rails = run["during_run"]["power_w"]
    if "VDD_IN" in rails:
        return f"{rails['VDD_IN']['median']:.1f}"
    if "VDD_GPU_SOC" in rails:
        return f"{rails['VDD_GPU_SOC']['median']:.1f}*"
    return "n/a"


def row(run: dict) -> str:
    timing, accuracy = run["timing"], run["accuracy_vs_pytorch"]["natural"]
    temps = run["during_run"]["temperature_c"]
    gpu_temp = temps.get("gpu-thermal", {}).get("max")
    status = "verified" + (", GPU shared" if run["contention"]["gpu_busy_or_shared"] else "")
    return (f"| {MODEL_ORDER[run['model']][1]} | {run['width']}x{run['height']} | {run['window']} | {run['precision']} | "
            f"{timing['fps_from_median_latency']:.1f} | {timing['latency_ms_median']:.1f} / {timing['latency_ms_p95']:.1f} | "
            f"{accuracy['median_rel'] * 100:.3f}% / {accuracy['p99_rel'] * 100:.2f}% | {power_text(run)} | "
            f"{gpu_temp if gpu_temp is None else round(gpu_temp, 1)} | {run['engine_mb']:.0f} | {status} |")


def failed_row(report: dict) -> str:
    reason = "; ".join(report.get("failures", [])) or "not verified"
    return (f"| {MODEL_ORDER[report['model']][1]} | {report['width']}x{report['height']} | {report['window']} | "
            f"{report['precision']} | - | - | - | - | - | - | **failed:** {reason} |")


def sort_key(item: dict) -> tuple:
    return (MODEL_ORDER[item["model"]][0], item["width"], item["window"], item["precision"])


def device_section(folder: Path) -> str:
    runs = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))
            if not p.name.endswith(".conversion.json") and not p.name.startswith(("temporal_stability", "soak_"))]
    reports = [json.loads(p.read_text()) for p in sorted(folder.glob("*.conversion.json"))]
    done = {(r["model"], r["width"], r["height"], r["window"], r["precision"]) for r in runs}
    failed = [r for r in reports if (r["model"], r["width"], r["height"], r["window"], r["precision"]) not in done
              and (r["model"] == "depthpro2-s" or r["precision"] != "fp16")]  # B, L and H+ are fp32 only
    rows = [(sort_key(r), row(r)) for r in runs] + [(sort_key(r), failed_row(r)) for r in failed]
    name = runs[0]["device"] if runs else folder.name
    lines = [f"### {name}", ""]
    if runs:
        mode = runs[0]["power_mode"]
        lines += [f"Power mode: {mode['name']}. L4T {runs[0]['versions']['l4t']}, TensorRT {runs[0]['versions']['tensorrt']}.", ""]
    lines += [HEADER] + [text for _, text in sorted(rows, key=lambda x: x[0])] + [""]
    lines += soak_lines(folder) + stability_lines(folder)
    return "\n".join(lines)


def soak_lines(folder: Path) -> list[str]:
    soaks = [json.loads(p.read_text()) for p in sorted(folder.glob("soak_*.json"))]
    if not soaks:
        return []
    lines = ["Sustained runs:", "",
             "| Model | Input | Frames | Precision | Duration (s) | FPS including upload | Latency median / p95 (ms) | "
             "Peak GPU temp (C) | GPU clock min (MHz) | Lowest free memory (MB) | GPU shared |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in soaks:
        t, d = s["timing"], s["during_run"]
        lines.append(f"| {MODEL_ORDER[s['model']][1]} | {s['width']}x{s['height']} | {s['window']} | {s['precision']} | "
                     f"{t['seconds']:.0f} | {t['fps_sustained_including_upload']:.1f} | "
                     f"{t['latency_ms_median']:.1f} / {t['latency_ms_p95']:.1f} | "
                     f"{d['temperature_c']['gpu-thermal']['max']:.1f} | {d['gpu_freq_mhz']['min']:.0f} | "
                     f"{s['memory']['system_mem_available_min_during_run_mb']:.0f} | "
                     f"{'yes' if s['contention']['gpu_busy_or_shared'] else 'no'} |")
    return lines + [""]


def stability_lines(folder: Path) -> list[str]:
    files = sorted(folder.glob("temporal_stability_*.json"), key=lambda p: json.loads(p.read_text())["width"])
    if not files:
        return []
    lines = ["Frame-to-frame stability on the sample video (lower is steadier; the camera moves, so part of "
             "each change is real):", "",
             "| Model | Input | Precision | Frames | Change, median | Change, p95 | Scale jump, median | "
             "Scale jump, p95 | Scale jump, max |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for path in files:
        r = json.loads(path.read_text())
        for window, key in ((1, "window_1"), (4, "window_4")):
            x = r[key]
            lines.append(f"| {MODEL_ORDER[r['model']][1]} | {r['width']}x{r['height']} | {r['precision']} | {window} | "
                         f"{x['relative_change_median'] * 100:.2f}% | {x['relative_change_p95'] * 100:.2f}% | "
                         f"{x['scale_jump_median'] * 100:.2f}% | {x['scale_jump_p95'] * 100:.2f}% | "
                         f"{x['scale_jump_max'] * 100:.2f}% |")
    return lines + [""]


def main() -> None:
    folders = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    body = "\n".join(device_section(folder) for folder in folders)
    note = ("FPS is `1000 / median latency` for TensorRT execution alone. "
            "Power on the Orin Nano is the module input rail (`VDD_IN`). Power marked * is the GPU and SoC rail "
            "(`VDD_GPU_SOC`) only, so the two devices' power figures are not comparable. "
            "Every rail is in the JSON files.\n")
    table = f"{body}\n{note}"
    (RESULTS / "RESULTS.md").write_text("# Benchmark results\n\nGenerated by `scripts/make_results_table.py`.\n\n" + table)
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        if START in text and END in text:
            head, rest = text.split(START, 1)
            _, tail = rest.split(END, 1)
            readme.write_text(f"{head}{START}\n{table}{END}{tail}")


if __name__ == "__main__":
    main()

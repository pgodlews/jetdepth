"""Read-only Jetson telemetry from sysfs and procfs. Nothing here changes device settings."""
from __future__ import annotations

import glob
import os
import re
import threading
import time
from pathlib import Path

GPU_LOAD_PATHS = ("/sys/devices/platform/bus@0/17000000.gpu/load", "/sys/devices/gpu.0/load")


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def gpu_load_percent() -> float | None:
    for path in GPU_LOAD_PATHS:
        value = _read(path)
        if value is not None and value.isdigit():
            return int(value) / 10
    return None


def gpu_freq_mhz() -> float | None:
    for path in glob.glob("/sys/devices/platform/bus@0/17000000.gpu/devfreq/*/cur_freq"):
        value = _read(path)
        if value and value.isdigit():
            return int(value) / 1e6
    return None


def power_watts() -> dict[str, float]:
    """Instantaneous rail power from the on-board INA3221 monitors, where readable."""
    rails = {}
    for label_path in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
        label = _read(label_path)
        match = re.search(r"in(\d+)_label$", label_path)
        if not label or not match:
            continue
        base = os.path.dirname(label_path)
        mv, ma = _read(f"{base}/in{match[1]}_input"), _read(f"{base}/curr{match[1]}_input")
        if mv and ma and mv.lstrip("-").isdigit() and ma.lstrip("-").isdigit():
            rails[label] = int(mv) * int(ma) / 1e6
    return rails


def temperatures_c() -> dict[str, float]:
    temps = {}
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        name, value = _read(f"{zone}/type"), _read(f"{zone}/temp")
        if name and value and value.lstrip("-").isdigit() and int(value) > -40000:
            temps[name] = int(value) / 1000
    return temps


def mem_available_mb() -> float | None:
    text = _read("/proc/meminfo") or ""
    match = re.search(r"MemAvailable:\s+(\d+) kB", text)
    return int(match[1]) / 1024 if match else None


def process_peak_rss_mb() -> float | None:
    text = _read("/proc/self/status") or ""
    match = re.search(r"VmHWM:\s+(\d+) kB", text)
    return int(match[1]) / 1024 if match else None


def power_mode() -> dict:
    """The active power mode, from the status file and the mode table. Never changes it."""
    status = _read("/var/lib/nvpmodel/status") or ""
    match = re.search(r"pmode:(\d+)", status)
    if not match:
        return {"id": None, "name": None}
    mode = int(match[1])
    table = _read("/etc/nvpmodel.conf") or ""
    named = re.search(rf"POWER_MODEL\s+ID=(?:0*){mode}\s+NAME=([^\s>]+)", table)
    return {"id": mode, "name": named[1] if named else None}


def device_model() -> str | None:
    text = _read("/proc/device-tree/model")
    return text.rstrip("\x00") if text else None


def l4t_release() -> str | None:
    first = (_read("/etc/nv_tegra_release") or "").splitlines()[:1]
    if not first:
        return None
    match = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", first[0])
    return f"{match[1]}.{match[2]}" if match else None


def process_cpu_seconds(names: list[str]) -> float:
    """Total CPU time of processes with these command names, to detect other GPU clients."""
    ticks = 0
    for stat in glob.glob("/proc/[0-9]*/stat"):
        text = _read(stat)
        if not text:
            continue
        right = text.rfind(")")
        comm = text[text.find("(") + 1:right]
        if comm in names:
            fields = text[right + 2:].split()
            ticks += int(fields[11]) + int(fields[12])
    return ticks / os.sysconf("SC_CLK_TCK")


class Sampler(threading.Thread):
    """Samples load, clock, power and temperature on a background thread."""

    def __init__(self, period: float = 0.5):
        super().__init__(daemon=True)
        self.period = period
        self.samples: list[dict] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.samples.append({"gpu_load": gpu_load_percent(), "gpu_mhz": gpu_freq_mhz(),
                                 "power": power_watts(), "temps": temperatures_c(),
                                 "mem_available_mb": mem_available_mb(), "t": time.monotonic()})
            self._halt.wait(self.period)

    def stop(self) -> list[dict]:
        self._halt.set()
        self.join(timeout=3)
        return self.samples

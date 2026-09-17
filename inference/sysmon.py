"""RAM/VRAM sampling around inference (baseline, peak, post-unload).

RAM tracks this process RSS via psutil. VRAM tracks the NVIDIA device via
NVML (nvidia-ml-py). Anything unmeasurable is None, never an estimate.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass

import psutil

try:
    with warnings.catch_warnings():
        # The standalone 'pynvml' dist shares its module name with
        # nvidia-ml-py and emits a deprecation nag on import. Either
        # distribution provides the same NVML bindings we need.
        warnings.filterwarnings(
            'ignore', message='.*pynvml package is deprecated.*'
        )
        import pynvml
except Exception:
    pynvml = None

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class SystemSample:
    ram_baseline_mb: float
    ram_peak_mb: float
    vram_baseline_mib: float | None
    vram_peak_mib: float | None
    vram_total_mib: float | None
    nvml_available: bool


def _process_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / _BYTES_PER_MB


class _NvmlDevice:
    """One NVML device handle, or an unavailable marker."""

    def __init__(self, index: int = 0) -> None:
        self.available = False
        self.handle: object = None
        self.total_mib: float | None = None
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if index >= count:
                return
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            total = float(pynvml.nvmlDeviceGetMemoryInfo(handle).total) / _BYTES_PER_MB
            self.handle = handle
            self.total_mib = total
            self.available = True
        except Exception:
            self.available = False

    def used_mib(self) -> float | None:
        if not self.available or pynvml is None:
            return None
        try:
            return float(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used) / _BYTES_PER_MB
        except Exception:
            return None


def sample_vram_once(index: int = 0) -> tuple[float | None, float | None]:
    """One-shot (used_mib, total_mib); (None, None) when NVML is unavailable."""
    device = _NvmlDevice(index)
    if not device.available:
        return None, None
    return device.used_mib(), device.total_mib


class SystemSampler:
    """Background sampler; use as a context manager around one generation."""

    def __init__(self, *, interval_s: float = 0.1, device_index: int = 0) -> None:
        self._interval_s = interval_s
        self._device = _NvmlDevice(device_index)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ram_peak = 0.0
        self._vram_peak: float | None = None
        self._ram_baseline = 0.0
        self._vram_baseline: float | None = None

    def __enter__(self) -> SystemSampler:
        self._ram_baseline = _process_rss_mb()
        self._ram_peak = self._ram_baseline
        self._vram_baseline = self._device.used_mib()
        self._vram_peak = self._vram_baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            rss = _process_rss_mb()
            if rss > self._ram_peak:
                self._ram_peak = rss
            used = self._device.used_mib()
            if used is not None and (
                self._vram_peak is None or used > self._vram_peak
            ):
                self._vram_peak = used

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # Final synchronous sample so short generations still get a peak.
        rss = _process_rss_mb()
        if rss > self._ram_peak:
            self._ram_peak = rss
        used = self._device.used_mib()
        if used is not None and (
            self._vram_peak is None or used > self._vram_peak
        ):
            self._vram_peak = used

    def sample(self) -> SystemSample:
        return SystemSample(
            ram_baseline_mb=self._ram_baseline,
            ram_peak_mb=self._ram_peak,
            vram_baseline_mib=self._vram_baseline,
            vram_peak_mib=self._vram_peak,
            vram_total_mib=self._device.total_mib,
            nvml_available=self._device.available,
        )

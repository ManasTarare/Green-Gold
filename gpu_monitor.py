"""
gpu_monitor.py — Real GPU telemetry via nvidia-ml-py (pynvml).

Provides a thin, safe wrapper around NVML. If no NVIDIA GPU is present
(or the driver is unavailable), every function falls back gracefully and
returns None so the rest of the app can continue in simulation mode.

Usage:
    from gpu_monitor import GPUMonitor
    mon = GPUMonitor()
    for info in mon.read_all():
        print(info)   # dict with temp, util, vram, etc.
    mon.shutdown()
"""

from __future__ import annotations
import logging

log = logging.getLogger(__name__)

# Try the modern package first, fall back to legacy pynvml name
try:
    import nvidia_ml_py as nvml          # nvidia-ml-py >= 12
    _NVML_PKG = "nvidia_ml_py"
except ImportError:
    try:
        import pynvml as nvml            # older pynvml shim
        _NVML_PKG = "pynvml"
    except ImportError:
        nvml = None
        _NVML_PKG = None


class GPUMonitor:
    """
    Thin NVML wrapper that reads real GPU metrics.

    Attributes
    ----------
    available : bool
        True if NVML initialised successfully and at least one GPU exists.
    gpu_count : int
        Number of physical GPUs detected.
    """

    def __init__(self) -> None:
        self.available = False
        self.gpu_count = 0
        self._handles: list = []

        if nvml is None:
            log.warning("nvidia-ml-py / pynvml not installed — GPU monitoring disabled.")
            return

        try:
            nvml.nvmlInit()
            self.gpu_count = nvml.nvmlDeviceGetCount()
            self._handles = [
                nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)
            ]
            self.available = self.gpu_count > 0
            if self.available:
                names = [self._safe_name(h) for h in self._handles]
                log.info("GPUMonitor: NVML OK — %d GPU(s): %s", self.gpu_count, names)
            else:
                log.warning("GPUMonitor: NVML initialised but no GPUs found.")
        except Exception as exc:  # NVMLError or anything else
            log.warning("GPUMonitor: NVML init failed (%s) — running in sim-only mode.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict]:
        """
        Return a list of dicts, one per physical GPU, with keys:
            gpu_index   int     — 0-based GPU index
            name        str     — e.g. "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
            temp_c      int     — GPU die temperature in °C
            util_pct    int     — GPU core utilisation 0-100
            mem_util_pct int    — GPU memory controller utilisation 0-100
            vram_used_mb float  — VRAM used in MiB
            vram_total_mb float — VRAM total in MiB
            vram_pct    float   — VRAM utilisation percentage
            power_w     float | None — power draw in watts (None if unsupported)
            fan_pct     int | None   — fan speed 0-100 (None if unsupported)
            source      str     — always "real"
        """
        if not self.available:
            return []

        results = []
        for idx, handle in enumerate(self._handles):
            try:
                temp   = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                util   = nvml.nvmlDeviceGetUtilizationRates(handle)
                mem    = nvml.nvmlDeviceGetMemoryInfo(handle)

                vram_used  = mem.used  / 1024 / 1024
                vram_total = mem.total / 1024 / 1024
                vram_pct   = round(vram_used / max(vram_total, 1) * 100, 1)

                power = self._safe_power(handle)
                fan   = self._safe_fan(handle)

                results.append({
                    "gpu_index":     idx,
                    "name":          self._safe_name(handle),
                    "temp_c":        temp,
                    "util_pct":      util.gpu,
                    "mem_util_pct":  util.memory,
                    "vram_used_mb":  round(vram_used,  1),
                    "vram_total_mb": round(vram_total, 1),
                    "vram_pct":      vram_pct,
                    "power_w":       power,
                    "fan_pct":       fan,
                    "source":        "real",
                })
            except Exception as exc:
                log.error("GPUMonitor.read_all: GPU %d read error: %s", idx, exc)

        return results

    def read_one(self, gpu_index: int = 0) -> dict | None:
        """Convenience: read a single GPU by index. Returns None on failure."""
        all_gpus = self.read_all()
        if gpu_index < len(all_gpus):
            return all_gpus[gpu_index]
        return None

    def shutdown(self) -> None:
        """Release NVML resources. Call once on app exit."""
        if nvml is not None and self.available:
            try:
                nvml.nvmlShutdown()
                log.info("GPUMonitor: NVML shutdown OK.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_name(self, handle) -> str:
        try:
            name = nvml.nvmlDeviceGetName(handle)
            return name.decode() if isinstance(name, bytes) else name
        except Exception:
            return "Unknown GPU"

    def _safe_power(self, handle) -> float | None:
        try:
            mw = nvml.nvmlDeviceGetPowerUsage(handle)
            return round(mw / 1000.0, 1)
        except Exception:
            return None

    def _safe_fan(self, handle) -> int | None:
        try:
            return nvml.nvmlDeviceGetFanSpeed(handle)
        except Exception:
            return None

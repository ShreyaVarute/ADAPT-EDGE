"""
Resource monitoring utility for ADAPT-EDGE.
Monitors CPU load, RAM usage, GPU load, GPU memory, and GPU temperature.
Works out-of-the-box on Windows using standard library ctypes (GlobalMemoryStatusEx, GetSystemTimes),
PyTorch CUDA runtime, and nvidia-smi if available.
Honest reporting: if any metric is unsupported or unavailable, returns None and labels it unavailable.
"""

import os
import sys
import time
import shutil
import subprocess
from typing import Dict, Any, Optional
import torch
import threading

# Try importing psutil if installed by user; otherwise use native fallback
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

# Native Windows ctypes structures for CPU and Memory
import ctypes

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

class _FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.c_ulong),
        ("dwHighDateTime", ctypes.c_ulong)
    ]


class ResourceMonitor:
    """
    Monitors device computational and hardware state without requiring external package installations.
    Honest reporting: never fabricates values; returns None with status note when unsupported.
    """

    def __init__(self, sample_interval_s: float = 0.05, default_cache_ttl_s: float = 0.5):
        self.sample_interval_s = sample_interval_s
        self.default_cache_ttl_s = default_cache_ttl_s
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._is_windows = (sys.platform == "win32")
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cached_snapshot_time: float = 0.0
        self._mock_snapshot: Optional[Dict[str, Any]] = None
        self._refresh_lock = threading.Lock()

    def set_mock_snapshot(self, mock_dict: Optional[Dict[str, Any]]) -> None:
        """Set or clear a deterministic mock snapshot for testing and experiments."""
        self._mock_snapshot = mock_dict

    def clear_cache(self) -> None:
        """Explicitly clear the snapshot cache."""
        self._cached_snapshot = None
        self._cached_snapshot_time = 0.0


    def snapshot(self, max_age_s: Optional[float] = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Capture or retrieve a thread-safe cached resource snapshot.
        """
        if self._mock_snapshot is not None:
            res = dict(self._mock_snapshot)
            res["is_mocked"] = True
            res["snapshot_age_s"] = 0.0
            res["monitoring_time_ms"] = 0.01
            return res

        ttl = max_age_s if max_age_s is not None else self.default_cache_ttl_s

        with self._refresh_lock:
            now = time.time()

            if not force_refresh and self._cached_snapshot is not None:
                age = now - self._cached_snapshot_time
                if age <= ttl:
                    cached_res = dict(self._cached_snapshot)
                    cached_res["snapshot_age_s"] = round(age, 3)
                    cached_res["monitoring_time_ms"] = 0.0
                    cached_res["is_cached"] = True
                    return cached_res

            t_start = time.perf_counter()
            ram = self.get_system_ram()
            cpu = self.get_cpu_utilization()
            gpu = self.get_gpu_status()
            monitoring_time = (time.perf_counter() - t_start) * 1000.0

            ram_available_mb = ram.get("free_mb") if ram.get("available") else None
            gpu_total_vram_mb = gpu.get("total_vram_mb")
            gpu_used_vram_mb = gpu.get("nvidia_smi_used_mb")
            gpu_free_vram_mb = None

            if gpu_total_vram_mb is not None and gpu_used_vram_mb is not None:
                gpu_free_vram_mb = gpu_total_vram_mb - gpu_used_vram_mb

            snap = {
                "timestamp": now,
                "cpu_percent": cpu.get("percent") if cpu.get("available") else None,
                "ram_used_mb": ram.get("used_mb") if ram.get("available") else None,
                "ram_total_mb": ram.get("total_mb") if ram.get("available") else None,
                "ram_available_mb": ram_available_mb,
                "ram_percent": ram.get("percent_used") if ram.get("available") else None,
                "gpu_available": gpu.get("available", False),
                "gpu_name": gpu.get("device_name"),
                "gpu_torch_allocated_mb": gpu.get("torch_memory_allocated_mb"),
                "gpu_torch_reserved_mb": gpu.get("torch_memory_reserved_mb"),
                "gpu_torch_peak_mb": gpu.get("torch_max_memory_allocated_mb"),
                "gpu_utilization_percent": gpu.get("utilization_gpu_percent"),
                "gpu_temperature_c": gpu.get("temperature_c"),
                "total_vram_mb": gpu_total_vram_mb,
                "gpu_used_vram_mb": gpu_used_vram_mb,
                "gpu_free_vram_mb": gpu_free_vram_mb,
                "nvidia_smi_used_mb": gpu_used_vram_mb,
                "snapshot_age_s": 0.0,
                "monitoring_time_ms": round(monitoring_time, 2),
                "is_cached": False,
                "diagnostics": {
                    "cpu_source": cpu.get("source"),
                    "ram_source": ram.get("source"),
                    "gpu_source": gpu.get("sensor_source", "unavailable")
                }
            }

            self._cached_snapshot = snap
            self._cached_snapshot_time = time.time()
            return dict(snap)


    def get_system_ram(self) -> Dict[str, Any]:
        """
        Returns RAM statistics: total_mb, used_mb, available_mb, used_percent.
        Uses psutil if available; otherwise native Windows ctypes GlobalMemoryStatusEx.
        """
        if _HAS_PSUTIL:
            vm = psutil.virtual_memory()
            total_mb = vm.total / (1024.0 ** 2)
            used_mb = vm.used / (1024.0 ** 2)
            avail_mb = vm.available / (1024.0 ** 2)
            return {
                "available": True,
                "total_mb": round(total_mb, 1),
                "used_mb": round(used_mb, 1),
                "free_mb": round(avail_mb, 1),
                "percent_used": round(vm.percent, 1),
                "source": "psutil"
            }

        if self._is_windows:
            try:
                stat = _MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_mb = stat.ullTotalPhys / (1024.0 ** 2)
                    avail_mb = stat.ullAvailPhys / (1024.0 ** 2)
                    used_mb = total_mb - avail_mb
                    pct = float(stat.dwMemoryLoad)
                    return {
                        "available": True,
                        "total_mb": round(total_mb, 1),
                        "used_mb": round(used_mb, 1),
                        "free_mb": round(avail_mb, 1),
                        "percent_used": round(pct, 1),
                        "source": "windows_ctypes_GlobalMemoryStatusEx"
                    }
            except Exception as e:
                return {
                    "available": False,
                    "error": str(e),
                    "source": "windows_ctypes_failed"
                }

        return {
            "available": False,
            "error": "No psutil or Windows ctypes support",
            "source": "unavailable"
        }

    def get_cpu_utilization(self, interval_s: Optional[float] = None) -> Dict[str, Any]:
        """
        Returns CPU utilization percentage across cores.
        """
        if _HAS_PSUTIL:
            pct = psutil.cpu_percent(interval=interval_s or self.sample_interval_s)
            return {
                "available": True,
                "percent": round(pct, 1),
                "source": "psutil"
            }

        if self._is_windows:
            try:
                def _filetime_to_int(ft):
                    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

                i1, k1, u1 = _FILETIME(), _FILETIME(), _FILETIME()
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i1), ctypes.byref(k1), ctypes.byref(u1))
                time.sleep(interval_s or self.sample_interval_s)
                i2, k2, u2 = _FILETIME(), _FILETIME(), _FILETIME()
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i2), ctypes.byref(k2), ctypes.byref(u2))

                idle = _filetime_to_int(i2) - _filetime_to_int(i1)
                kernel = _filetime_to_int(k2) - _filetime_to_int(k1)
                user = _filetime_to_int(u2) - _filetime_to_int(u1)
                total = kernel + user

                if total > 0:
                    pct = 100.0 * (1.0 - (idle / float(total)))
                    return {
                        "available": True,
                        "percent": round(max(0.0, min(100.0, pct)), 1),
                        "source": "windows_ctypes_GetSystemTimes"
                    }
            except Exception as e:
                return {
                    "available": False,
                    "error": str(e),
                    "source": "windows_ctypes_failed"
                }

        return {
            "available": False,
            "error": "CPU monitoring unavailable without psutil on this platform",
            "source": "unavailable"
        }

    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Returns GPU status: device_name, memory_allocated_mb, memory_reserved_mb,
        utilization_gpu_percent, temperature_celsius.
        Uses PyTorch CUDA API for memory and nvidia-smi query for utilization & temperature.
        """
        if not torch.cuda.is_available():
            return {
                "available": False,
                "reason": "PyTorch reports CUDA unavailable"
            }

        status: Dict[str, Any] = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "torch_memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024.0 ** 2), 2),
            "torch_memory_reserved_mb": round(torch.cuda.memory_reserved() / (1024.0 ** 2), 2),
            "torch_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024.0 ** 2), 2),
            "utilization_gpu_percent": None,
            "temperature_c": None,
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2), 1),
        }

        # Query nvidia-smi for utilization and temperature if installed
        if self._has_nvidia_smi:
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
                if res.returncode == 0 and res.stdout.strip():
                    parts = [p.strip() for p in res.stdout.strip().split(",")]
                    if len(parts) >= 3:
                        status["utilization_gpu_percent"] = float(parts[0])
                        status["nvidia_smi_used_mb"] = float(parts[1])
                        status["temperature_c"] = float(parts[2])
                        status["sensor_source"] = "nvidia-smi"
            except Exception as e:
                status["sensor_query_error"] = str(e)
                status["sensor_source"] = "query_failed"
        else:
            status["sensor_source"] = "nvidia-smi_not_found"

        return status




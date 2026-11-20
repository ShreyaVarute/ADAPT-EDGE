import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.resource_monitor import ResourceMonitor


class TestResourceMonitor(unittest.TestCase):
    """
    Unit tests for utils/resource_monitor.py:
    - RAM query (valid numerical output or honest unavailable)
    - CPU query (valid percentage or honest unavailable)
    - GPU status (PyTorch CUDA query & nvidia-smi if present)
    - Snapshot composition
    """

    def setUp(self):
        self.monitor = ResourceMonitor(sample_interval_s=0.01)

    def test_ram_query(self):
        ram = self.monitor.get_system_ram()
        self.assertIsInstance(ram, dict)
        self.assertIn("available", ram)
        if ram["available"]:
            self.assertGreater(ram["total_mb"], 0)
            self.assertGreater(ram["used_mb"], 0)
            self.assertGreaterEqual(ram["percent_used"], 0.0)
            self.assertLessEqual(ram["percent_used"], 100.0)

    def test_cpu_query(self):
        cpu = self.monitor.get_cpu_utilization(interval_s=0.01)
        self.assertIsInstance(cpu, dict)
        self.assertIn("available", cpu)
        if cpu["available"]:
            self.assertGreaterEqual(cpu["percent"], 0.0)
            self.assertLessEqual(cpu["percent"], 100.0)

    def test_gpu_status(self):
        gpu = self.monitor.get_gpu_status()
        self.assertIsInstance(gpu, dict)
        self.assertIn("available", gpu)
        if gpu["available"]:
            self.assertIn("device_name", gpu)
            self.assertIn("torch_memory_allocated_mb", gpu)
            # Temperature and utilization may be None if sensors are unavailable or float if present
            if gpu.get("temperature_c") is not None:
                self.assertGreater(gpu["temperature_c"], 0)

    def test_snapshot_structure(self):
        snap = self.monitor.snapshot()
        self.assertIsInstance(snap, dict)
        self.assertIn("timestamp", snap)
        self.assertIn("cpu_percent", snap)
        self.assertIn("ram_used_mb", snap)
        self.assertIn("gpu_available", snap)
        self.assertIn("diagnostics", snap)


if __name__ == "__main__":
    unittest.main()


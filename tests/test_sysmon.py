"""Self-check for ResourceMonitor.summary() peak-load rollup."""
from __future__ import annotations

from ocr_bench.sysmon import GpuSample, HIGH_LOAD_THRESHOLD, ResourceMonitor, SystemSample


def test_summary_high_load_duration_and_temps():
    mon = ResourceMonitor(interval_s=2.0)
    mon._samples = [
        SystemSample(cpu_percent=20, cpu_count=4, ram_percent=50, ram_used_mb=100,
                     ram_total_mb=200, disk_percent=None, cpu_temp_c=40.0, gpus=[]),
        SystemSample(cpu_percent=95, cpu_count=4, ram_percent=50, ram_used_mb=100,
                     ram_total_mb=200, disk_percent=None, cpu_temp_c=60.0,
                     gpus=[GpuSample(0, "gpu", 91.0, None, None, 65.0)]),
        SystemSample(cpu_percent=30, cpu_count=4, ram_percent=50, ram_used_mb=100,
                     ram_total_mb=200, disk_percent=None, cpu_temp_c=41.0,
                     gpus=[GpuSample(0, "gpu", 92.0, None, None, 66.0)]),
    ]
    s = mon.summary()
    assert s["high_load_threshold"] == HIGH_LOAD_THRESHOLD
    assert s["high_load_samples"] == 2  # sample 2 (CPU 95%) and sample 3 (GPU 92%)
    assert s["high_load_duration_s"] == 4.0  # 2 samples * 2s interval
    assert s["high_load_cpu_temp_c_max"] == 60.0
    assert s["high_load_gpu_temp_c_max"] == 66.0


if __name__ == "__main__":
    test_summary_high_load_duration_and_temps()
    print("ok")

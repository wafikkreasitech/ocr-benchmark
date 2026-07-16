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


def test_summary_ram_delta_subtracts_idle_baseline():
    """RAM delta must report only what the run added on top of idle — a
    machine sitting at 4000 MB idle that peaks at 5500 MB during the run
    should report a 1500 MB delta, not the raw 5500 MB peak."""
    mon = ResourceMonitor(interval_s=2.0)
    mon.baseline = SystemSample(cpu_percent=5, cpu_count=4, ram_percent=20,
                                 ram_used_mb=4000, ram_total_mb=20000,
                                 disk_percent=None, cpu_temp_c=None, gpus=[])
    mon._samples = [
        SystemSample(cpu_percent=40, cpu_count=4, ram_percent=25, ram_used_mb=5000,
                     ram_total_mb=20000, disk_percent=None, cpu_temp_c=None, gpus=[]),
        SystemSample(cpu_percent=60, cpu_count=4, ram_percent=27.5, ram_used_mb=5500,
                     ram_total_mb=20000, disk_percent=None, cpu_temp_c=None, gpus=[]),
    ]
    s = mon.summary()
    assert s["baseline_ram_used_mb"] == 4000
    assert s["ram_used_mb_max"] == 5500          # raw peak, unchanged
    assert s["ram_used_mb_delta_max"] == 1500    # 5500 - 4000: the real cost
    assert s["ram_used_mb_delta_avg"] == 1250    # avg(5000,5500) - 4000
    assert s["cpu_percent_delta_max"] == 55       # 60 - 5


def test_summary_ram_delta_absent_without_baseline():
    """No baseline captured (e.g. summary() called without start()) — delta
    fields must be None, never a misleading fallback to raw peak."""
    mon = ResourceMonitor(interval_s=2.0)
    mon._samples = [
        SystemSample(cpu_percent=20, cpu_count=4, ram_percent=50, ram_used_mb=100,
                     ram_total_mb=200, disk_percent=None, cpu_temp_c=None, gpus=[]),
    ]
    s = mon.summary()
    assert s["ram_used_mb_delta_max"] is None
    assert s["baseline_ram_used_mb"] is None


if __name__ == "__main__":
    test_summary_high_load_duration_and_temps()
    test_summary_ram_delta_subtracts_idle_baseline()
    test_summary_ram_delta_absent_without_baseline()
    print("ok")

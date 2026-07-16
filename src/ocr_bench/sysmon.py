"""System resource sampling — CPU / RAM / GPU / temperature.

Best-effort only: every stat degrades gracefully to ``None`` when the host
doesn't expose it (e.g. no NVIDIA GPU, no temperature sensors on Windows).
Nothing in here should ever raise — a monitoring feature must never crash a
benchmark run.

Two entry points:
  ``sample_dict()``      — one instantaneous reading, used by ``GET /api/system``.
  ``ResourceMonitor``    — background sampler used by ``runner.run()`` to record
                           avg/peak usage for the whole run, saved into history.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

import psutil

log = logging.getLogger(__name__)

HIGH_LOAD_THRESHOLD = 90.0  # percent — CPU/GPU utilization considered "peak load"


def _is_high_load(s: "SystemSample") -> bool:
    return s.cpu_percent >= HIGH_LOAD_THRESHOLD or any(
        g.util_percent is not None and g.util_percent >= HIGH_LOAD_THRESHOLD for g in s.gpus
    )

# Prime psutil's internal CPU-percent tracker so the very first real sample
# isn't a meaningless 0.0 (psutil.cpu_percent compares against the last call).
psutil.cpu_percent(interval=None)


@dataclass
class GpuSample:
    index: int
    name: str
    util_percent: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None
    temp_c: float | None


@dataclass
class SystemSample:
    cpu_percent: float
    cpu_count: int
    ram_percent: float
    ram_used_mb: float
    ram_total_mb: float
    disk_percent: float | None
    cpu_temp_c: float | None
    gpus: list[GpuSample] = field(default_factory=list)
    timestamp: str = ""


@lru_cache(maxsize=1)
def _nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


@lru_cache(maxsize=1)
def _tegrastats_path() -> str | None:
    return shutil.which("tegrastats")


def _parse_tegrastats_line(line: str) -> dict | None:
    """Parse one tegrastats line and return GPU util/temp + CPU/RAM info.

    Format: RAM 3360/7546MB CPU [17%@729,...] GR3D_FREQ 0% gpu@52.718C/... tj@53.25C/...
    Returns dict with keys: gpu_pct, gpu_temp_c, cpu_pct, ram_used_mb, ram_total_mb.
    """
    import re
    ram_m = re.search(r"RAM (\d+)/(\d+)MB", line)
    cpu_m = re.search(r"CPU \[(.*?)\]", line)
    gpu_m = re.search(r"GR3D_FREQ (\d+)%", line)
    gpu_temp_m = re.search(r"gpu@([\d.]+)C", line)
    if not (ram_m and cpu_m and gpu_m):
        return None
    cpu_cores = re.findall(r"(\d+)%", cpu_m.group(1))
    if not cpu_cores:
        return None
    return {
        "gpu_pct": int(gpu_m.group(1)),
        "gpu_temp_c": float(gpu_temp_m.group(1)) if gpu_temp_m else None,
        "cpu_pct": sum(int(c) for c in cpu_cores) / len(cpu_cores),
        "ram_used_mb": int(ram_m.group(1)),
        "ram_total_mb": int(ram_m.group(2)),
    }


def _read_tegrastats() -> GpuSample | None:
    """One-shot tegrastats sample (runs tegrastats --interval 1000, reads 2 lines)."""
    exe = _tegrastats_path()
    if not exe:
        return None
    try:
        proc = subprocess.Popen(
            [exe, "--interval", "1000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        # Read 2 lines (first may be stale, second is a fresh sample)
        line = ""
        for _ in range(2):
            next_line = proc.stdout.readline()
            if next_line:
                line = next_line
        proc.terminate()
        proc.wait(timeout=2)
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("tegrastats read failed: %s", e)
        return None
    parsed = _parse_tegrastats_line(line) if line else None
    if not parsed:
        return None
    return GpuSample(
        index=0,
        name="Orin GPU (GR3D)",
        util_percent=float(parsed["gpu_pct"]),
        mem_used_mb=None,  # tegrastats doesn't expose GPU dedicated mem (unified memory)
        mem_total_mb=None,
        temp_c=parsed.get("gpu_temp_c"),
    )


def _read_gpus() -> list[GpuSample]:
    # Prefer tegrastats on Jetson (nvidia-smi shows N/A for integrated GPU)
    tegrastats = _read_tegrastats()
    if tegrastats is not None:
        return [tegrastats]
    # Fall back to nvidia-smi (discrete GPUs, desktop)
    exe = _nvidia_smi_path()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("nvidia-smi query failed: %s", e)
        return []
    gpus: list[GpuSample] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(GpuSample(
                index=int(parts[0]), name=parts[1],
                util_percent=float(parts[2]), mem_used_mb=float(parts[3]),
                mem_total_mb=float(parts[4]), temp_c=float(parts[5]),
            ))
        except ValueError:
            continue
    return gpus


def _read_cpu_temp() -> float | None:
    """Best-effort CPU temperature.

    ``psutil.sensors_temperatures()`` only exists on Linux; most Windows/macOS
    hosts have no supported sensor path, so this quietly returns ``None``
    there rather than raising.
    """
    reader = getattr(psutil, "sensors_temperatures", None)
    if reader is None:
        return None
    try:
        temps = reader()
    except (AttributeError, OSError, NotImplementedError):
        return None
    for entries in temps.values():
        for entry in entries:
            if entry.current:
                return float(entry.current)
    return None


def _read_disk_percent() -> float | None:
    try:
        return psutil.disk_usage(os.getcwd()).percent
    except OSError:
        return None


def sample(cpu_interval: float | None = None) -> SystemSample:
    """One instantaneous reading.

    ``cpu_interval=None`` is non-blocking (compares against the last call —
    fine for periodic polling). Pass e.g. ``0.2`` for a one-shot accurate
    reading with no prior call to compare against.
    """
    cpu = psutil.cpu_percent(interval=cpu_interval)
    vm = psutil.virtual_memory()
    return SystemSample(
        cpu_percent=cpu,
        cpu_count=psutil.cpu_count() or 0,
        ram_percent=vm.percent,
        ram_used_mb=vm.used / (1024 * 1024),
        ram_total_mb=vm.total / (1024 * 1024),
        disk_percent=_read_disk_percent(),
        cpu_temp_c=_read_cpu_temp(),
        gpus=_read_gpus(),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def sample_dict(cpu_interval: float | None = None) -> dict:
    d = asdict(sample(cpu_interval))
    return d


class ResourceMonitor:
    """Background sampler — records CPU/RAM/GPU/temp every ``interval_s``
    while a benchmark run is in flight.

    ``.latest`` is read by the progress sidecar so the UI can show live usage
    during a run; ``.summary()`` (avg/peak) is saved into the run's history
    snapshot when the run finishes.

    Idle RAM/GPU-mem is never zero (OS + drivers + the web server itself sit
    on a baseline before any model loads), so raw peak-used numbers overstate
    what the *benchmark* actually costs. ``start()`` grabs one baseline sample
    before the run's engines are constructed; ``summary()`` subtracts it so
    ``ram_used_mb_delta_*`` reports only what this run added on top of idle.
    """

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[SystemSample] = []
        self._high_load_count = 0
        self.latest: dict | None = None
        self.baseline: SystemSample | None = None

    def start(self) -> None:
        self._stop.clear()
        # ponytail: synchronous baseline read at the exact "run started" moment
        # (before OCR/TTS engines get constructed) — must never crash the run.
        try:
            self.baseline = sample(cpu_interval=None)
        except Exception:  # noqa: BLE001
            log.debug("baseline resource sample failed", exc_info=True)
            self.baseline = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = sample(cpu_interval=None)
                with self._lock:
                    self._samples.append(s)
                    if _is_high_load(s):
                        self._high_load_count += 1
                    self.latest = asdict(s)
                    # Running "how long has this run been pegged" — read live by
                    # the progress sidecar so the UI can reassure the user mid-run
                    # instead of only reporting it after the fact in history.
                    self.latest["high_load_elapsed_s"] = self._high_load_count * self.interval_s
            except Exception:  # noqa: BLE001 — monitoring must never kill the run
                log.debug("resource sample failed", exc_info=True)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        """Avg/peak rollup across every sample taken during the run."""
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {}

        cpu_vals = [s.cpu_percent for s in samples]
        ram_vals = [s.ram_percent for s in samples]
        ram_used = [s.ram_used_mb for s in samples]
        cpu_temps = [s.cpu_temp_c for s in samples if s.cpu_temp_c is not None]
        gpu_util = [g.util_percent for s in samples for g in s.gpus if g.util_percent is not None]
        gpu_mem = [g.mem_used_mb for s in samples for g in s.gpus if g.mem_used_mb is not None]
        gpu_temp = [g.temp_c for s in samples for g in s.gpus if g.temp_c is not None]
        last_gpus = samples[-1].gpus
        gpu_name = last_gpus[0].name if last_gpus else None

        def avg(vals: list[float]) -> float | None:
            return (sum(vals) / len(vals)) if vals else None

        # Peak-load window: samples where CPU or GPU hit HIGH_LOAD_THRESHOLD.
        # Duration is approximate (sample count * interval), good enough for
        # "how long was this run pegged" without needing raw time-series storage.
        high_load = [s for s in samples if _is_high_load(s)]
        high_load_cpu_temps = [s.cpu_temp_c for s in high_load if s.cpu_temp_c is not None]
        high_load_gpu_temps = [g.temp_c for s in high_load for g in s.gpus if g.temp_c is not None]

        # Delta vs idle baseline — the honest "what did THIS run actually
        # cost" number, since RAM/GPU-mem/CPU never start at zero (OS,
        # drivers, the web server process itself all sit on a baseline).
        # None when no baseline was captured (e.g. monitor constructed but
        # start() never called) rather than silently reporting raw peak.
        base = self.baseline
        ram_used_delta_avg = ram_used_delta_max = None
        ram_percent_delta_max = None
        gpu_mem_delta_avg = gpu_mem_delta_max = None
        cpu_percent_delta_max = None
        if base is not None:
            ram_used_delta_avg = avg(ram_used) - base.ram_used_mb if ram_used else None
            ram_used_delta_max = (max(ram_used) - base.ram_used_mb) if ram_used else None
            ram_percent_delta_max = (max(ram_vals) - base.ram_percent) if ram_vals else None
            cpu_percent_delta_max = (max(cpu_vals) - base.cpu_percent) if cpu_vals else None
            base_gpu = base.gpus[0] if base.gpus else None
            if base_gpu is not None and base_gpu.mem_used_mb is not None and gpu_mem:
                gpu_mem_delta_avg = avg(gpu_mem) - base_gpu.mem_used_mb
                gpu_mem_delta_max = max(gpu_mem) - base_gpu.mem_used_mb

        return {
            "samples": len(samples),
            "cpu_percent_avg": avg(cpu_vals),
            "cpu_percent_max": max(cpu_vals) if cpu_vals else None,
            "ram_percent_avg": avg(ram_vals),
            "ram_percent_max": max(ram_vals) if ram_vals else None,
            "ram_used_mb_avg": avg(ram_used),
            "ram_used_mb_max": max(ram_used) if ram_used else None,
            "ram_total_mb": samples[-1].ram_total_mb,
            "cpu_temp_c_avg": avg(cpu_temps),
            "cpu_temp_c_max": max(cpu_temps) if cpu_temps else None,
            "gpu_name": gpu_name,
            "gpu_percent_avg": avg(gpu_util),
            "gpu_percent_max": max(gpu_util) if gpu_util else None,
            "gpu_mem_used_mb_avg": avg(gpu_mem),
            "gpu_mem_used_mb_max": max(gpu_mem) if gpu_mem else None,
            "gpu_mem_total_mb": last_gpus[0].mem_total_mb if last_gpus else None,
            "gpu_temp_c_avg": avg(gpu_temp),
            "gpu_temp_c_max": max(gpu_temp) if gpu_temp else None,
            "high_load_threshold": HIGH_LOAD_THRESHOLD,
            "high_load_samples": len(high_load),
            "high_load_duration_s": len(high_load) * self.interval_s,
            "high_load_cpu_temp_c_avg": avg(high_load_cpu_temps),
            "high_load_cpu_temp_c_max": max(high_load_cpu_temps) if high_load_cpu_temps else None,
            "high_load_gpu_temp_c_avg": avg(high_load_gpu_temps),
            "high_load_gpu_temp_c_max": max(high_load_gpu_temps) if high_load_gpu_temps else None,
            # ── Idle-baseline-subtracted deltas — the "real" cost of this run ──
            "baseline_ram_used_mb": base.ram_used_mb if base else None,
            "baseline_ram_percent": base.ram_percent if base else None,
            "baseline_cpu_percent": base.cpu_percent if base else None,
            "ram_used_mb_delta_avg": ram_used_delta_avg,
            "ram_used_mb_delta_max": ram_used_delta_max,
            "ram_percent_delta_max": ram_percent_delta_max,
            "cpu_percent_delta_max": cpu_percent_delta_max,
            "gpu_mem_used_mb_delta_avg": gpu_mem_delta_avg,
            "gpu_mem_used_mb_delta_max": gpu_mem_delta_max,
        }

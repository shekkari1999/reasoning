"""CUDA capture, memory reporting, and training-metric utilities."""

import json
from pathlib import Path
from collections import defaultdict

import torch


# ---------------------------------------------------------------------------
# CUDA Profiler control
# ---------------------------------------------------------------------------

class ProfilerControl:
    """Controls cudaProfiler start/stop for Nsight Systems capture.
    
    Use with `nsys profile --capture-range=cudaProfilerApi` to only
    capture the steps you care about (skip warmup).
    
    Usage:
        profiler = ProfilerControl(warmup_steps=10, capture_steps=20)
        for step in range(100):
            profiler.step(step)
            # ... training code ...
        profiler.stop()
    """

    def __init__(self, warmup_steps: int = 10, capture_steps: int = 20, enabled: bool = True):
        self.warmup_steps = warmup_steps
        self.capture_end = warmup_steps + capture_steps
        self.enabled = enabled
        self.started = False
        self.stopped = False

    def step(self, current_step: int):
        if not self.enabled:
            return

        if current_step == self.warmup_steps and not self.started:
            torch.cuda.cudart().cudaProfilerStart()
            self.started = True
            print(f"[Profiler] Started capture at step {current_step}")

        if current_step == self.capture_end and not self.stopped:
            self.stop()

    def stop(self):
        if self.enabled and self.started and not self.stopped:
            torch.cuda.cudart().cudaProfilerStop()
            self.stopped = True
            print("[Profiler] Stopped capture")


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def log_memory(tag: str = "", device: int = 0):
    """Log current GPU memory usage."""
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"[Memory{' ' + tag if tag else ''}] "
          f"Allocated: {allocated:.2f}GB | "
          f"Reserved: {reserved:.2f}GB | "
          f"Peak: {max_allocated:.2f}GB")


def get_memory_stats(device: int = None) -> dict:
    """Get memory stats as a dict for logging."""
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / 1e9
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return {
        "gpu_id": device,
        "gpu_name": props.name,
        "total_gb": round(total_gb, 1),
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / 1e9, 3),
        "peak_gb": round(peak_gb, 3),
        "memory_utilization_pct": round(peak_gb / total_gb * 100, 1) if total_gb > 0 else 0.0,
    }


def get_visible_gpu_memory() -> list[dict]:
    """Memory stats for every visible CUDA device (single-process eval)."""
    if not torch.cuda.is_available():
        return []
    return [get_memory_stats(i) for i in range(torch.cuda.device_count())]


def gather_all_gpu_memory() -> list[dict]:
    """Collect per-rank GPU memory stats in distributed training."""
    import torch.distributed as dist

    stats = get_memory_stats()
    if dist.is_initialized():
        stats["rank"] = dist.get_rank()
        world_size = dist.get_world_size()
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, stats)
        return gathered
    stats["rank"] = 0
    return [stats]


def save_memory_report(
    output_path: str | Path,
    stage: str,
    gpus: list[dict],
    extra: dict | None = None,
) -> None:
    """Save a compact memory report as JSON."""
    peak_per_gpu = [g["peak_gb"] for g in gpus]
    report = {
        "stage": stage,
        "num_gpus": len(gpus),
        "peak_mem_gb_per_gpu": peak_per_gpu,
        "peak_mem_gb_max": round(max(peak_per_gpu), 3) if peak_per_gpu else 0.0,
        "gpus": gpus,
    }
    if extra:
        report.update(extra)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Memory report saved to {path}")


def reset_peak_memory(device: int = 0):
    """Reset peak memory stats for a fresh measurement window."""
    torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Collect, print, and save per-step training metrics."""

    def __init__(self, log_dir: str = "results"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = defaultdict(list)
        self.current = {}

    def update(self, step: int, **kwargs):
        """Record metrics for a given step."""
        self.current["step"] = step
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.current[key] = value
            self.history[key].append({"step": step, "value": value})

    def log(self, step: int, prefix: str = ""):
        """Print current metrics."""
        parts = [f"[Step {step}]"]
        if prefix:
            parts[0] = f"[{prefix} Step {step}]"
        for key, value in self.current.items():
            if key == "step":
                continue
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
        print(" | ".join(parts))

    def save(self, filename: str = "training_metrics.json"):
        """Save all history to JSON."""
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"Metrics saved to {path}")

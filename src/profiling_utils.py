"""
Profiling utilities for Nsight Systems and Nsight Compute.

Provides:
  - NVTX range context managers for clean annotation
  - cudaProfiler start/stop wrappers (capture only the steps you want)
  - GPU memory logging
  - Metric tracking for training curves
"""

import time
import json
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
from typing import Optional

import torch
import torch.cuda.nvtx as nvtx


# ---------------------------------------------------------------------------
# NVTX annotation
# ---------------------------------------------------------------------------

@contextmanager
def nvtx_range(name: str):
    """Context manager for NVTX range annotation.

    Usage:
        with nvtx_range("forward"):
            logits = model(input_ids)
    
    Shows up as a labeled range in Nsight Systems timeline.
    """
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def nvtx_mark(name: str):
    """Place a single marker (not a range) on the timeline."""
    torch.cuda.nvtx.mark(name)


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
        self.capture_steps = capture_steps
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
            print(f"[Profiler] Stopped capture")


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


def get_memory_stats(device: int = 0) -> dict:
    """Get memory stats as a dict for logging."""
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / 1e9, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
    }


def reset_peak_memory(device: int = 0):
    """Reset peak memory stats for a fresh measurement window."""
    torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Tracks training metrics for logging and plotting.
    
    Usage:
        tracker = MetricTracker(log_dir="results")
        tracker.update(step=10, loss=0.5, reward_mean=0.3, kl=0.1)
        tracker.log(step=10)
        tracker.save()
    """

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

    def get_history(self, key: str) -> tuple[list, list]:
        """Get (steps, values) for a given metric."""
        entries = self.history.get(key, [])
        steps = [e["step"] for e in entries]
        values = [e["value"] for e in entries]
        return steps, values


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

class Timer:
    """Simple CUDA-aware timer.
    
    Usage:
        timer = Timer()
        with timer("forward"):
            logits = model(x)
        print(timer.summary())
    """

    def __init__(self, cuda_sync: bool = False):
        self.cuda_sync = cuda_sync
        self.times = defaultdict(list)

    @contextmanager
    def __call__(self, name: str):
        if self.cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.cuda_sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self.times[name].append(elapsed)

    def summary(self) -> str:
        lines = []
        for name, times in self.times.items():
            avg = sum(times) / len(times)
            total = sum(times)
            lines.append(f"  {name}: avg={avg*1000:.1f}ms total={total:.2f}s ({len(times)} calls)")
        return "\n".join(lines)

    def reset(self):
        self.times.clear()

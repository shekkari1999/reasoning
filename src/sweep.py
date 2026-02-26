"""
Parameter Sweep — Find optimal training config for 2×A100 40GB.

Runs short training bursts with different configs, measures:
  - Throughput (tokens/sec, samples/sec)
  - Peak GPU memory
  - Step wall-clock time breakdown (forward, backward, optimizer)
  - FSDP communication overhead

Produces a report and optionally Nsight Systems traces for the best config.

Usage:
    # Quick sweep (no profiling, just throughput + memory)
    torchrun --nproc_per_node=2 src/sweep.py --mode quick

    # Full sweep with NVTX annotations (run under nsys)
    nsys profile --trace=cuda,nvtx,nccl --output=profiles/sweep \
        torchrun --nproc_per_node=2 src/sweep.py --mode full

    # Single config test
    torchrun --nproc_per_node=2 src/sweep.py --mode single \
        --micro_batch 2 --accum_steps 4 --seq_len 1024
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import load_model, load_tokenizer, wrap_model_fsdp
from src.profiling_utils import nvtx_range, log_memory, get_memory_stats, Timer


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """A single config to benchmark."""
    micro_batch_size: int
    gradient_accumulation_steps: int
    seq_len: int
    label: str = ""

    @property
    def effective_batch(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return self.micro_batch_size * self.gradient_accumulation_steps * world_size

    @property
    def tokens_per_step(self):
        return self.effective_batch * self.seq_len

    def __post_init__(self):
        if not self.label:
            self.label = f"mb{self.micro_batch_size}_acc{self.gradient_accumulation_steps}_seq{self.seq_len}"


# SFT sweep configs — varying batch size and seq length
SFT_SWEEP_CONFIGS = [
    SweepConfig(micro_batch_size=1, gradient_accumulation_steps=8, seq_len=1024),
    SweepConfig(micro_batch_size=2, gradient_accumulation_steps=4, seq_len=1024),
    SweepConfig(micro_batch_size=4, gradient_accumulation_steps=2, seq_len=1024),
    SweepConfig(micro_batch_size=2, gradient_accumulation_steps=4, seq_len=512),
    SweepConfig(micro_batch_size=4, gradient_accumulation_steps=4, seq_len=512),
    SweepConfig(micro_batch_size=2, gradient_accumulation_steps=8, seq_len=1024),
]

# GRPO sweep configs — tighter memory (2 models in VRAM)
GRPO_SWEEP_CONFIGS = [
    SweepConfig(micro_batch_size=1, gradient_accumulation_steps=4, seq_len=512),
    SweepConfig(micro_batch_size=1, gradient_accumulation_steps=8, seq_len=512),
    SweepConfig(micro_batch_size=2, gradient_accumulation_steps=4, seq_len=512),
    SweepConfig(micro_batch_size=1, gradient_accumulation_steps=4, seq_len=768),
    SweepConfig(micro_batch_size=2, gradient_accumulation_steps=2, seq_len=512),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def create_dummy_batch(micro_batch_size: int, seq_len: int, vocab_size: int,
                       device: torch.device) -> dict:
    """Create a dummy training batch for benchmarking."""
    input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (micro_batch_size, seq_len), device=device)
    attention_mask = torch.ones(micro_batch_size, seq_len, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def benchmark_config(
    model: FSDP,
    config: SweepConfig,
    vocab_size: int,
    num_warmup: int = 3,
    num_steps: int = 10,
    profile_nvtx: bool = False,
) -> dict:
    """Benchmark a single config. Returns throughput and memory metrics."""
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    timer = Timer(cuda_sync=True)
    step_times = []
    oom = False

    try:
        for step in range(num_warmup + num_steps):
            is_measuring = step >= num_warmup

            if profile_nvtx and is_measuring:
                nvtx_tag = f"sweep_{config.label}_step_{step - num_warmup}"
            else:
                nvtx_tag = None

            if nvtx_tag:
                torch.cuda.nvtx.range_push(nvtx_tag)

            t_step_start = time.perf_counter()

            # Gradient accumulation loop
            for micro_step in range(config.gradient_accumulation_steps):
                batch = create_dummy_batch(
                    config.micro_batch_size, config.seq_len, vocab_size, device
                )

                # Forward
                if nvtx_tag:
                    torch.cuda.nvtx.range_push(f"forward_micro{micro_step}")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    logits = outputs.logits
                    loss = F.cross_entropy(
                        logits[:, :-1, :].reshape(-1, vocab_size),
                        batch["labels"][:, 1:].reshape(-1),
                    ) / config.gradient_accumulation_steps
                if nvtx_tag:
                    torch.cuda.nvtx.range_pop()

                # Backward
                if nvtx_tag:
                    torch.cuda.nvtx.range_push(f"backward_micro{micro_step}")
                loss.backward()
                if nvtx_tag:
                    torch.cuda.nvtx.range_pop()

            # Optimizer step
            if nvtx_tag:
                torch.cuda.nvtx.range_push("optimizer_step")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if nvtx_tag:
                torch.cuda.nvtx.range_pop()

            # Sync for accurate timing
            torch.cuda.synchronize()
            t_step_end = time.perf_counter()

            if is_measuring:
                step_times.append(t_step_end - t_step_start)

            if nvtx_tag:
                torch.cuda.nvtx.range_pop()

    except torch.cuda.OutOfMemoryError:
        oom = True
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"  OOM: {config.label}")

    # Gather results
    if oom:
        return {
            "config": config.label,
            "oom": True,
            "effective_batch": config.effective_batch,
            "tokens_per_step": config.tokens_per_step,
        }

    memory = get_memory_stats(device)
    avg_step_time = sum(step_times) / len(step_times)
    tokens_per_sec = config.tokens_per_step / avg_step_time
    samples_per_sec = config.effective_batch / avg_step_time

    result = {
        "config": config.label,
        "oom": False,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "seq_len": config.seq_len,
        "effective_batch": config.effective_batch,
        "tokens_per_step": config.tokens_per_step,
        "avg_step_time_s": round(avg_step_time, 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "samples_per_sec": round(samples_per_sec, 2),
        "peak_memory_gb": memory["peak_gb"],
        "allocated_memory_gb": memory["allocated_gb"],
        "memory_utilization_pct": round(memory["peak_gb"] / 40.0 * 100, 1),
        "num_steps_measured": len(step_times),
        "step_time_std": round(
            (sum((t - avg_step_time)**2 for t in step_times) / len(step_times)) ** 0.5,
            4
        ),
    }

    # Clean up optimizer states
    del optimizer
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    model_name: str,
    configs: list[SweepConfig],
    mode: str,
    num_warmup: int = 3,
    num_steps: int = 10,
    output_dir: str = "results",
) -> list[dict]:
    """Run sweep across all configs."""
    rank = dist.get_rank()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"PARAMETER SWEEP — {len(configs)} configs × {num_steps} steps each")
        print(f"{'='*70}")

    # Load model once
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    vocab_size = model.config.vocab_size

    results = []

    for i, config in enumerate(configs):
        if rank == 0:
            print(f"\n--- Config {i+1}/{len(configs)}: {config.label} ---")
            print(f"    effective_batch={config.effective_batch} | "
                  f"tokens/step={config.tokens_per_step:,}")

        # Re-wrap model with FSDP for each config
        # Need a fresh copy because FSDP modifies the model in place
        if i > 0:
            # Cleanup previous FSDP wrapper
            del fsdp_model
            torch.cuda.empty_cache()
            model = load_model(model_name)

        fsdp_model = wrap_model_fsdp(
            model,
            mixed_precision="bf16",
            activation_checkpointing=True,
            forward_prefetch=True,
        )

        result = benchmark_config(
            fsdp_model, config, vocab_size,
            num_warmup=num_warmup,
            num_steps=num_steps,
            profile_nvtx=(mode == "full"),
        )
        results.append(result)

        if rank == 0:
            if result["oom"]:
                print(f"    RESULT: OOM")
            else:
                print(f"    RESULT: {result['tokens_per_sec']:,.0f} tok/s | "
                      f"{result['samples_per_sec']:.1f} samples/s | "
                      f"peak={result['peak_memory_gb']:.1f}GB "
                      f"({result['memory_utilization_pct']}%) | "
                      f"step={result['avg_step_time_s']:.3f}s")

        dist.barrier()

    return results


def print_sweep_report(results: list[dict], rank: int = 0):
    """Print a formatted comparison table."""
    if rank != 0:
        return

    print(f"\n{'='*90}")
    print(f"SWEEP RESULTS")
    print(f"{'='*90}")
    print(f"{'Config':<35} {'tok/s':>10} {'samp/s':>8} {'step(s)':>8} "
          f"{'peak GB':>8} {'mem%':>6} {'OOM':>5}")
    print(f"{'-'*90}")

    # Sort by throughput (non-OOM first)
    valid = [r for r in results if not r["oom"]]
    oom = [r for r in results if r["oom"]]
    valid.sort(key=lambda r: r["tokens_per_sec"], reverse=True)

    for r in valid:
        print(f"{r['config']:<35} {r['tokens_per_sec']:>10,.0f} "
              f"{r['samples_per_sec']:>8.1f} {r['avg_step_time_s']:>8.3f} "
              f"{r['peak_memory_gb']:>8.1f} {r['memory_utilization_pct']:>5.1f}% "
              f"{'':>5}")

    for r in oom:
        print(f"{r['config']:<35} {'—':>10} {'—':>8} {'—':>8} "
              f"{'—':>8} {'—':>6} {'OOM':>5}")

    if valid:
        best = valid[0]
        print(f"\n{'='*90}")
        print(f"RECOMMENDED CONFIG: {best['config']}")
        print(f"  Throughput: {best['tokens_per_sec']:,.0f} tokens/sec")
        print(f"  Peak memory: {best['peak_memory_gb']:.1f} GB ({best['memory_utilization_pct']}%)")
        print(f"  Step time: {best['avg_step_time_s']:.3f}s ± {best.get('step_time_std', 0):.3f}s")
        print(f"{'='*90}")

        # Memory headroom analysis
        headroom = 40.0 - best["peak_memory_gb"]
        print(f"\n  Memory headroom: {headroom:.1f} GB")
        if headroom < 3:
            print(f"  ⚠ Tight. Consider reducing batch size for GRPO (2 models in memory).")
        elif headroom < 8:
            print(f"  ✓ Good for SFT. May be tight for GRPO — run GRPO sweep separately.")
        else:
            print(f"  ✓ Plenty of room. Could increase batch size further.")


# ---------------------------------------------------------------------------
# GRPO-specific sweep (2 models in memory)
# ---------------------------------------------------------------------------

def run_grpo_memory_test(
    model_name: str,
    configs: list[SweepConfig],
    output_dir: str = "results",
) -> list[dict]:
    """Test GRPO memory usage — loads 2 models (policy + reference)."""
    rank = dist.get_rank()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"GRPO MEMORY SWEEP — 2 models (policy + frozen ref)")
        print(f"{'='*70}")

    results = []

    for i, config in enumerate(configs):
        if rank == 0:
            print(f"\n--- Config {i+1}/{len(configs)}: {config.label} ---")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Load policy model
            policy = load_model(model_name)
            policy = wrap_model_fsdp(policy, activation_checkpointing=True)

            # Load reference model (frozen)
            ref = load_model(model_name)
            for p in ref.parameters():
                p.requires_grad = False
            ref = wrap_model_fsdp(ref, activation_checkpointing=False)
            ref.eval()

            mem_after_models = get_memory_stats()

            # Simulate forward passes for both
            device = torch.cuda.current_device()
            vocab_size = policy.module.config.vocab_size if hasattr(policy, 'module') else 151936

            batch = create_dummy_batch(config.micro_batch_size, config.seq_len, vocab_size, device)

            # Policy forward + backward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = policy(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = F.cross_entropy(
                    out.logits[:, :-1, :].reshape(-1, vocab_size),
                    batch["labels"][:, 1:].reshape(-1),
                )
            loss.backward()

            # Ref forward (no grad)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    ref_out = ref(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            torch.cuda.synchronize()
            peak_mem = get_memory_stats()

            result = {
                "config": config.label,
                "oom": False,
                "peak_memory_gb": peak_mem["peak_gb"],
                "memory_after_model_load_gb": mem_after_models["peak_gb"],
                "memory_utilization_pct": round(peak_mem["peak_gb"] / 40.0 * 100, 1),
                "headroom_gb": round(40.0 - peak_mem["peak_gb"], 1),
            }

            del policy, ref
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            result = {"config": config.label, "oom": True}
            torch.cuda.empty_cache()

        results.append(result)

        if rank == 0:
            if result["oom"]:
                print(f"    OOM")
            else:
                print(f"    peak={result['peak_memory_gb']:.1f}GB "
                      f"({result['memory_utilization_pct']}%) "
                      f"headroom={result['headroom_gb']:.1f}GB")

        dist.barrier()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "full", "single", "grpo"],
                        help="quick=throughput only, full=with NVTX, "
                             "single=one config, grpo=memory test with 2 models")
    parser.add_argument("--micro_batch", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # Init distributed
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        configs = [SweepConfig(args.micro_batch, args.accum_steps, args.seq_len)]
    elif args.mode == "grpo":
        configs = GRPO_SWEEP_CONFIGS
    else:
        configs = SFT_SWEEP_CONFIGS

    # Run sweep
    if args.mode == "grpo":
        results = run_grpo_memory_test(args.model, configs, args.output_dir)
    else:
        results = run_sweep(
            args.model, configs, args.mode,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
        )

    # Report
    if args.mode != "grpo":
        print_sweep_report(results, rank)

    # Save
    if rank == 0:
        sweep_file = output_dir / f"sweep_{args.mode}_results.json"
        with open(sweep_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep results saved to {sweep_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

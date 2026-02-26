"""
Model loading and FSDP wrapping for Qwen-2.5-3B.

Provides:
  - load_model(): Load Qwen with proper dtype
  - wrap_model_fsdp(): Wrap with FSDP, mixed precision, activation checkpointing
  - load_reference_model(): Frozen copy for GRPO KL penalty
"""

import functools
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.profiling_utils import log_memory


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name: str = "Qwen/Qwen2.5-3B",
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[str] = None,
) -> AutoModelForCausalLM:
    """Load Qwen model. Optionally from a local checkpoint.
    
    Args:
        model_name: HuggingFace model name
        dtype: Model dtype (bf16 for A100)
        checkpoint_path: Local path to saved checkpoint (overrides model_name)
    
    Returns:
        Model on CPU (will be moved to GPU by FSDP wrapping)
    """
    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {params:.2f}B parameters, dtype={dtype}")
    return model


def load_tokenizer(model_name: str = "Qwen/Qwen2.5-3B") -> AutoTokenizer:
    """Load tokenizer with proper padding config."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation
    return tokenizer


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def _get_qwen_wrap_policy(model):
    """Get auto wrap policy targeting Qwen transformer layers.
    
    FSDP wraps at the granularity of transformer blocks — each block
    is a separate FSDP unit that gets sharded independently.
    """
    # Qwen2.5 uses Qwen2DecoderLayer as the transformer block
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    return auto_wrap_policy


def get_mixed_precision_policy(dtype: str = "bf16") -> MixedPrecision:
    """Create FSDP mixed precision policy."""
    if dtype == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif dtype == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        return None  # fp32, no mixed precision


def wrap_model_fsdp(
    model: AutoModelForCausalLM,
    mixed_precision: str = "bf16",
    activation_checkpointing: bool = True,
    forward_prefetch: bool = True,
    sync_module_states: bool = True,
) -> FSDP:
    """Wrap model with FSDP for distributed training.
    
    Args:
        model: HuggingFace model (on CPU)
        mixed_precision: "bf16", "fp16", or "fp32"
        activation_checkpointing: Enable gradient checkpointing (saves memory)
        forward_prefetch: Prefetch next FSDP unit during forward (overlaps comm)
        sync_module_states: Broadcast params from rank 0 (ensures consistency)
    
    Returns:
        FSDP-wrapped model on local GPU
    """
    local_rank = dist.get_rank()
    
    # Enable activation checkpointing BEFORE FSDP wrapping
    if activation_checkpointing:
        model.gradient_checkpointing_enable()
        print(f"[Rank {local_rank}] Activation checkpointing enabled")

    mp_policy = get_mixed_precision_policy(mixed_precision)
    wrap_policy = _get_qwen_wrap_policy(model)

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        forward_prefetch=forward_prefetch,
        sync_module_states=sync_module_states,
        limit_all_gathers=True,  # memory efficient — one all-gather at a time
    )

    if local_rank == 0:
        log_memory("after FSDP wrap")

    return model


# ---------------------------------------------------------------------------
# Reference model (for GRPO KL penalty)
# ---------------------------------------------------------------------------

def load_reference_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    mixed_precision: str = "bf16",
) -> FSDP:
    """Load a frozen reference model for KL divergence computation.
    
    The reference model is FSDP-wrapped (for sharding) but has:
      - No gradient computation
      - No optimizer states
      - Costs only ~3GB/GPU for 3B model
    """
    model = load_model(model_name_or_path, dtype=dtype)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Wrap with FSDP for memory-efficient sharding (no grad, no optim states)
    model = wrap_model_fsdp(
        model,
        mixed_precision=mixed_precision,
        activation_checkpointing=False,  # not needed — no backward pass
        forward_prefetch=True,
        sync_module_states=True,
    )

    model.eval()
    print("Reference model loaded and frozen")
    return model


# ---------------------------------------------------------------------------
# Checkpoint saving/loading
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str,
    rank: int = 0,
):
    """Save FSDP model checkpoint.
    
    Uses FULL_STATE_DICT — gathers sharded params to rank 0 and saves.
    Only rank 0 actually writes to disk.
    """
    from pathlib import Path
    save_dir = Path(output_dir) / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Gather full state dict to rank 0
    full_sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_sd_config):
        state_dict = model.state_dict()

    if rank == 0:
        # Save as HuggingFace format for easy loading
        # Need to unwrap FSDP to get the underlying model for save_pretrained
        torch.save(state_dict, save_dir / "pytorch_model.bin")
        print(f"[Rank 0] Checkpoint saved to {save_dir}")

    # Barrier to ensure save completes before any rank continues
    dist.barrier()


def save_hf_checkpoint(
    model: FSDP,
    tokenizer: AutoTokenizer,
    step: int,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-3B",
    rank: int = 0,
):
    """Save as HuggingFace-compatible checkpoint for easy eval/loading."""
    from pathlib import Path
    save_dir = Path(output_dir) / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    full_sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_sd_config):
        state_dict = model.state_dict()

    if rank == 0:
        # Load a fresh model on CPU and inject the state dict
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        hf_model.load_state_dict(state_dict)
        hf_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[Rank 0] HF checkpoint saved to {save_dir}")

    dist.barrier()

"""
Data loading and tokenization for SFT and RL training.

Provides:
  - SFTDataset: Tokenized chain-of-thought traces for supervised fine-tuning
  - RLPromptDataset: Prompts only (no solutions) for GRPO/Dr.GRPO/DAPO
  - Collator functions for batching
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems step by step. "
    "Show your reasoning inside <think>...</think> tags, then give your "
    "final answer inside <answer>...</answer> tags."
)


def format_sft_example(question: str, reasoning: str, answer: str) -> str:
    """Format a single SFT training example.
    
    Target format:
        <think>
        {reasoning}
        </think>
        <answer>{answer}</answer>
    """
    return f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>"


def format_sft_prompt(question: str) -> str:
    """Format the prompt (input) for SFT."""
    return f"{SYSTEM_PROMPT}\n\nProblem: {question}\n\nSolution:"


class SFTDataset(Dataset):
    """Dataset for chain-of-thought supervised fine-tuning.
    
    Each example is tokenized as:
        [prompt tokens] [completion tokens]
    
    Labels are set to -100 for prompt tokens (don't compute loss on prompt).
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "open-r1/OpenR1-Math-220k",
        subset: str = "default",
        split: str = "train",
        max_samples: int = 10000,
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print(f"Loading SFT dataset: {dataset_name} ({split}, max_samples={max_samples})")
        raw = load_dataset(dataset_name, subset, split=split)

        if max_samples > 0 and len(raw) > max_samples:
            indices = random.sample(range(len(raw)), max_samples)
            raw = raw.select(indices)

        self.examples = self._process_dataset(raw, dataset_name)
        print(f"SFT dataset ready: {len(self.examples)} examples")

    def _process_dataset(self, raw, dataset_name: str) -> list[dict]:
        """Process raw HF dataset into tokenized examples."""
        examples = []

        for row in raw:
            try:
                prompt, completion = self._extract_prompt_completion(row, dataset_name)
            except (KeyError, ValueError, IndexError):
                continue

            if prompt is None or completion is None:
                continue

            # Tokenize prompt and completion separately
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

            # Add EOS token at the end of completion
            completion_ids = completion_ids + [self.tokenizer.eos_token_id]

            # Truncate to max_seq_len
            total_len = len(prompt_ids) + len(completion_ids)
            if total_len > self.max_seq_len:
                # Truncate completion, keep full prompt
                max_completion = self.max_seq_len - len(prompt_ids)
                if max_completion < 32:  # too short to be useful
                    continue
                completion_ids = completion_ids[:max_completion]

            input_ids = prompt_ids + completion_ids

            # Labels: -100 for prompt tokens (no loss), actual ids for completion
            labels = [-100] * len(prompt_ids) + completion_ids

            examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        return examples

    def _extract_prompt_completion(self, row: dict, dataset_name: str) -> tuple:
        """Extract prompt and completion from a dataset row.
        
        Handles multiple dataset formats.
        """
        if "open-r1" in dataset_name.lower() or "openr1" in dataset_name.lower():
            # OpenR1-Math format: has 'problem' and 'solution' or similar
            # Check for messages format first
            if "messages" in row:
                messages = row["messages"]
                question = None
                solution = None
                for msg in messages:
                    if msg["role"] == "user":
                        question = msg["content"]
                    elif msg["role"] == "assistant":
                        solution = msg["content"]
                if question and solution:
                    prompt = format_sft_prompt(question)
                    return prompt, solution

            # Direct fields
            question = row.get("problem", row.get("question", None))
            solution = row.get("solution", row.get("answer", None))
            if question and solution:
                prompt = format_sft_prompt(question)
                return prompt, solution

        elif "gsm8k" in dataset_name.lower():
            question = row["question"]
            raw_answer = row["answer"]
            # GSM8K has CoT in the answer field, final answer after ####
            prompt = format_sft_prompt(question)
            # Wrap in our format
            parts = raw_answer.split("####")
            if len(parts) == 2:
                reasoning = parts[0].strip()
                answer = parts[1].strip()
                completion = format_sft_example(question, reasoning, answer)
                return prompt, completion

        raise ValueError(f"Could not extract from row with keys: {row.keys()}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SFTCollator:
    """Collator that pads SFT examples to the same length within a batch."""

    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 1024):
        self.pad_id = tokenizer.pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[dict]) -> dict:
        max_len = min(
            max(len(ex["input_ids"]) for ex in batch),
            self.max_seq_len,
        )

        input_ids = []
        labels = []
        attention_mask = []

        for ex in batch:
            ids = ex["input_ids"][:max_len]
            labs = ex["labels"][:max_len]

            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# RL Prompt Dataset (for GRPO / Dr. GRPO / DAPO)
# ---------------------------------------------------------------------------

class RLPromptDataset(Dataset):
    """Dataset that yields prompts only (no solutions) for RL training.
    
    During RL, we:
      1. Sample prompts from this dataset
      2. Generate G completions per prompt (rollouts)
      3. Score completions with reward function
      4. Compute policy gradient
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "openai/gsm8k",
        split: str = "train",
        max_samples: int = 0,
        max_prompt_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

        print(f"Loading RL prompts: {dataset_name} ({split})")
        raw = load_dataset(dataset_name, "main", split=split)

        if max_samples > 0 and len(raw) > max_samples:
            indices = random.sample(range(len(raw)), max_samples)
            raw = raw.select(indices)

        self.examples = []
        for row in raw:
            question = row.get("question", row.get("problem", ""))
            answer = row.get("answer", "")
            prompt = format_sft_prompt(question)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[:max_prompt_len]

            self.examples.append({
                "prompt_ids": prompt_ids,
                "prompt_text": prompt,
                "question": question,
                "raw_answer": answer,  # for reward computation
            })

        print(f"RL prompt dataset ready: {len(self.examples)} prompts")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RLPromptCollator:
    """Collator for RL prompts — pads prompt_ids to same length."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(ex["prompt_ids"]) for ex in batch)

        prompt_ids = []
        attention_mask = []

        for ex in batch:
            ids = ex["prompt_ids"]
            pad_len = max_len - len(ids)
            # Left-pad for generation
            prompt_ids.append([self.pad_id] * pad_len + ids)
            attention_mask.append([0] * pad_len + [1] * len(ids))

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "questions": [ex["question"] for ex in batch],
            "raw_answers": [ex["raw_answer"] for ex in batch],
            "prompt_texts": [ex["prompt_text"] for ex in batch],
        }


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------

def create_sft_dataloader(
    tokenizer: AutoTokenizer,
    dataset_name: str = "open-r1/OpenR1-Math-220k",
    subset: str = "default",
    split: str = "train",
    max_samples: int = 10000,
    max_seq_len: int = 1024,
    micro_batch_size: int = 2,
    num_workers: int = 4,
    distributed: bool = True,
) -> DataLoader:
    """Create DataLoader for SFT training."""
    dataset = SFTDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        max_samples=max_samples,
        max_seq_len=max_seq_len,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    collator = SFTCollator(tokenizer, max_seq_len=max_seq_len)

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def create_rl_dataloader(
    tokenizer: AutoTokenizer,
    dataset_name: str = "openai/gsm8k",
    split: str = "train",
    max_samples: int = 0,
    max_prompt_len: int = 256,
    batch_size: int = 4,
    num_workers: int = 4,
    distributed: bool = True,
) -> DataLoader:
    """Create DataLoader for RL training (prompts only)."""
    dataset = RLPromptDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        max_prompt_len=max_prompt_len,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    collator = RLPromptCollator(tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

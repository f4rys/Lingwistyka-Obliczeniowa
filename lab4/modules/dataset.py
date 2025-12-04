from __future__ import annotations

from typing import Callable, Iterator, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_name_or_path: str = "gpt2") -> PreTrainedTokenizerBase:
    """Build a Hugging Face tokenizer with proper padding token."""
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def shift_labels(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Shift labels for next-token prediction."""
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id
    return labels


def make_lm_collate(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
) -> Callable:
    """Create a collate function for language modeling."""
    def collate(texts):
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        return input_ids, labels, attention_mask
    return collate


def stream_batches(
    tokenizer,
    batch_size: int,
    max_length: int,
    split: str = "train",
    pad_token_id: Optional[int] = None,
    question_field: str = "Body",
    max_batches: Optional[int] = None,
    skip_samples: int = 0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Stream batches from the StackOverflow dataset.
    
    Note: This dataset only has a 'train' split. For evaluation purposes,
    use skip_samples to skip the training portion and get different samples.
    """
    collate = make_lm_collate(tokenizer, max_length=max_length)
    # Always use 'train' split since that's all this dataset has
    stream = load_dataset("mikex86/stackoverflow-posts", split="train", streaming=True)
    texts = []
    batch_count = 0
    samples_skipped = 0
    
    for ex in stream:
        # Skip samples for pseudo-validation split
        if samples_skipped < skip_samples:
            samples_skipped += 1
            continue
        text = ex.get(question_field) or ""
        if text:
            texts.append(text)
            if len(texts) >= batch_size:
                input_ids, labels, attention_mask = collate(texts)
                labels = shift_labels(input_ids, pad_id=pad_token_id or 0)
                yield input_ids, labels, attention_mask
                texts = []
                batch_count += 1
                if max_batches and batch_count >= max_batches:
                    return
    
    if texts and (max_batches is None or batch_count < max_batches):
        input_ids, labels, attention_mask = collate(texts)
        labels = shift_labels(input_ids, pad_id=pad_token_id or 0)
        yield input_ids, labels, attention_mask


__all__ = ["build_tokenizer", "shift_labels", "make_lm_collate", "stream_batches"]

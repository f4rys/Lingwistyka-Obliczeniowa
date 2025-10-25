from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_name_or_path: str = "gpt2") -> PreTrainedTokenizerBase:
    """
    Build a Hugging Face tokenizer. If the tokenizer has no pad token, reuse EOS as PAD
    to avoid changing vocab_size (keeps the model config consistent).
    """
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token_id is None:
        # Prefer reusing EOS over adding a new token to keep vocab_size stable
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            # As a fallback, add a [PAD] token, increasing vocab_size
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def make_lm_collate(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
) -> Callable[[Sequence[str]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Returns a collate function that tokenizes a batch of raw strings into
    (input_ids, labels, attention_mask) tensors.
    Labels are identical to input_ids for a simple next-token objective; pads are ignored via loss ignore_index.
    """

    def collate(texts: Sequence[str]):
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


__all__ = ["build_tokenizer", "make_lm_collate"]

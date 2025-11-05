from __future__ import annotations
from typing import Callable, Sequence, Tuple
import torch


def make_lm_collate(
    tokenizer: Callable[..., dict],
    max_length: int = 256,
) -> Callable[[Sequence[str]], Tuple[torch.Tensor, torch.Tensor]]:
    """Create a simple collate function for language modelling."""
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
        return input_ids, attention_mask

    return collate


__all__ = ["make_lm_collate"]
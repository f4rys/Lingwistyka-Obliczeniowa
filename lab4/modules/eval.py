from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .dataset import stream_batches


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    max_examples: int = 2000,
    batch_size: int = 16,
    max_length: int = 256,
    use_bf16: bool = False,
    skip_samples: int = 50000,  # Skip training samples to get "validation" data
) -> Tuple[float, float]:
    """Evaluate perplexity on validation data.
    
    Since the dataset only has a 'train' split, we skip the first `skip_samples`
    samples to use a different portion for evaluation.
    """
    device = next(model.parameters()).device
    model.eval()
    
    pad_id = tokenizer.pad_token_id or 0
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = max_examples // batch_size
    
    for step, (input_ids, labels, attention_mask) in enumerate(
        stream_batches(
            tokenizer, batch_size, max_length, 
            split="train",  # Only split available
            pad_token_id=pad_id, 
            max_batches=num_batches,
            skip_samples=skip_samples,  # Skip to get different samples
        )
    ):
        if step >= num_batches:
            break
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        if use_bf16:
            with autocast(dtype=torch.bfloat16):
                logits, _ = model(input_ids, attention_mask=attention_mask, targets=None)
        else:
            logits, _ = model(input_ids, attention_mask=attention_mask, targets=None)
        
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=pad_id,
            reduction="sum"
        )
        n_tokens = (labels.view(-1) != pad_id).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


__all__ = ["evaluate_perplexity"]

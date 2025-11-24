from __future__ import annotations

import time
import torch
from .device import sync_if_needed


@torch.no_grad()
def measure_throughput(model, vocab_size: int, seq_len: int = 256, batch_size: int = 8, iters: int = 30) -> float:
    device = next(model.parameters()).device
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    for _ in range(5):
        model(x)

    sync_if_needed(device)
    t0 = time.time()

    for _ in range(iters):
        model(x)

    sync_if_needed(device)
    t1 = time.time()

    tokens = batch_size * seq_len * iters
    tok_per_s = tokens / max(t1 - t0, 1e-9)
    return tok_per_s


@torch.no_grad()
def measure_generation_latency(
    model,
    vocab_size: int,
    *,
    prompt_len: int = 16,
    new_tokens: int = 64,
    iters: int = 10,
) -> float:
    device = next(model.parameters()).device
    prompt = torch.randint(0, vocab_size, (1, prompt_len),
                           device=device, dtype=torch.long)
    # Assuming model has generate method
    if hasattr(model, 'generate'):
        model.generate(prompt, max_new_tokens=8)

        sync_if_needed(device)
        t0 = time.time()
        for _ in range(iters):
            model.generate(prompt, max_new_tokens=new_tokens)

        sync_if_needed(device)
        t1 = time.time()
        total_gen = new_tokens * iters
        ms_per_token = ((t1 - t0) / max(total_gen, 1)) * 1000.0
        return ms_per_token
    return 0.0


def measure_training_step_time(
    model,
    vocab_size: int,
    *,
    seq_len: int = 256,
    batch_size: int = 8,
    iters: int = 30,
    lr: float = 1e-3,
) -> float:
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    # For classification, targets would be different, but for step time measurement it doesn't matter much
    # We'll assume the model can handle dummy inputs or we adapt.
    # If model is classifier, it expects targets of shape (B,)
    # If model is LM, it expects targets of shape (B, T) or None (if it computes loss internally with targets arg)
    
    # Let's try to infer or just use a try-except block or assume LM for now.
    # Actually, for this lab, we have classifiers.
    # Let's make a robust dummy step.
    
    try:
        # Try classification style
        y = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.long)
        _, loss = model(x, targets=y)
    except Exception:
        # Try LM style
        y = x.clone()
        _, loss = model(x, targets=y)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        # Re-run forward to be safe
        try:
             y = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.long)
             _, loss = model(x, targets=y)
        except Exception:
             y = x.clone()
             _, loss = model(x, targets=y)
             
        loss.backward()
        optimizer.step()

    sync_if_needed(device)
    t0 = time.time()

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        try:
             y = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.long)
             _, loss = model(x, targets=y)
        except Exception:
             y = x.clone()
             _, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()

    sync_if_needed(device)
    t1 = time.time()

    ms_per_step = ((t1 - t0) / max(iters, 1)) * 1000.0
    return ms_per_step


__all__ = [
    "measure_throughput",
    "measure_generation_latency",
    "measure_training_step_time",
]

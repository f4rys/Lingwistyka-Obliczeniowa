from __future__ import annotations

from typing import Optional

import torch
from .device import best_device


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    device: Optional[torch.device] = None,
) -> str:
    device = device or best_device()
    model = model.to(device).eval()
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    out_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_id,
    )
    text = tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return text

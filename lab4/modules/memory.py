from __future__ import annotations

import gc
import math
from typing import Callable, Dict

import torch
from torch.cuda.amp import autocast

from .device import DEVICE


def clear_cuda_memory():
    """Aggressively clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


class MemoryTracker:
    """Track GPU memory usage during training."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.forward_memory = 0
        self.backward_memory = 0
        self.peak_memory = 0
    
    def record_forward(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.forward_memory = torch.cuda.memory_allocated() / 1e9
    
    def record_backward(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.backward_memory = torch.cuda.memory_allocated() / 1e9
    
    def record_peak(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    def get_current_memory(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0
    
    def get_peak_memory(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        return 0
    
    def summary(self) -> Dict:
        return {
            "forward_memory_gb": round(self.forward_memory, 3),
            "backward_memory_gb": round(self.backward_memory, 3),
            "peak_memory_gb": round(self.peak_memory, 3),
        }


def find_max_batch_size(
    model_fn: Callable,
    tokenizer,
    max_length: int = 256,
    use_bf16: bool = False,
    start_batch: int = 1,
    max_batch: int = 256,
) -> int:
    """Find the maximum batch size that fits in GPU memory."""
    device = DEVICE
    working_batch_size = start_batch
    
    for batch_size in [2**i for i in range(int(math.log2(start_batch)), int(math.log2(max_batch)) + 1)]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = model_fn()
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Create dummy data
            input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, max_length), device=device)
            labels = torch.randint(0, tokenizer.vocab_size, (batch_size, max_length), device=device)
            attention_mask = torch.ones(batch_size, max_length, device=device)
            
            # Forward and backward pass
            model.train()
            optimizer.zero_grad()
            
            if use_bf16:
                with autocast(dtype=torch.bfloat16):
                    logits, loss = model(input_ids, attention_mask=attention_mask, targets=labels)
                loss.backward()
            else:
                logits, loss = model(input_ids, attention_mask=attention_mask, targets=labels)
                loss.backward()
            
            optimizer.step()
            torch.cuda.synchronize()
            
            working_batch_size = batch_size
            print(f"  Batch size {batch_size}: OK (Peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")
            
            del model, optimizer, input_ids, labels, attention_mask, logits, loss
            clear_cuda_memory()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch size {batch_size}: OOM")
                # Clean up any partial allocations
                try:
                    del model, optimizer, input_ids, labels, attention_mask
                except:
                    pass
                clear_cuda_memory()
                break
            else:
                raise e
    
    clear_cuda_memory()
    return working_batch_size


__all__ = ["MemoryTracker", "find_max_batch_size", "clear_cuda_memory"]

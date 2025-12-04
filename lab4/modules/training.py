from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .dataset import stream_batches
from .memory import MemoryTracker


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 16
    max_length: int = 256
    steps_per_epoch: int = 2000
    num_epochs: int = 1
    lr: float = 3e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    
    # Memory optimization flags
    use_bf16: bool = False
    use_flash_attention: bool = False
    use_windowed_attention: bool = False
    window_size: int = 128
    gradient_checkpointing: bool = False


def train_one_epoch(
    model: nn.Module,
    tokenizer,
    config: TrainingConfig,
    device: torch.device,
    memory_tracker: Optional[MemoryTracker] = None,
) -> Dict:
    """Train the model for one epoch and return metrics."""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    # Warmup scheduler
    total_steps = config.steps_per_epoch
    warmup_steps = min(config.warmup_steps, total_steps)
    
    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min((step + 1) / float(warmup_steps), 1.0)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    pad_id = tokenizer.pad_token_id or 0
    
    # Metrics
    total_loss = 0.0
    step_times = []
    
    if memory_tracker:
        memory_tracker.reset()
    
    # Training loop
    pbar = tqdm(
        stream_batches(
            tokenizer, 
            config.batch_size, 
            config.max_length, 
            split="train", 
            pad_token_id=pad_id,
            max_batches=config.steps_per_epoch
        ),
        total=config.steps_per_epoch,
        desc="Training"
    )
    
    for step, (input_ids, labels, attention_mask) in enumerate(pbar):
        if step >= config.steps_per_epoch:
            break
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        
        step_start = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        if config.use_bf16:
            with autocast(dtype=torch.bfloat16):
                logits, loss = model(input_ids, attention_mask=attention_mask, targets=labels)
            loss.backward()
        else:
            logits, loss = model(input_ids, attention_mask=attention_mask, targets=labels)
            loss.backward()
        
        if memory_tracker and step == 0:
            memory_tracker.record_forward()
        
        # Record memory after first backward
        if memory_tracker and step == 0:
            memory_tracker.record_backward()
        
        if config.grad_clip > 0:
            clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        total_loss += loss.item()
        
        if step % 100 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'step_time': f'{step_time:.3f}s'
            })
    
    if memory_tracker:
        memory_tracker.record_peak()
    
    avg_loss = total_loss / config.steps_per_epoch
    avg_step_time = sum(step_times[20:]) / len(step_times[20:]) if len(step_times) > 20 else sum(step_times) / len(step_times)
    total_time = sum(step_times)
    
    return {
        "avg_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "avg_step_time": avg_step_time,
        "total_time": total_time,
        "steps": config.steps_per_epoch,
    }


__all__ = ["TrainingConfig", "train_one_epoch"]

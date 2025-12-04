from __future__ import annotations

import gc
from typing import Dict

import torch

from .device import DEVICE
from .transformer import TransformerConfig, TransformerLanguageModel, count_parameters
from .training import TrainingConfig, train_one_epoch
from .memory import MemoryTracker, find_max_batch_size, clear_cuda_memory
from .eval import evaluate_perplexity


def run_experiment(
    experiment_name: str,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    tokenizer,
    find_max_bs: bool = True,
) -> Dict:
    """Run a complete training experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    device = DEVICE
    results = {"name": experiment_name}
    
    # Clear memory before starting
    clear_cuda_memory()
    
    # Create model factory
    def create_model():
        return TransformerLanguageModel(model_config)
    
    # Find maximum batch size if requested
    if find_max_bs and torch.cuda.is_available():
        print("\nFinding maximum batch size...")
        # Limit max batch based on available GPU memory
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Rough heuristic: smaller GPUs need smaller max batch to test
        max_batch = min(256, int(32 * gpu_mem_gb / 4))  # Scale with GPU memory
        max_bs = find_max_batch_size(
            create_model,
            tokenizer,
            max_length=training_config.max_length,
            use_bf16=training_config.use_bf16,
            start_batch=2,  # Start smaller
            max_batch=max_batch,
        )
        print(f"Maximum batch size: {max_bs}")
        results["max_batch_size"] = max_bs
        training_config.batch_size = max_bs
    else:
        results["max_batch_size"] = training_config.batch_size
    
    # Create model and move to device
    clear_cuda_memory()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    model = create_model()
    
    # Enable gradient checkpointing if specified
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"BF16: {training_config.use_bf16}")
    print(f"Flash Attention: {training_config.use_flash_attention}")
    print(f"Windowed Attention: {training_config.use_windowed_attention}")
    print(f"Gradient Checkpointing: {training_config.gradient_checkpointing}")
    
    # Memory tracker
    memory_tracker = MemoryTracker()
    
    # Train for one epoch
    print("\nTraining...")
    train_results = train_one_epoch(
        model,
        tokenizer,
        training_config,
        device,
        memory_tracker,
    )
    
    results.update(train_results)
    results.update(memory_tracker.summary())
    
    # Evaluate perplexity
    print("\nEvaluating perplexity on validation set...")
    val_ppl, val_loss = evaluate_perplexity(
        model,
        tokenizer,
        max_examples=2000,
        batch_size=min(training_config.batch_size, 32),
        max_length=training_config.max_length,
        use_bf16=training_config.use_bf16,
    )
    
    results["val_perplexity"] = val_ppl
    results["val_loss"] = val_loss
    
    # Cleanup
    del model
    clear_cuda_memory()
    
    print(f"\nResults for {experiment_name}:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return results


__all__ = ["run_experiment"]

from __future__ import annotations

import torch


def has_mps() -> bool:
    try:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def best_device() -> torch.device:
    """Pick CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        try:
            # Test CUDA by creating a small tensor
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            print("Warning: CUDA available but failed to initialize. Using CPU.")
            return torch.device("cpu")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_set_tf32() -> None:
    """Enable TF32 for CUDA operations if available."""
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def reset_cuda() -> None:
    """Reset CUDA state - useful after errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


# Global device
DEVICE = best_device()

# Enable TF32 by default
maybe_set_tf32()

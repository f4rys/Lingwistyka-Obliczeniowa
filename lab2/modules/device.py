from __future__ import annotations

import torch


def has_mps() -> bool:
    try:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
    except Exception:
        return False


def best_device() -> torch.device:
    """Pick CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_mps():
        return torch.device("mps")
    return torch.device("cpu")


def sync_if_needed(device: torch.device) -> None:
    """Synchronize GPU if supported to get accurate timings."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def maybe_set_tf32() -> None:
    """Try enabling matmul optimizations if available (no-op on MPS)."""
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass


__all__ = [
    "has_mps",
    "best_device",
    "sync_if_needed",
    "maybe_set_tf32",
]

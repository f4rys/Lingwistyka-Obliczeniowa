import torch

def shift_labels(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id
    return labels


__all__ = [
    "shift_labels"
]

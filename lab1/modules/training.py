from __future__ import annotations

import os
import json
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from .device import best_device, maybe_set_tf32
from datasets import load_dataset


def shift_labels(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id
    return labels


def latest_checkpoint(dir_path: str) -> Optional[str]:
    if not os.path.isdir(dir_path):
        return None
    files = [f for f in os.listdir(dir_path) if f.endswith(".pt")]
    if not files:
        return None
    if "last.pt" in files:
        return os.path.join(dir_path, "last.pt")
    epochs: list[tuple[int, str]] = []
    for f in files:
        if f.startswith("epoch_") and f.endswith(".pt"):
            try:
                e = int(f.split("_")[1].split(".")[0])
                epochs.append((e, f))
            except Exception:
                pass
    if not epochs:
        return None
    epochs.sort()
    return os.path.join(dir_path, epochs[-1][1])


def stream_batches(
    tokenizer,
    batch_size: int,
    max_length: int,
    split: str = "train",
    pad_token_id: Optional[int] = None,
    question_field: str = "Body",
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    from data.dataset import make_lm_collate

    collate = make_lm_collate(tokenizer, max_length=max_length)
    stream = load_dataset("mikex86/stackoverflow-posts", split=split, streaming=True)
    texts: list[str] = []
    for ex in stream:
        text = ex.get(question_field) or ""
        if text:
            texts.append(text)
            if len(texts) >= batch_size:
                input_ids, labels, attention_mask = collate(texts)
                labels = shift_labels(input_ids, pad_id=pad_token_id or 0)
                yield input_ids, labels, attention_mask
                texts = []
    if texts:
        input_ids, labels, attention_mask = collate(texts)
        labels = shift_labels(input_ids, pad_id=pad_token_id or 0)
        yield input_ids, labels, attention_mask


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: Dict,
    tokenizer_name: str,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "tokenizer": tokenizer_name,
        },
        path,
    )


def save_final_model(
    out_dir: str,
    model: torch.nn.Module,
    config: Dict,
    tokenizer_name: str,
    weights_name: str = "final.pt",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, weights_name)
    torch.save(model.state_dict(), weights_path)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer": tokenizer_name}, f, indent=2)
    return weights_path


def train_streamed_lm(
    model: torch.nn.Module,
    tokenizer,
    config: Dict,
    *,
    ckpt_dir: str,
    final_dir: str,
    batch_size: int = 16,
    max_length: int = 256,
    steps_per_epoch: int = 2000,
    num_epochs: int = 3,
    save_every: int = 1,
    lr: float = 3e-4,
    warmup_steps: int = 1000,
    grad_clip: Optional[float] = 1.0,
    device: Optional[torch.device] = None,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    maybe_set_tf32()
    device = device or best_device()
    print(f"Training on device: {device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # resume
    start_epoch = 0
    global_step = 0
    ckpt_path = latest_checkpoint(ckpt_dir)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])  # type: ignore[index]
        optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[index]
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}, step {global_step}")

    # scheduler (linear warmup to 1.0, then flat)
    total_steps = max(steps_per_epoch * num_epochs, 1)
    effective_warmup = max(min(warmup_steps, total_steps), 0)

    def _lr_lambda(step: int) -> float:
        if effective_warmup <= 0:
            return 1.0
        return min((step + 1) / float(effective_warmup), 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_lr_lambda, last_epoch=global_step - 1
    )

    model.train()
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        step_in_epoch = 0
        for input_ids, labels, attention_mask in stream_batches(
            tokenizer, batch_size, max_length, split="train", pad_token_id=pad_id
        ):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(input_ids, attention_mask=attention_mask, targets=labels)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            step_in_epoch += 1
            if step_in_epoch % 100 == 0:
                print(f"step {global_step} loss {loss.item():.4f}")
            if step_in_epoch >= steps_per_epoch:
                break

        if (epoch + 1) % save_every == 0:
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                ep_path, model, optimizer, epoch + 1, global_step, config, tokenizer.name_or_path
            )
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                tokenizer.name_or_path,
            )
            print(f"Saved checkpoint: {ep_path}")

    # save final
    weights_path = save_final_model(final_dir, model, config, tokenizer.name_or_path)
    print("Training complete. Saved to:", weights_path)
    return weights_path

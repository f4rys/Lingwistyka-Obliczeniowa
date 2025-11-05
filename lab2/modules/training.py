from __future__ import annotations

import os
import json
import math
import random
from typing import Any, Dict, Iterator, Optional, Tuple, cast

import torch
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset

from .device import best_device, maybe_set_tf32
from .dataset import make_lm_collate
from .eval import evaluate_token_nll, word_and_char_perplexity, iter_hf_texts
from .shift_labels import shift_labels


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

    collate = make_lm_collate(tokenizer, max_length=max_length)
    stream = load_dataset("mikex86/stackoverflow-posts", split=split, streaming=True)
    texts: list[str] = []

    for ex in stream:
        ex = cast(Dict[str, Any], ex)
        text = ex.get(question_field) or ""

        if text:
            texts.append(text)
            if len(texts) >= batch_size:
                input_ids, attention_mask = collate(texts)
                labels = shift_labels(input_ids, pad_id=pad_token_id or 0)

                yield input_ids, labels, attention_mask
                texts = []
    if texts:
        input_ids, attention_mask = collate(texts)
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
    grad_accum_steps: int = 2,
    use_amp: bool = True,
    seed: Optional[int] = 42,
    log_every: int = 100,
    eval_every: int = 0,
    eval_max_examples: int = 2000,
    device: Optional[torch.device] = None,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Seeding for reproducibility across tokenizers
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    maybe_set_tf32()
    device = device or best_device()
    print(f"Training on device: {device}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0

    start_epoch = 0
    global_step = 0  # micro-steps (batches seen)
    opt_steps_done = 0  # optimizer steps (after accumulation)
    ckpt_path = latest_checkpoint(ckpt_dir)

    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])  # type: ignore[index]
        optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[index]
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}, step {global_step}")
    
    grad_accum_steps = max(int(grad_accum_steps), 1)
    # Compute schedule in terms of optimizer steps
    total_opt_steps = max(math.ceil(steps_per_epoch / grad_accum_steps) * num_epochs, 1)
    effective_warmup = max(min(math.ceil(warmup_steps / grad_accum_steps), total_opt_steps), 0)

    def _lr_lambda(step: int) -> float:
        if effective_warmup <= 0:
            return 1.0
        return min((step + 1) / float(effective_warmup), 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_lr_lambda, last_epoch=opt_steps_done - 1
    )

    # AMP setup (CUDA only by default for stability)
    use_cuda_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    model.train()
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        step_in_epoch = 0  # micro-steps
        opt_step_in_epoch = 0
        running_loss = 0.0
        running_count = 0

        for input_ids, labels, attention_mask in stream_batches(
            tokenizer, batch_size, max_length, split="train", pad_token_id=pad_id
        ):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            with torch.cuda.amp.autocast(enabled=use_cuda_amp):
                logits, loss = model(input_ids, attention_mask=attention_mask, targets=labels)
                assert loss is not None
                loss_to_backprop = loss / grad_accum_steps

            if use_cuda_amp:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            step_in_epoch += 1
            global_step += 1
            running_loss += float(loss.item())
            running_count += 1

            should_step = (step_in_epoch % grad_accum_steps == 0)
            if should_step:
                if grad_clip is not None and grad_clip > 0:
                    if use_cuda_amp:
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                if use_cuda_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_steps_done += 1
                opt_step_in_epoch += 1

            if log_every and (step_in_epoch % log_every == 0):
                avg_loss = running_loss / max(running_count, 1)
                print(f"micro_step {global_step} avg_loss {avg_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e}")
                running_loss = 0.0
                running_count = 0

            if step_in_epoch >= steps_per_epoch:
                # zero-out any leftover gradients from partial accumulation to avoid leak
                optimizer.zero_grad(set_to_none=True)
                break

        if (epoch + 1) % save_every == 0:
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                ep_path, model, optimizer, epoch + 1, global_step, config, getattr(tokenizer, "name_or_path", str(tokenizer))
            )
            save_checkpoint(
                os.path.join(ckpt_dir, "last.pt"),
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                getattr(tokenizer, "name_or_path", str(tokenizer)),
            )
            print(f"Saved checkpoint: {ep_path}")

        # Optional evaluation (token -> word/char perplexity)
        if eval_every and ((epoch + 1) % eval_every == 0):
            try:
                model.eval()
                eval_texts = iter_hf_texts(split="test", field="Body")  # streaming iterator
                avg_nll, tok_count = evaluate_token_nll(
                    model, tokenizer, texts=eval_texts, max_examples=eval_max_examples, batch_size=batch_size, max_length=max_length
                )
                w_ppl, c_ppl = word_and_char_perplexity(avg_nll, tokenizer, texts=iter_hf_texts(split="test", field="Body"), max_examples=eval_max_examples)
                print(f"Validation (approx): token_NLL {avg_nll:.4f} | word_PPL {w_ppl:.2f} | char_PPL {c_ppl:.2f} (tokens {tok_count})")
            except Exception as e:
                print(f"Validation failed: {e}")
            finally:
                model.train()

    weights_path = save_final_model(final_dir, model, config, getattr(tokenizer, "name_or_path", str(tokenizer)))
    print("Training complete. Saved to:", weights_path)
    return weights_path


__all__ = [
    "latest_checkpoint",
    "stream_batches",
    "save_checkpoint",
    "save_final_model",
    "train_streamed_lm",
]

from __future__ import annotations

import math
from typing import Iterable, Iterator, Optional, Tuple

import torch
from datasets import load_dataset

from data.dataset import make_lm_collate
from .training import shift_labels


def iter_hf_texts(split: str = "test", field: str = "Body") -> Iterator[str]:
    """Iterate texts from the StackOverflow HF dataset (streaming)."""
    ds = load_dataset("mikex86/stackoverflow-posts", split=split, streaming=True)
    for ex in ds:
        text = ex.get(field) or ""
        if text:
            yield text


def iter_file_lines(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: Iterable[str],
    *,
    max_examples: Optional[int] = 5000,
    batch_size: int = 16,
    max_length: int = 256,
) -> Tuple[float, float, int]:
    """Return (perplexity, avg_loss, token_count)."""
    device = next(model.parameters()).device
    collate = make_lm_collate(tokenizer, max_length=max_length)

    def batcher():
        buf = []
        count = 0
        for t in texts:
            buf.append(t)
            count += 1
            if max_examples is not None and count >= max_examples:
                break
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    ignore_index = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    for batch_texts in batcher():
        enc_input_ids, _, attention_mask = collate(batch_texts)
        labels = shift_labels(enc_input_ids, pad_id=tokenizer.pad_token_id or 0)

        input_ids = enc_input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        logits, _ = model(input_ids, attention_mask=attention_mask, targets=None)
        vocab = logits.size(-1)
        loss_sum = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), labels.view(-1), ignore_index=ignore_index, reduction="sum"
        )
        n_tokens = int((labels.view(-1) != ignore_index).sum().item())
        total_loss_sum += float(loss_sum.item())
        total_tokens += n_tokens

    avg_loss = total_loss_sum / max(total_tokens, 1)
    ppl = float(math.exp(avg_loss))
    return ppl, avg_loss, total_tokens


def evaluate_on_hf_or_file(
    model: torch.nn.Module,
    tokenizer,
    *,
    hf_split: str = "test",
    fallback_path: str = "test.txt",
    max_examples: Optional[int] = 5000,
    batch_size: int = 16,
    max_length: int = 256,
) -> Tuple[float, float, int, str]:
    """Try HF split first; if unavailable, use fallback file. Returns (ppl, loss, tokens, source)."""
    # Try HF split
    try:
        _ = load_dataset("mikex86/stackoverflow-posts", split=hf_split, streaming=True)
        source = f"huggingface:{hf_split}"
        texts = iter_hf_texts(split=hf_split)
    except Exception:
        source = f"file:{fallback_path}"
        texts = iter_file_lines(fallback_path)

    model.eval()
    return (*evaluate_perplexity(model, tokenizer, texts, max_examples=max_examples, batch_size=batch_size, max_length=max_length), source)  # type: ignore[misc]


@torch.no_grad()
def evaluate_on_stackoverflow_test(
    model: torch.nn.Module,
    tokenizer,
    *,
    max_examples: Optional[int] = 5000,
    batch_size: int = 16,
    max_length: int = 256,
) -> Tuple[float, float, int, str]:
    """Evaluate perplexity on the StackOverflow test split (streamed)."""
    texts = iter_hf_texts(split="test")
    ppl, loss, tokens = evaluate_perplexity(
        model, tokenizer, texts, max_examples=max_examples, batch_size=batch_size, max_length=max_length
    )
    return ppl, loss, tokens, "huggingface:test"

from __future__ import annotations

import re
import math
import os
from typing import Any, Iterable, Iterator, Optional, Tuple, cast

import torch
from datasets import load_dataset

from .dataset import make_lm_collate
from .shift_labels import shift_labels


def iter_hf_texts(split: str = "test", field: str = "Body") -> Iterator[str]:
    if split == "test":
        test_file = os.path.join(os.path.dirname(__file__), '..', 'test.txt')
        yield from iter_file_lines(test_file)
    else:
        ds = load_dataset("mikex86/stackoverflow-posts", split=split, streaming=True)
        for ex in ds:
            ex = cast(dict[str, Any], ex)
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
def evaluate_token_nll(
    model: torch.nn.Module,
    tokenizer,
    texts: Iterable[str],
    *,
    max_examples: Optional[int] = 5000,
    batch_size: int = 16,
    max_length: int = 256,
) -> Tuple[float, int]:
    """Return (avg_token_nll, token_count) using the model's tokenizer space."""
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
    ignore_index = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else -100

    for batch_texts in batcher():
        enc_input_ids, attention_mask = collate(batch_texts)
        labels = shift_labels(enc_input_ids, pad_id=getattr(tokenizer, "pad_token_id", 0) or 0)

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

    avg_nll = total_loss_sum / max(total_tokens, 1)
    return avg_nll, total_tokens


def word_and_char_perplexity(
    avg_token_nll: float,
    tokenizer,
    texts: Iterable[str],
    *,
    max_examples: Optional[int] = 5000,
) -> Tuple[float, float]:
    """Approximate word- and char-level perplexity from token-level NLL.

    We estimate average tokens/word and tokens/char on the provided texts using the tokenizer
    then scale the token NLL accordingly: NLL(word) = NLL(token) * tokens/word.
    PPL(x) = exp(NLL(x)).
    """
    def take_n(it, n):
        count = 0
        for x in it:
            yield x
            count += 1
            if n is not None and count >= n:
                break

    texts_list = list(take_n(texts, max_examples))
    if not texts_list:
        return float("nan"), float("nan")

    total_words = 0
    total_chars = 0
    total_tokens = 0

    word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    for t in texts_list:
        total_chars += len(t)
        words = word_re.findall(t)
        total_words += max(len(words), 1)
        enc = tokenizer(t, return_tensors="pt")
        total_tokens += int(enc["attention_mask"].sum().item())

    tok_per_word = total_tokens / max(total_words, 1)
    tok_per_char = total_tokens / max(total_chars, 1)

    word_ppl = math.exp(avg_token_nll * tok_per_word)
    char_ppl = math.exp(avg_token_nll * tok_per_char)
    return float(word_ppl), float(char_ppl)


def whitespace_oov_stats(vocab: set[str], texts: Iterable[str], max_examples: Optional[int] = 5000) -> Tuple[int, int, float]:
    """Return (#OOV_words, total_words, percent_OOV)."""
    def take_n(it, n):
        count = 0
        for x in it:
            yield x
            count += 1
            if n is not None and count >= n:
                break

    word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    total_words = 0
    oov_words = 0
    for t in take_n(texts, max_examples):
        words = word_re.findall(t)
        for w in words:
            if w.strip():
                total_words += 1
                if w not in vocab:
                    oov_words += 1
    pct = (oov_words / total_words) * 100.0 if total_words else 0.0
    return oov_words, total_words, pct


__all__ = [
    "iter_hf_texts",
    "iter_file_lines",
    "evaluate_token_nll",
    "word_and_char_perplexity",
    "whitespace_oov_stats",
]

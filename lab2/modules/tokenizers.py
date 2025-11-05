from __future__ import annotations

import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sentencepiece as spm
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# Pretrained (HF) tokenizer
def build_pretrained_tokenizer(model_name_or_path: str = "gpt2") -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok

# Whitespace tokenizer
_WS_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def ws_tokenize(text: str) -> List[str]:
    return _WS_RE.findall(text)


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"

    @property
    def size(self) -> int:
        return len(self.itos)


def build_ws_vocab(texts: Iterable[str], vocab_size: int, *, min_freq: int = 1) -> Vocab:
    counter: Counter[str] = Counter()

    for t in texts:
        counter.update(ws_tokenize(t))

    # Reserve specials at the start
    itos: List[str] = ["<PAD>", "<UNK>"]
    for tok, _ in counter.most_common():
        if tok in ("<PAD>", "<UNK>"):
            continue
        if counter[tok] < min_freq:
            continue
        itos.append(tok)
        if len(itos) >= vocab_size:
            break

    stoi = {s: i for i, s in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


class WhitespaceTokenizer:
    name_or_path: str

    def __init__(self, vocab: Vocab, name_or_path: str = "whitespace") -> None:
        self.vocab = vocab
        self.name_or_path = name_or_path
        self.pad_token = vocab.pad_token
        self.unk_token = vocab.unk_token
        self.pad_token_id = vocab.stoi.get(vocab.pad_token, 0)
        self.unk_token_id = vocab.stoi.get(vocab.unk_token, 1)

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in ws_tokenize(text):
            ids.append(self.vocab.stoi.get(tok, self.unk_token_id))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        # Reconstruct with space rules: no space before punctuation, space otherwise
        toks = [self.vocab.itos[i] if 0 <= i < len(self.vocab.itos) else self.unk_token for i in ids]
        out = []
        for i, t in enumerate(toks):
            if i > 0 and not re.match(r"^[,.;:!?)]$", t) and not re.match(r"^[(']$", toks[i - 1]):
                out.append(" ")
            out.append(t)
        return "".join(out)

    def tokens(self, text: str) -> List[str]:
        return [self.vocab.itos[i] if 0 <= i < len(self.vocab.itos) else self.unk_token for i in self.encode(text)]

    def __call__(
        self,
        texts: Sequence[str] | str,
        return_tensors: Optional[str] = None,
        padding: bool | str = True,
        truncation: bool | str = True,
        max_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        ids_list = [self.encode(t)[: max_length if truncation else None] for t in texts]
        max_len = max(len(x) for x in ids_list) if ids_list else 0

        if padding:
            max_len = max(max_len, max_length) if isinstance(padding, str) and padding == "max_length" else max_len

        batch = torch.full((len(ids_list), max_len), fill_value=self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(ids_list), max_len), dtype=torch.long)

        for i, ids in enumerate(ids_list):
            L = len(ids)
            batch[i, :L] = torch.tensor(ids, dtype=torch.long)
            attn[i, :L] = 1

        return {"input_ids": batch, "attention_mask": attn}


# SentencePiece tokenizer
class SentencePieceTokenizer:
    def __init__(self, model_file: str, name_or_path: Optional[str] = None):
        self.sp = spm.SentencePieceProcessor()  # type: ignore
        self.sp.load(model_file)  # type: ignore
        self.name_or_path = name_or_path or os.path.basename(model_file)
        # Define specials for alignment with other tokenizers
        self.pad_token_id = 0 if self.sp.pad_id() < 0 else self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()  # type: ignore

    def encode(self, text: str) -> List[int]:
        return list(self.sp.encode(text, out_type=int))  # type: ignore

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode(ids)  # type: ignore

    def tokens(self, text: str) -> List[str]:
        return list(self.sp.encode(text, out_type=str))  # type: ignore

    def __call__(
        self,
        texts: Sequence[str] | str,
        return_tensors: Optional[str] = None,
        padding: bool | str = True,
        truncation: bool | str = True,
        max_length: int = 256,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        ids_list = [self.encode(t)[: max_length if truncation else None] for t in texts]
        max_len = max(len(x) for x in ids_list) if ids_list else 0

        if padding:
            max_len = max(max_len, max_length) if isinstance(padding, str) and padding == "max_length" else max_len

        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        batch = torch.full((len(ids_list), max_len), fill_value=pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids_list), max_len), dtype=torch.long)

        for i, ids in enumerate(ids_list):
            L = len(ids)
            batch[i, :L] = torch.tensor(ids, dtype=torch.long)
            attn[i, :L] = 1

        return {"input_ids": batch, "attention_mask": attn}


def train_sentencepiece(
    input_path: str,
    model_prefix: str,
    vocab_size: int,
    *,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    user_defined_symbols: Optional[List[str]] = None,
) -> str:
    """Train SentencePiece model and return model file path."""
    args = {
        "input": input_path,
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        # Set explicit ids so we have a real PAD distinct from UNK
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": -1,
        "eos_id": -1,
    }
    if user_defined_symbols:
        args["user_defined_symbols"] = ",".join(user_defined_symbols)
    spm.SentencePieceTrainer.train(**args)  # type: ignore[arg-type]
    model_file = model_prefix + ".model"
    return model_file


# Tokenizer metrics
def tokenizer_throughput(tokenizer, texts: Sequence[str], iters: int = 1) -> Tuple[float, int]:
    """Return (tokens_per_second, total_tokens)."""
    # warmup
    _ = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    t0 = time.time()
    total_tokens = 0
    for _ in range(iters):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        total_tokens += int(enc["attention_mask"].sum().item())
    t1 = time.time()
    dt = max(t1 - t0, 1e-9)
    return total_tokens / dt, total_tokens


def avg_tokens_per_word(tokenizer, texts: Iterable[str], max_examples: Optional[int] = 5000) -> float:
    def take_n(it, n):
        count = 0
        for x in it:
            yield x
            count += 1
            if n is not None and count >= n:
                break

    texts_list = list(take_n(texts, max_examples))

    if not texts_list:
        return float("nan")

    word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    total_words = 0
    total_tokens = 0

    for t in texts_list:
        total_words += max(len(word_re.findall(t)), 1)
        enc = tokenizer(t, return_tensors="pt")
        total_tokens += int(enc["attention_mask"].sum().item())

    return total_tokens / max(total_words, 1)


def percent_words_encoded_directly(tokenizer, texts: Iterable[str], max_examples: Optional[int] = 100) -> float:
    """Percent of words that are encoded as a single token without UNK.
    For subword tokenizers, we check each word in isolation and see 
    if its tokenization length is 1.
    """
    word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    total = 0
    single = 0
    count = 0

    for t in texts:
        for w in word_re.findall(t):
            if not w.strip():
                continue

            enc = tokenizer(w, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()  # type: ignore[index]
            length = int((enc["attention_mask"][0]).sum().item())  # type: ignore[index]

            if length == 1:
                # consider it directly encoded if length==1 and not unk
                if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
                    if ids[0] != tokenizer.unk_token_id:
                        single += 1
                else:
                    single += 1
            total += 1
        count += 1

        if max_examples is not None and count >= max_examples:
            break

    return (single / total) * 100.0 if total else 0.0


def avg_tokens_per_word_min_bytes(
    tokenizer,
    texts: Iterable[str],
    *,
    min_bytes: int = 1_000_000,
) -> float:
    """Average tokens per word, reading texts until at least `min_bytes` 
    of input is processed. This helps standardize measurements across tokenizers.
    """
    word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    total_words = 0
    total_tokens = 0
    seen_bytes = 0

    for t in texts:
        if not t:
            continue

        seen_bytes += len(t.encode("utf-8", errors="ignore"))
        total_words += max(len(word_re.findall(t)), 1)
        enc = tokenizer(t, return_tensors="pt")
        total_tokens += int(enc["attention_mask"].sum().item())

        if seen_bytes >= min_bytes:
            break

    if total_words == 0:
        return float("nan")

    return total_tokens / total_words


__all__ = [
    "build_pretrained_tokenizer",
    "WhitespaceTokenizer",
    "build_ws_vocab",
    "SentencePieceTokenizer",
    "train_sentencepiece",
    "tokenizer_throughput",
    "avg_tokens_per_word",
    "avg_tokens_per_word_min_bytes",
    "percent_words_encoded_directly",
]

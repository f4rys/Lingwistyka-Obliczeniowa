from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    emb_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    pad_token_id: Optional[int] = 0
    tie_embeddings: bool = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :] # type: ignore


def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
    return mask


class TransformerLanguageModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_token_id)
        self.pos = PositionalEncoding(config.emb_dim, max_len=config.max_seq_len)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=config.emb_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(encoder_layer, num_layers=config.n_layers)
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.proj = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.proj.weight = self.embed.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        if self.proj.weight is not self.embed.weight:
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert input_ids.dtype == torch.long
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids)
        x = self.pos(x)
        memory = torch.zeros(B, 1, self.config.emb_dim, device=device)
        tgt_mask = _causal_mask(T, device)
        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = attention_mask == 0

        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        y = self.ln_f(y)
        logits = self.proj(y)

        loss = None
        if targets is not None:
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id if self.config.pad_token_id is not None else -100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        device = input_ids.device

        for _ in range(max_new_tokens):
            B, T = generated.shape
            x = self.embed(generated)
            x = self.pos(x)

            memory = torch.zeros(B, 1, self.config.emb_dim, device=device)
            tgt_mask = _causal_mask(T, device)

            y = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
            y = self.ln_f(y)
            logits = self.proj(y)[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

            if eos_token_id is not None and torch.all(next_id.squeeze(-1) == eos_token_id):
                break

        return generated


__all__ = ["TransformerConfig", "TransformerLanguageModel"]

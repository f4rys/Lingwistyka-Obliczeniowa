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
    num_classes: int = 2


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


class TransformerClassifier(nn.Module):
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
        self.classifier = nn.Linear(config.emb_dim, config.num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.classifier:
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
            # TransformerDecoder expects True for padded positions
            tgt_key_padding_mask = attention_mask == 0

        y = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        y = self.ln_f(y)
        
        # Mean Pooling
        if attention_mask is None:
            mask = torch.ones((B, T), device=device)
        else:
            mask = attention_mask.float()
            
        mask_expanded = mask.unsqueeze(-1).expand(y.size())
        sum_embeddings = torch.sum(y * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        logits = self.classifier(pooled)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

__all__ = ["TransformerConfig", "TransformerClassifier"]

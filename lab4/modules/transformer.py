from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if FlashAttention is available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


@dataclass
class TransformerConfig:
    """Configuration for the Transformer Language Model."""
    vocab_size: int
    emb_dim: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    pad_token_id: Optional[int] = 0
    tie_embeddings: bool = True
    
    # Memory optimization flags
    use_flash_attention: bool = False
    use_windowed_attention: bool = False
    window_size: int = 128
    gradient_checkpointing: bool = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
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
        return x + self.pe[:, :T, :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for:
    - Standard attention
    - FlashAttention 2
    - Windowed (sliding window) attention
    """
    def __init__(
        self, 
        emb_dim: int, 
        n_heads: int, 
        dropout: float = 0.0,
        use_flash_attention: bool = False,
        use_windowed_attention: bool = False,
        window_size: int = 128,
    ):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.use_windowed_attention = use_windowed_attention
        self.window_size = window_size
        
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, n_heads, T, head_dim)
        
        if self.use_flash_attention and FLASH_ATTN_AVAILABLE:
            # FlashAttention expects (B, T, n_heads, head_dim)
            q = q.transpose(1, 2).contiguous()  # (B, T, n_heads, head_dim)
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            if self.use_windowed_attention:
                # Windowed attention with FlashAttention
                attn_out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=is_causal,
                    window_size=(self.window_size, 0),  # Left window only for causal
                )
            else:
                # Full FlashAttention
                attn_out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=is_causal,
                )
            attn_out = attn_out.reshape(B, T, C)
        else:
            # Standard attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool), 
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            
            # Apply windowed attention mask if enabled (without FlashAttention)
            if self.use_windowed_attention and not (self.use_flash_attention and FLASH_ATTN_AVAILABLE):
                window_mask = torch.ones(T, T, device=x.device, dtype=torch.bool)
                for i in range(T):
                    start = max(0, i - self.window_size)
                    window_mask[i, start:i+1] = False
                attn_weights = attn_weights.masked_fill(window_mask, float('-inf'))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(pad_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm architecture."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.attn = MultiHeadAttention(
            config.emb_dim,
            config.n_heads,
            config.dropout,
            use_flash_attention=config.use_flash_attention,
            use_windowed_attention=config.use_windowed_attention,
            window_size=config.window_size,
        )
        self.ln2 = nn.LayerNorm(config.emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.emb_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.emb_dim),
            nn.Dropout(config.dropout),
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    """Decoder-only Transformer for language modeling."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_token_id)
        self.pos = PositionalEncoding(config.emb_dim, max_len=config.max_seq_len)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.proj = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        
        if config.tie_embeddings:
            self.proj.weight = self.embed.weight
        
        # Gradient checkpointing flag
        self._gradient_checkpointing = config.gradient_checkpointing
        
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
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory savings."""
        self._gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.drop(x)
        
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x = torch.utils.checkpoint.checkpoint(
                    block, x, attention_mask,
                    use_reentrant=False
                )
            else:
                x = block(x, attention_mask)
        
        x = self.ln_f(x)
        logits = self.proj(x)
        
        loss = None
        if targets is not None:
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id if self.config.pad_token_id is not None else -100,
            )
        
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["TransformerConfig", "TransformerLanguageModel", "count_parameters", "FLASH_ATTN_AVAILABLE"]

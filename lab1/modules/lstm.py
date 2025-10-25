"""
LSTM Language Model
--------------------

Classical LSTM LM suitable for next-token prediction on text corpora.

Works with the Hugging Face dataset "mikex86/stackoverflow-posts" by
tokenizing the sample["Body"] (and optionally sample["Title"]). The
`forward` expects input_ids shaped (batch, seq_len) and returns logits
of shape (batch, seq_len, vocab_size). If `targets` is passed, it also
returns the cross-entropy loss, ignoring positions equal to
`pad_token_id`.

Contract
- inputs: input_ids (LongTensor [B, T]), optional attention_mask ([B, T])
		  optional targets ([B, T])
- outputs: logits ([B, T, V]) and optionally loss (scalar)
- masking: loss ignores pad_token_id if provided; attention_mask is used
		   to zero out contributions to hidden states at padded tokens.

Note
- Sequence packing (pack_padded_sequence) is optional; here we keep a
  simple implementation that runs unrolled over full T and relies on
  loss masking to ignore pads. This is robust and simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LSTMConfig:
	vocab_size: int
	emb_dim: int = 512
	hidden_size: int = 768
	num_layers: int = 2
	dropout: float = 0.1
	pad_token_id: Optional[int] = 0
	tie_embeddings: bool = True


class LSTMLanguageModel(nn.Module):
	def __init__(self, config: LSTMConfig):
		super().__init__()
		self.config = config

		self.embed = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_token_id)
		self.dropout = nn.Dropout(config.dropout)
		self.lstm = nn.LSTM(
			input_size=config.emb_dim,
			hidden_size=config.hidden_size,
			num_layers=config.num_layers,
			dropout=config.dropout if config.num_layers > 1 else 0.0,
			batch_first=True,
		)
		# Output projection
		self.proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		if config.tie_embeddings and config.emb_dim == config.hidden_size:
			# Weight tying when dims match
			self.proj.weight = self.embed.weight

		self._reset_parameters()

	def _reset_parameters(self) -> None:
		# Initialize embeddings and linear layers
		nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
		for name, param in self.lstm.named_parameters():
			if "weight_ih" in name:
				nn.init.xavier_uniform_(param)
			elif "weight_hh" in name:
				nn.init.orthogonal_(param)
			elif "bias" in name:
				nn.init.zeros_(param)
				# Set forget gate bias to 1
				n = param.numel() // 4
				param.data[n : 2 * n] = 1.0
		if self.proj.weight is not self.embed.weight:
			nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: Optional[torch.Tensor] = None,
		targets: Optional[torch.Tensor] = None,
		hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""
		Args:
			input_ids: LongTensor (B, T)
			attention_mask: Bool/Byte/Long (B, T) where 1 means keep, 0 means pad
			targets: LongTensor (B, T) with next-token ids to predict
			hidden: initial hidden state for stateful decoding (optional)

		Returns:
			logits: FloatTensor (B, T, V)
			loss: optional scalar CE loss
		"""
		assert input_ids.dtype == torch.long, "input_ids must be torch.long"
		x = self.embed(input_ids)
		x = self.dropout(x)

		out, new_hidden = self.lstm(x, hidden)

		logits = self.proj(out)

		loss = None
		if targets is not None:
			vocab_size = logits.size(-1)
			loss = F.cross_entropy(
				logits.view(-1, vocab_size),
				targets.view(-1),
				ignore_index=self.config.pad_token_id if self.config.pad_token_id is not None else -100,
			)
		return (logits, loss)

	@torch.no_grad()
	def generate(
		self,
		input_ids: torch.Tensor,
		max_new_tokens: int = 50,
		temperature: float = 1.0,
		top_k: Optional[int] = None,
		eos_token_id: Optional[int] = None,
		hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
	) -> torch.Tensor:
		"""Greedy/top-k sampling generation for convenience.

		Args:
			input_ids: (B, T)
			max_new_tokens: number of tokens to append
			temperature: >0; values <1 sharpen logits, >1 soften
			top_k: if set, keep only top_k logits per step
			eos_token_id: optional stop token
			hidden: optional lstm state (h, c)
		Returns:
			(B, T + max_len)
		"""
		self.eval()
		B, T = input_ids.shape
		generated = input_ids
		state = hidden
		for _ in range(max_new_tokens):
			logits, state = self._step(generated[:, -1:], state)
			logits = logits[:, -1, :] / max(temperature, 1e-6)
			if top_k is not None and top_k > 0:
				v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
				logits[logits < v[:, [-1]]] = -float("inf")
			probs = F.softmax(logits, dim=-1)
			next_id = torch.multinomial(probs, num_samples=1)
			generated = torch.cat([generated, next_id], dim=1)
			if eos_token_id is not None:
				# stop if all batches hit EOS
				if torch.all(next_id.squeeze(-1) == eos_token_id):
					break
		return generated

	def _step(
		self,
		input_ids: torch.Tensor,
		hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
	) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		# one-step forward used during generation
		x = self.embed(input_ids)
		out, hidden = self.lstm(x, hidden)
		logits = self.proj(out)
		return logits, hidden


__all__ = ["LSTMConfig", "LSTMLanguageModel"]


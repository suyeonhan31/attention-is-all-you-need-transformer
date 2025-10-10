from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    # learned embedding, scaled by sqrt(D); init N(0, D^-0.5) so post-scale
    # variance matches the positional encoding it's added to (not Xavier)
    def __init__(self, vocabulary_size: int, d_model: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, d_model)  # (vocab, D)
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.d_model**-0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens:(B, L)-> (B, L, D)
        return self.token_embedding(tokens) * self.scale

class PositionalEncoding(nn.Module):
    # fixed sinusoidal position signal (Sec 3.5):
    # PE(pos, 2i) = sin(pos / 10000^(2i/D)), PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model) # (max_len, D)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1) # (max_len, 1)
        two_i = torch.arange(0, d_model, 2, dtype=torch.float32) # (D/2,)
        inv_freq = torch.exp(-two_i * (math.log(10000.0) / d_model)) # (D/2,), 10000^(-2i/D) in log space

        pos_encoding[:, 0::2] = torch.sin(pos * inv_freq) # (max_len, D/2) -> even columns
        # cosine half is one column short when D is odd
        pos_encoding[:, 1::2] = torch.cos(pos * inv_freq)[:, : pos_encoding[:, 1::2].size(1)]

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0), persistent=False)  # (1, max_len, D)

    # x: (B, L, D); offset = tokens already decoded (0 for a full forward pass)
    # pos_encoding slice: (1, L, D), broadcasts over batch
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.dropout(x + self.pos_encoding[:, offset : offset + x.size(1)])

    def encoding_at(self, pos: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding[0, pos]


# trained position table, alternative to the fixed sinusoids
#cannot extrapolate past max_len the way PositionalEncoding can
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model) #(max_len, D)
        self.max_len = max_len
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pe.weight, mean=0.0, std=1.0)  # unit variance, matching token embedding scale

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (B, L, D) -> (B, L, D)
        positions = torch.arange(offset, offset + x.size(1), device=x.device)  # (L,)
        return self.dropout(x + self.pe(positions).unsqueeze(0))  # pe(positions): (L, D) -> (1, L, D)


class OutputProjection(nn.Module):
    def __init__(
        self, d_model: int, vocab_size: int, tied_embedding= None
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size, bias=False)  # weight: (vocab, D)
        if tied_embedding is not None:
            if tied_embedding.weight.shape != self.projection.weight.shape:
                raise ValueError(
                    f"cannot tie embedding of shape {tuple(tied_embedding.weight.shape)} "
                    f"to a projection of shape {tuple(self.projection.weight.shape)}"
                )
            self.projection.weight = tied_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, L, vocab_size), raw logits
        return self.projection(x)

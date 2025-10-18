import torch
from torch import nn

from .attention import MultiHeadAttention
from .cache import DecoderCache
from .config import ModelConfig
from .decoder import Decoder, DecoderLayer
from .embedding import (LearnedPositionalEmbedding,OutputProjection,PositionalEncoding,TokenEmbedding,)
from .encoder import Encoder, EncoderLayer
from .layers import PositionwiseFeedForward

"""
Data flow:
Source
(B, Ls) -> scaled embedding -> + pos encoding -> encoder -> memory -> (B, Ls, D)

Target 
(B, Lt) -> scaled embedding -> + pos encoding -> decoder

Self attention causal over target, cross attention over memory -> (B, Lt, D)
"""

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        c = config

        def make_attention() -> MultiHeadAttention:
            return MultiHeadAttention(
                d_model= c.d_model,
                num_heads= c.num_heads,
                dropout= c.attention_dropout,
                d_k= c.d_k,
                d_v= c.d_v,
                bias= c.attention_bias,
            )

        def make_ffn() -> PositionwiseFeedForward:
            return PositionwiseFeedForward(c.d_model, c.d_ff, c.ffn_dropout)

        self.encoder = Encoder(
            EncoderLayer(c.d_model, make_attention(), make_ffn(), c.dropout)
            for _ in range(c.num_layers)
        )
        self.decoder = Decoder(
            DecoderLayer(c.d_model, make_attention(), make_attention(), make_ffn(), c.dropout)
            for _ in range(c.num_layers)
        )

        self.src_embed = TokenEmbedding(c.src_vocab_size, c.d_model)  # (src_vocab, D)
        if c.share_embeddings:
            self.tgt_embed = self.src_embed  #Sec 3.4: one shared weight matrix
        else:
            self.tgt_embed = TokenEmbedding(c.tgt_vocab_size, c.d_model)  # (tgt_vocab, D)

        pos_cls = PositionalEncoding if c.positional_encoding == "sinusoidal" else LearnedPositionalEmbedding
        self.src_pos = pos_cls(c.d_model, max_len=c.max_len, dropout=c.dropout)

        self.tgt_pos = self.src_pos if c.positional_encoding == "sinusoidal" else pos_cls(
            c.d_model, max_len=c.max_len, dropout=c.dropout
        )

        self.output_projection = OutputProjection(
            c.d_model,
            c.tgt_vocab_size,
            tied_embedding=self.tgt_embed.token_embedding if c.share_embeddings else None,
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        tables = {id(m.weight) for m in self.modules() if isinstance(m, nn.Embedding)}
        for p in self.parameters():
            if p.dim() < 2 or id(p) in tables:
                continue
            nn.init.xavier_uniform_(p)

        for module in {self.src_embed, self.tgt_embed, self.src_pos, self.tgt_pos}:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def encode(
        self,
        src: torch.Tensor,#(B, Ls)
        src_mask = None,     
        store_attention: bool = False,
    ) -> torch.Tensor:
        x = self.src_pos(self.src_embed(src))  # (B, Ls) -> (B, Ls, D) -> (B, Ls, D)
        return self.encoder(x, src_mask, store_attention=store_attention)  # -> (B, Ls, D)

    def decode(
        self,
        memory: torch.Tensor,#(B, Ls, D)
        tgt: torch.Tensor, #(B, Lt) full sequence, or newest tokens only if cache is set
        src_mask = None, #broadcastable to (B, H, Lt, Ls)
        tgt_mask = None, #broadcastable to (B, H, Lt, Lt); optional with cache
        cache = None,
        store_attention: bool = False,
    ) -> torch.Tensor:

        offset = cache.position if cache is not None else 0
        x = self.tgt_pos(self.tgt_embed(tgt), offset=offset)  # (B, Lt) -> (B, Lt, D) -> (B, Lt, D)
        return self.decoder(
            x, memory, src_mask, tgt_mask, cache=cache, store_attention=store_attention
        )  # -> (B, Lt, D)

    def forward(
        self,
        src: torch.Tensor, #(B, Ls)
        tgt: torch.Tensor, #(B, Lt)
        src_mask = None, #broadcastable to (B, H, *, Ls)
        tgt_mask= None, #broadcastable to (B, H, Lt, Lt)
        store_attention: bool = False,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask, store_attention=store_attention)  # (B, Ls, D)
        return self.decode(memory, tgt, src_mask, tgt_mask, store_attention=store_attention)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )
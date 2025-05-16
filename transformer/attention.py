import math
import torch
from torch import nn
from torch.nn import functional as F

#Scaled Dot-Product Attention
# Attention(Q, K, V) 

def scaled_dot_product(
    query: torch.Tensor,#(Batch, Heads, Lq (tokens), dk)
    key: torch.Tensor, #(B, H, Lk, dk)
    value: torch.Tensor, #(B, H, Lk, dv)
    mask:torch.Tensor = None, #broadcastable to (B, H, Lq, Lk) True = keep False = mask out
    dropout: nn.Module = None,) -> tuple[torch.Tensor, torch.Tensor]:

    d_k = query.size(-1) #query-key dimension
    # (B, H, Lq, dk) @ (B, H, dk, Lk) -> (B, H, Lq, Lk)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #swap last two (B, H, Lk, dk) -> (B, H, dk, Lk)
    #row 3, column 7 tells you how well token 3's query matched token 7's key


    # 3.2.3: illegal connections are set to -inf in the input of the softmax -> contribute 0 weight
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float("-inf"))  # masked_fill(where, value)

    # weights[b, h, i, :] sums to 1 -> a probability distribution over key positions for query i
    weights = F.softmax(scores, dim=-1) #(B, H, Lq, Lk) softmax over the Lk key axis
    if dropout is not None:
        weights = dropout(weights) #regularizer that prevents overfitting ->A Simple Way to Prevent Neural Networks from Overfitting
    return torch.matmul(weights, value), weights

#Multi-Head Attention 
#MultiHead(Q, K, V)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 d_k: int | None = None, d_v: int | None = None, bias=False) -> None:
        #D 512, H 8 

        super().__init__()
        if d_k is None: #per head query 
            self.d_k = d_model // num_heads #dk, e.g. 512 // 8 = 64
        else:
            self.d_k = d_k

        if d_v is None:
            self.d_v = d_model // num_heads
        else:
            self.d_v = d_v

        self.num_heads = num_heads    
        self.d_model = d_model

        #Each linear layer projects all heads at once: D -> H * per-head-dim.
        #split into individual heads afterward with split_heads().

        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=bias) #D -> H*dk
        self.w_k = nn.Linear(d_model, num_heads * self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, num_heads * self.d_v, bias=bias) #D -> H*dv
        self.w_o = nn.Linear(num_heads * self.d_v, d_model, bias=bias) #H*dv -> D

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None # holds (B, H, Lq, Lk) if stored

    #W_k and W_v split into heads
    #Projects raw key/value inputs and splits each into per head shape.
    def project_kv(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.split_heads(self.w_k(key), self.d_k), #(B, Lk, D) -> (B, Lk, H*dk) -> (B, H, Lk, dk)
            self.split_heads(self.w_v(value), self.d_v),# (B, Lk, D) -> (B, Lk, H*dv) -> (B, H, Lk, dv)
        )

    #(B, L, h*depth) -> (B, h, L, head_dim)
    def split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, head_dim).transpose(1, 2)
    
    #(B, h, L, d_v) -> (B, L, h*d_v) concat
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.num_heads * self.d_v)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask= None, cache= None, store_attention: bool = False,) -> torch.Tensor:
        q = self.split_heads(self.w_q(query), self.d_k)

        if cache is None:
            k, v = self.project_kv(key, value)
        else: #key values 
            mask = cache.attention_mask(mask, q.size(2), q.device)
            k, v = cache.keys_values(lambda: self.project_kv(key, value))

        attended, weights = scaled_dot_product(q, k, v, mask, self.dropout)
        self.attention_weights = weights.detach() if store_attention else None
        return self.w_o(self.merge_heads(attended))

    

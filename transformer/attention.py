import math
import torch
from torch import nn
from torch.nn import functional as F

#Scaled Dot-Product Attention
# Attention(Q, K, V) 
#batch, heads, n (tokens), d_k
def scaled_dot_product(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    mask:torch.Tensor = None, dropout: nn.Module = None,) -> tuple[torch.Tensor, torch.Tensor]:

    d_k = query.size(-1) #query-key dimension
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #swap last two 
    #row 3, column 7 tells you how well token 3's query matched token 7's key
    if mask is not None:
        #3.2.3: illegal connections are set to -inf in the input of the softmax, so they contribute zero weight.
        scores = scores.masked_fill(~mask, float("-inf"))  # masked_fill(where, value)

    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights) #regularizer that prevents overfitting ->A Simple Way to Prevent Neural Networks from Overfitting
    return torch.matmul(weights, value), weights

#Multi-Head Attention 
#MultiHead(Q, K, V)
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 d_k: int | None = None, d_v: int | None = None, bias=False):
    
    """
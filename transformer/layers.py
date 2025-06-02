import torch
from torch import nn
from torch.nn import functional as F

def layer_stack(layers) -> nn.ModuleList:
    # -> nn.ModuleList of N distinct layer objects (not N references to one shared layer)
    stack = nn.ModuleList(layers)
    if len({id(layer) for layer in stack}) != len(stack):
        raise ValueError("stack layers must be distinct module instances, not [layer] * N")
    return stack

class LayerNorm(nn.Module):
    # normalizes over the last feature axis, gain/bias: (D,)
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features)) #(D,)
        self.bias = nn.Parameter(torch.zeros(features)) #(D,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D) -> (.., D), mean/var computed per-position over the D axis
        mean = x.mean(dim=-1, keepdim=True) # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False) #(..., 1), biased variance
        return self.gain * (x - mean) / torch.sqrt(var + self.eps) + self.bias #(.., D)

# FFN(x) = max(0, x W_1 + b_1) W_2 + b_2 applied identically at every position
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # D -> d_ff
        self.w_2 = nn.Linear(d_ff, d_model)# d_ff -> D
        self.dropout = nn.Dropout(dropout) # applied to the ReLU output, between the two linears

    # x: (B, L, D) -> (B, L, d_ff) -> (B, L, D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, features: int, dropout: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = LayerNorm(features, eps=eps)
        self.dropout = nn.Dropout(dropout)

    #residual connection + post-norm: x, sublayer(x): (B, L, D) -> (B, L, D)
    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer(x)))
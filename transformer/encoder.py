import torch
from torch import nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection, layer_stack


class EncoderLayer(nn.Module):
    # two sub-layers per layer: self-attention, then the position-wise FFN,
    # each wrapped in a residual connection + LayerNorm (SublayerConnection)
    def __init__(
        self, d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward, dropout: float,) -> None:

        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(
        self,
        x: torch.Tensor,#(B, L, D)
        src_mask = None, #broadcastable to (B, H, L, L)
        store_attention: bool = False,) -> torch.Tensor:
        # self-attention: y stands in for Q, K, and V alike -> (B, L, D)

        x = self.sublayers[0](
            x,lambda y: self.self_attn(y, y, y, src_mask, store_attention=store_attention),
        )
        return self.sublayers[1](x, self.feed_forward)  #(B, L, D) -> (B, L, D)

#stack of N EncoderLayers, each with its own parameters (see layer_stack)
#layers arrive pre-built -> this class only holds and runs them in sequence
class Encoder(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = layer_stack(layers)

    def forward(
        self,
        x: torch.Tensor, # (B, L, D)
        src_mask= None, # broadcastable to (B, H, L, L)
        store_attention: bool = False)-> torch.Tensor:

        for layer in self.layers:
            x = layer(x, src_mask, store_attention=store_attention)  #(B, L, D) -> (B, L, D)
        return x

    def attention_weights(self) -> list:
        # one entry per layer each -> None or (B, H, L, L) from the last forward pass
        return [layer.self_attn.attention_weights for layer in self.layers]
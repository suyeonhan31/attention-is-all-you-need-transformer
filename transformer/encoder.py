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
        self, d_model: int, vocab_size: int, tied_embedding: Optional[nn.Embedding] = None
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

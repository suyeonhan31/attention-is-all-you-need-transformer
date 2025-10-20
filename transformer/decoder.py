import torch
from torch import nn

from .attention import MultiHeadAttention
from .cache import DecoderCache, LayerCache
from .layers import PositionwiseFeedForward, SublayerConnection, layer_stack

class DecoderLayer(nn.Module):
    # three sub-layers: masked self-attention, cross-attention over the encoder
    # output, then the position-wise FFN, each wrapped in a SublayerConnection
    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,) -> None:

        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(
        self,
        x: torch.Tensor, # (B, Lt, D), decoder input
        memory: torch.Tensor, # (B, Ls, D), encoder output
        src_mask= None, # broadcastable to (B, H, Lt, Ls)
        tgt_mask= None, # broadcastable to (B, H, Lt, Lt)
        cache= None, # this layer's KV cache for incremental decoding
        store_attention: bool = False,
    ) -> torch.Tensor:
        self_cache = cache.self_attn if cache is not None else None
        cross_cache = cache.cross_attn if cache is not None else None

        # self-attention: Q=K=V=y (decoder attending to itself), masked by tgt_mask
        x = self.sublayers[0](
            x,
            lambda y: self.self_attn(
                y, y, y, tgt_mask, cache=self_cache, store_attention=store_attention
            ),
        )  #(B, Lt, D) -> (B, Lt, D)

        # cross-attention: Q=y (decoder), K=V=memory (encoder output), masked by src_mask
        x = self.sublayers[1](
            x,
            lambda y: self.cross_attn(
                y, memory, memory, src_mask, cache=cross_cache, store_attention=store_attention
            ),
        )  #(B, Lt, D) -> (B, Lt, D)

        return self.sublayers[2](x, self.feed_forward)  #(B, Lt, D) -> (B, Lt, D)


class Decoder(nn.Module):
    # stack of N DecoderLayers, each with its own parameters (see layer_stack)
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = layer_stack(layers)

    def forward(
        self,
        x: torch.Tensor, # (B, Lt, D)
        memory: torch.Tensor, # (B, Ls, D)
        src_mask= None,
        tgt_mask= None,
        cache= None, # one LayerCache per layer, or None for a full pass
        store_attention: bool = False,
    ) -> torch.Tensor:
        if cache is not None and len(cache.layers) != len(self.layers):
            raise ValueError(
                f"cache has {len(cache.layers)} layers but decoder has {len(self.layers)}"
            )
        for i, layer in enumerate(self.layers):
            layer_cache = cache.layers[i] if cache is not None else None
            x = layer(
                x, memory, src_mask, tgt_mask, cache=layer_cache, store_attention=store_attention
            )  # (B, Lt, D) -> (B, Lt, D)
        return x

    def attention_weights(self) -> dict[str, list]:
        # per-layer weights from the last forward pass:
        # "self" entries: None or (B, H, Lt, Lt) "cross" entries: None or (B, H, Lt, Ls)
        return {
            "self": [layer.self_attn.attention_weights for layer in self.layers],
            "cross": [layer.cross_attn.attention_weights for layer in self.layers],
        }
import torch
from .masking import subsequent_mask

class AttentionCache:
    def __init__(self, growing: bool) -> None:
        self.growing = growing
        self.k = None  #(B, H, L_cached, dk)
        self.v = None  #(B, H, L_cached, dv)

    def keys_values(self, compute) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.growing:
            # cross-attention: compute once from the encoder memory, then reuse
            if self.k is None:
                self.k, self.v = compute()
            return self.k, self.v

        # self-attention: compute kv for only the newest tokens, append to the cache
        new_k, new_v = compute() #(B, H, L_new, dk), (B, H, L_new, dv)
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2) #(B, H, L_cached + L_new, dk)
            self.v = torch.cat([self.v, new_v], dim=2)#(B, H, L_cached + L_new, dv)
        return self.k, self.v

    def attention_mask(self, mask, new_length: int, device: torch.device):
        if not self.growing:
            return mask

        past = 0 if self.k is None else self.k.size(2)
        causal = subsequent_mask(new_length, device=device, past=past)  # (1, 1, new_length, past+new_length)
        if mask is None:
            return causal

class LayerCache:
    def __init__(self) -> None:
        self.self_attn = AttentionCache(growing=True)
        self.cross_attn = AttentionCache(growing=False)

class DecoderCache:
    def __init__(self, num_layers: int) -> None:
        self.layers = [LayerCache() for _ in range(num_layers)]

    def position(self) -> int:
        k = self.layers[0].self_attn.k
        return 0 if k is None else k.size(2)
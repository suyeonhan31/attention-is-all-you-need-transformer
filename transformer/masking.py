import torch

# Masks are boolean;True = "this query may attend to this key".
# Shapes are (B, H, Lq, Lk) or anything that broadcasts to it.

PAD_ID = 0
NEG_INF = float("-inf")  #score val:maskedout positions before softmax


# tokens: (B, L) -> (B, 1, 1, L), True = real token, False = <pad>
# broadcasts over H and over every query position
def padding_mask(tokens: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    return (tokens != pad_id).unsqueeze(1).unsqueeze(2)

# lower-triangular mask
def subsequent_mask(
    size: int, device= None, past: int = 0
) -> torch.Tensor:
    # -> (1, 1, size, past + size), True = attend allowed
    #query i has absolute position (past + i) and may attend to keys j <= past + i
    if past < 0:
        raise ValueError(f"past must be non-negative {past}")
    ones = torch.ones(size, past + size, dtype=torch.bool, device=device)  #(size, past+size)
    return torch.tril(ones, diagonal=past).unsqueeze(0).unsqueeze(0)  #(1, 1, size, past+size)


def target_mask(tokens: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    # tokens:(B, L) -> (B, 1, L, L) padding mask and causal mask
    # assume right-padding: a fully-masked row (NaN after softmax)-> query has no valid key -> only happens with left-padding
    pad = padding_mask(tokens, pad_id) # (B, 1, 1, L)
    causal = subsequent_mask(tokens.size(-1), tokens.device)# (1, 1, L, L)
    return pad & causal # broadcasts -> (B, 1, L, L)

# tokens: (B, L) with format [<s>, y_1, ..., y_m, </s>]
# -> decoder_input: (B, L-1) = tokens[:, :-1] = [<s>, y_1, ..., y_m]
    #labels:(B, L-1) = tokens[:, 1:]  = [y_1, ..., y_m, </s>]
def shift_target(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.size(1) < 2:
        raise ValueError("Target sequences must contain at least a BOS and one token.")
    return tokens[:, :-1], tokens[:, 1:]
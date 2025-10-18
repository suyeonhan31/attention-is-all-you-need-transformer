from dataclasses import dataclass

class ModelConfig: #Defaults: paper's base model: N=6, d_model=512, d_ff=2048, h=8
    src_vocab_size: int
    tgt_vocab_size: int

    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    d_k= None
    d_v= None
    dropout: float = 0.1

    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    attention_bias: bool = False

    share_embeddings: bool = True
    positional_encoding: str = "sinusoidal"
    max_len: int = 5000

    def __post_init__(self) -> None:
        if self.positional_encoding not in ("sinusoidal", "learned"):
            raise ValueError(
                f"positional_encoding must be 'sinusoidal' or 'learned', "
                f"got {self.positional_encoding!r}"
            )
        if self.share_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError(
                "share_embeddings requires src_vocab_size == tgt_vocab_size "
                f"(got {self.src_vocab_size} and {self.tgt_vocab_size})"
            )

        if self.d_k is None and self.d_model % self.num_heads:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by num_heads="
                f"{self.num_heads}; set d_k and d_v explicitly to allow this"
            )
        for name in ("d_model", "num_heads", "num_layers", "d_ff", "max_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        for name in ("d_k", "d_v"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        for name in ("dropout", "attention_dropout", "ffn_dropout"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)}")

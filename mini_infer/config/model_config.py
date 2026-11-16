from dataclasses import dataclass, fields


@dataclass
class Qwen3Config:
    attention_bias: bool = False
    bos_token_id: int = 151643
    eos_token_id: int | list[int] = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: dict | None = None
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, data: dict) -> "Qwen3Config":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

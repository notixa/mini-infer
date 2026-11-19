from typing import cast

import einops
import torch

from mini_infer.config.model_config.qwen3_config import Qwen3Config
from mini_infer.layers.activation import SiluAndMul
from mini_infer.layers.attention import Attention
from mini_infer.layers.layernorm import RMSNorm
from mini_infer.layers.rotary_embedding import RotaryEmbedding
from mini_infer.utils.context import get_context


class Qwen3Attention(torch.nn.Module):

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.qkv_proj = torch.nn.Linear(config.hidden_size,
                                        self.q_size + 2 * self.kv_size,
                                        bias=config.attention_bias)

        setattr(self.qkv_proj.weight, "weight_loader", self.weight_loader)  # noqa: B010

        self.o_proj = torch.nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(head_dim, config.max_position_embeddings,
                                          config.rope_theta)
        self.attn = Attention(self.num_heads, head_dim, head_dim**-0.5,
                              self.num_kv_heads)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = einops.rearrange(q,
                             "... (num_heads head_dim) -> ... num_heads head_dim",
                             num_heads=self.num_heads)
        k = einops.rearrange(k,
                             "... (num_kv_heads head_dim) -> ... num_kv_heads head_dim",
                             num_kv_heads=self.num_kv_heads)
        v = einops.rearrange(v,
                             "... (num_kv_heads head_dim) -> ... num_kv_heads head_dim",
                             num_kv_heads=self.num_kv_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        if loaded_shard_id == "q":
            offset, size = 0, self.q_size
        elif loaded_shard_id == "k":
            offset, size = self.q_size, self.kv_size
        elif loaded_shard_id == "v":
            offset, size = self.q_size + self.kv_size, self.kv_size
        else:
            raise ValueError(f"Unknown shard_id: {loaded_shard_id}")

        param_slice = param.data.narrow(0, offset, size)
        assert param_slice.shape == loaded_weight.shape
        param_slice.copy_(loaded_weight)


class Qwen3MLP(torch.nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_up_size = config.intermediate_size
        self.gate_up_proj = torch.nn.Linear(config.hidden_size,
                                            2 * config.intermediate_size,
                                            bias=False)

        setattr(self.gate_up_proj.weight, "weight_loader", self.weight_loader)

        self.down_proj = torch.nn.Linear(config.intermediate_size,
                                         config.hidden_size,
                                         bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor,
                      shard_id: int):
        if shard_id == 0:
            offset, size = 0, self.gate_up_size
        elif shard_id == 1:
            offset, size = self.gate_up_size, self.gate_up_size
        else:
            raise ValueError(f"Unknown shard_id: {shard_id}")

        param_slice = param.data.narrow(0, offset, size)
        assert param_slice.shape == loaded_weight.shape
        param_slice.copy_(loaded_weight)


class Qwen3DecoderLayer(torch.nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        residual = cast(torch.Tensor, residual)
        return hidden_states, residual


class Qwen3Model(torch.nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(torch.nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = torch.nn.Linear(config.hidden_size,
                                       config.vocab_size,
                                       bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
        }

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        return self.lm_head(hidden_states)

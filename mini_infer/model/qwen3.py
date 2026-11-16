from typing import cast

import einops
import torch

from mini_infer.config.model_config import Qwen3Config
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
        self.q_size_ = self.num_heads * head_dim
        self.kv_size_ = self.num_kv_heads * head_dim
        self.qkv_proj = torch.nn.Linear(config.hidden_size,
                                        self.q_size_ + 2 * self.kv_size_,
                                        bias=config.attention_bias)
        self.o_proj = torch.nn.Linear(self.q_size_, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(head_dim, config.max_position_embeddings,
                                          config.rope_theta)
        self.attn = Attention(self.num_heads, head_dim, head_dim**-0.5,
                              self.num_kv_heads)

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size_, self.kv_size_, self.kv_size_], dim=-1)
        q = einops.rearrange(q,
                             "... (q_size) -> ... num_heads head_dim",
                             num_heads=self.num_heads)
        k = einops.rearrange(k,
                             "... (k_size) -> ... num_kv_heads head_dim",
                             num_kv_heads=self.num_kv_heads)
        v = einops.rearrange(v,
                             "... (v_size) -> ... num_kv_heads head_dim",
                             num_kv_heads=self.num_kv_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Qwen3MLP(torch.nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(config.hidden_size,
                                            2 * config.intermediate_size,
                                            bias=False)
        self.down_proj = torch.nn.Linear(config.intermediate_size,
                                         config.hidden_size,
                                         bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


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

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill_ and context.cu_seqlens_q_ is not None:
            last_indices = context.cu_seqlens_q_[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        return self.lm_head(hidden_states)

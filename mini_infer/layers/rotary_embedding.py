import torch

from mini_infer.config.engine_config import DeviceConfig


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    cos_sin_cache: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        device = DeviceConfig.device
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        inv_freq = 1.0 / (base**(torch.arange(0, head_dim, 2, dtype=torch.float32) /
                                 head_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()),
                                  dim=-1).unsqueeze(1).to(device=device)
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim in (1, 2)
        assert q.ndim == positions.ndim + 2 and k.ndim == positions.ndim + 2
        assert q.shape[:positions.ndim] == positions.shape
        assert k.shape[:positions.ndim] == positions.shape
        assert q.shape[-1] == self.head_dim and k.shape[-1] == self.head_dim

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

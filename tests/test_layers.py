import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from mini_infer.layers.activation import SiluAndMul
from mini_infer.layers.layernorm import RMSNorm
from mini_infer.layers.rotary_embedding import RotaryEmbedding
from mini_infer.layers.sampler import Sampler


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(7, 1024), (5, 16, 128), (4, 192)])  # 192 tests non-power-of-2 mask
def test_rmsnorm_matches_qwen3(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    device = "cuda"
    dim = shape[-1]
    eps = 1e-6

    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(dim, device=device, dtype=dtype)

    norm = RMSNorm(dim, eps=eps, dtype=dtype)
    ref = Qwen3RMSNorm(dim, eps=eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
        ref.weight.copy_(weight)

    out = norm(x)
    ref_out = ref(x)

    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (2e-2, 2e-2)
    assert out.shape == ref_out.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(7, 1024), (4, 8, 128), (3, 192)])
def test_rmsnorm_with_residual_matches_reference(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    torch.manual_seed(1)
    device = "cuda"
    dim = shape[-1]
    eps = 1e-6

    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(dim, device=device, dtype=dtype)

    norm = RMSNorm(dim, eps=eps, dtype=dtype)
    ref = Qwen3RMSNorm(dim, eps=eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
        ref.weight.copy_(weight)

    expected_res = (x.float() + residual.float()).to(dtype)
    expected_out = ref(expected_res)

    out, new_res = norm(x, residual.clone())

    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (2e-2, 2e-2)
    torch.testing.assert_close(new_res, expected_res, atol=atol, rtol=rtol)
    torch.testing.assert_close(out, expected_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rotary_embedding_matches_qwen3(dtype: torch.dtype) -> None:
    torch.manual_seed(2)
    device = "cuda"
    head_dim = 128
    num_heads = 16
    num_kv_heads = 8
    max_pos = 2048
    rope_theta = 1000000.0

    positions = torch.tensor([0, 1, 17, 255, 1023], device=device, dtype=torch.int64)
    N = positions.shape[0]
    q = torch.randn((N, num_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((N, num_kv_heads, head_dim), device=device, dtype=dtype)

    rope = RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        base=rope_theta,
    )
    q_out, k_out = rope(positions, q, k)

    hf_cfg = Qwen3Config(
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        rope_theta=rope_theta,
    )
    hf_rope = Qwen3RotaryEmbedding(hf_cfg).to(device=device)
    # HF expects [batch, seq_len, ...] and position_ids [batch, seq_len]
    q_hf = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, N, head_dim]
    k_hf = k.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, N, head_dim]
    cos, sin = hf_rope(q_hf, positions.unsqueeze(0))
    q_ref, k_ref = apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    q_ref = q_ref.transpose(1, 2).squeeze(0)
    k_ref = k_ref.transpose(1, 2).squeeze(0)

    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (2e-2, 2e-2)
    assert q_out.dtype == dtype and k_out.dtype == dtype
    torch.testing.assert_close(q_out, q_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_out, k_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(3)
    device = "cuda"
    x = torch.randn((11, 6144), device=device, dtype=dtype)
    gate, up = x.chunk(2, dim=-1)
    expected = F.silu(gate) * up

    act = SiluAndMul()
    out = act(x)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_sampler_greedy_mixed_and_low_temp() -> None:
    torch.manual_seed(4)
    device = "cuda"
    sampler = Sampler()

    logits = torch.tensor(
        [
            [0.1, 5.0, -1.0, 2.0],
            [3.0, 0.2, 8.0, 1.0],
            [-2.0, -3.0, -1.0, 4.0],
            [10.0, -5.0, -5.0, -5.0],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    expected_greedy = torch.tensor([1, 2, 3, 0], device=device)

    # 1. None / scalar 0.0 / tensor of zeros -> exact greedy
    assert torch.equal(sampler(logits, None), expected_greedy)
    assert torch.equal(sampler(logits, 0.0), expected_greedy)
    assert torch.equal(
        sampler(logits, torch.zeros(4, device=device, dtype=torch.float32)),
        expected_greedy,
    )

    # 2. Mixed batch: greedy at idx 0 & 2, stochastic at idx 1 & 3 (no NaN, valid range)
    temps = torch.tensor([0.0, 0.7, 0.0, 0.05], device=device, dtype=torch.float32)
    out = sampler(logits, temps)
    assert out.shape == (4,)
    assert out[0] == expected_greedy[0]
    assert out[2] == expected_greedy[2]
    assert 0 <= out[1].item() < 4
    assert out[3] == 0  # peaked logit 10.0 with temp=0.05

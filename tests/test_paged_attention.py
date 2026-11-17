import pytest
import torch
import torch.nn.functional as F

from mini_infer.config.engine_config import CacheConfig
from mini_infer.engine.block_manager import BlockManager
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sequence import Sequence
from mini_infer.layers.attention import Attention, store_kvcache
from mini_infer.utils.context import get_context, reset_context


def sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference causal GQA attention for a single sequence: inputs [seq_len, heads, head_dim]."""
    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(
        q_b,
        k_b,
        v_b,
        is_causal=True,
        scale=scale,
        enable_gqa=True,
    )
    return out.squeeze(0).transpose(0, 1)


def test_store_kvcache_noncontiguous_nonpow2_and_padding_slot() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_blocks = 2
    block_size = 256
    num_heads = 6
    num_kv_heads = 3  # D = 3 * 64 = 192 (non-power-of-2)
    head_dim = 64
    num_tokens = 5

    qkv = torch.randn(
        num_tokens,
        (num_heads + 2 * num_kv_heads) * head_dim,
        device=device,
        dtype=dtype,
    )
    _, k, v = qkv.split(
        [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim],
        dim=-1,
    )
    k = k.view(num_tokens, num_kv_heads, head_dim)
    v = v.view(num_tokens, num_kv_heads, head_dim)
    assert not k.is_contiguous()

    k_cache = torch.full(
        (num_blocks, block_size, num_kv_heads, head_dim), -7.0, device=device, dtype=dtype
    )
    v_cache = torch.full_like(k_cache, -9.0)

    slots = [3, 257, -1, 0, 511]
    slot_mapping = torch.tensor(slots, device=device, dtype=torch.int32)

    store_kvcache(k, v, k_cache, v_cache, slot_mapping)

    for tok_idx, slot in enumerate(slots):
        if slot == -1:
            continue
        bid, off = divmod(slot, block_size)
        assert torch.equal(k_cache[bid, off], k[tok_idx])
        assert torch.equal(v_cache[bid, off], v[tok_idx])

    written_set = {s for s in slots if s != -1}
    for bid in range(num_blocks):
        for off in range(block_size):
            slot = bid * block_size + off
            if slot not in written_set:
                assert torch.all(k_cache[bid, off] == -7.0)
                assert torch.all(v_cache[bid, off] == -9.0)


def test_model_runner_prepare_prefill_and_decode_context() -> None:
    bs = CacheConfig.block_size
    bm = BlockManager(num_blocks=6)
    runner = ModelRunner()

    s0 = Sequence(list(range(bs + 44)))  # 300 tokens -> 2 blocks
    s1 = Sequence(list(range(bs)))       # 256 tokens -> 1 block
    bm.allocate(s0)
    bm.allocate(s1)

    try:
        runner.prepare_prefill_context([s0, s1])
        ctx = get_context()
        assert ctx.is_prefill is True
        assert ctx.cu_seqlens_q.tolist() == [0, 300, 556]
        assert ctx.cu_seqlens_k.tolist() == [0, 300, 556]
        assert ctx.max_seqlen_q == 300
        assert ctx.max_seqlen_k == 300
        assert ctx.block_tables is None

        expected_slots = (
            list(range(s0.block_table[0] * bs, s0.block_table[0] * bs + bs))
            + list(range(s0.block_table[1] * bs, s0.block_table[1] * bs + 44))
            + list(range(s1.block_table[0] * bs, s1.block_table[0] * bs + bs))
        )
        assert ctx.slot_mapping.tolist() == expected_slots

        s0.append_token(900)
        s1.append_token(901)
        bm.may_append(s0)
        bm.may_append(s1)

        runner.prepare_decode_context([s0, s1])
        ctx = get_context()
        assert ctx.is_prefill is False
        assert ctx.context_lens.tolist() == [301, 257]
        assert ctx.block_tables.shape == (2, 2)
        assert ctx.block_tables.tolist() == [s0.block_table, s1.block_table]
        assert ctx.slot_mapping.tolist() == [
            s0.block_table[-1] * bs + 44,
            s1.block_table[-1] * bs + 0,
        ]
    finally:
        reset_context()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_attention_prefill_and_decode_matches_sdpa(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    device = "cuda"
    bs = CacheConfig.block_size
    num_blocks = 8
    num_heads = 8
    num_kv_heads = 2
    head_dim = 64
    scale = head_dim ** -0.5

    bm = BlockManager(num_blocks=num_blocks)
    runner = ModelRunner()
    attn = Attention(num_heads=num_heads, head_dim=head_dim, scale=scale, num_kv_heads=num_kv_heads)
    k_cache = torch.empty((num_blocks, bs, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)
    attn.k_cache = k_cache
    attn.v_cache = v_cache

    dummy = Sequence(list(range(bs)))
    s0 = Sequence(list(range(bs - 1)))
    s1 = Sequence(list(range(bs + 4)))
    bm.allocate(s0)
    bm.allocate(dummy)
    bm.allocate(s1)
    bm.deallocate(dummy)

    decode_steps = 4
    total_lens = [len(s0.token_ids) + decode_steps, len(s1.token_ids) + decode_steps]
    q_full = [torch.randn(L, num_heads, head_dim, device=device, dtype=dtype) for L in total_lens]
    k_full = [torch.randn(L, num_kv_heads, head_dim, device=device, dtype=dtype) for L in total_lens]
    v_full = [torch.randn(L, num_kv_heads, head_dim, device=device, dtype=dtype) for L in total_lens]

    ref_full = [sdpa_reference(q_full[i], k_full[i], v_full[i], scale) for i in range(2)]

    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (1.5e-2, 1.5e-2)

    try:
        len0, len1 = len(s0.token_ids), len(s1.token_ids)
        q_prefill = torch.cat([q_full[0][:len0], q_full[1][:len1]], dim=0)
        k_prefill = torch.cat([k_full[0][:len0], k_full[1][:len1]], dim=0)
        v_prefill = torch.cat([v_full[0][:len0], v_full[1][:len1]], dim=0)

        runner.prepare_prefill_context([s0, s1])
        out_prefill = attn(q_prefill, k_prefill, v_prefill)
        reset_context()

        assert out_prefill.shape == (len0 + len1, num_heads, head_dim)
        torch.testing.assert_close(out_prefill[:len0], ref_full[0][:len0], atol=atol, rtol=rtol)
        torch.testing.assert_close(out_prefill[len0:], ref_full[1][:len1], atol=atol, rtol=rtol)

        for step in range(decode_steps):
            pos0 = len0 + step
            pos1 = len1 + step
            s0.append_token(1000 + step)
            s1.append_token(2000 + step)
            bm.may_append(s0)
            bm.may_append(s1)

            q_dec = torch.stack([q_full[0][pos0], q_full[1][pos1]], dim=0)
            k_dec = torch.stack([k_full[0][pos0], k_full[1][pos1]], dim=0)
            v_dec = torch.stack([v_full[0][pos0], v_full[1][pos1]], dim=0)

            runner.prepare_decode_context([s0, s1])
            out_dec = attn(q_dec, k_dec, v_dec)
            reset_context()

            assert out_dec.shape == (2, num_heads, head_dim)
            torch.testing.assert_close(out_dec[0], ref_full[0][pos0], atol=atol, rtol=rtol)
            torch.testing.assert_close(out_dec[1], ref_full[1][pos1], atol=atol, rtol=rtol)
    finally:
        reset_context()
        bm.deallocate(s0)
        bm.deallocate(s1)
        assert len(bm.used_block_ids) == 0
        assert len(bm.free_block_ids) == num_blocks

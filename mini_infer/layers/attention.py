from typing import cast

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from mini_infer.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    stride_kn,
    stride_vn,
    stride_kcn,
    stride_vcn,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    key = tl.load(key_ptr + pid * stride_kn + offsets, mask=mask)
    value = tl.load(value_ptr + pid * stride_vn + offsets, mask=mask)
    tl.store(k_cache_ptr + stride_kcn * slot + offsets, key, mask=mask)
    tl.store(v_cache_ptr + stride_vcn * slot + offsets, value, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
                  v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_kv_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and key.stride(1) == head_dim
    assert k_cache.is_contiguous()
    D = num_kv_heads * head_dim
    BLOCK_D = triton.next_power_of_2(D)
    grid = (N, )
    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        key.stride(0),
        value.stride(0),
        k_cache.stride(1),
        v_cache.stride(1),
        slot_mapping,
        D=D,  # pyright: ignore[reportArgumentType]
        BLOCK_D=BLOCK_D,  # pyright: ignore[reportArgumentType]
    )


class Attention(torch.nn.Module):

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        _, num_heads, _ = q.shape
        num_kv_heads = k.shape[1]
        assert num_heads % num_kv_heads == 0
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert q.dtype is torch.float16 or q.dtype is torch.bfloat16
        context = get_context()
        assert context.slot_mapping is not None
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k = self.k_cache
                v = self.v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                context.max_seqlen_q,
                context.max_seqlen_k,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            o = cast(torch.Tensor, o)
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                self.k_cache,
                self.v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            o = cast(torch.Tensor, o)
            o = o.squeeze(1)
        return o

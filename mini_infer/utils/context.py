from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill_: bool = False
    cu_seqlens_q_: torch.Tensor | None = None
    cu_seqlens_k_: torch.Tensor | None = None
    max_seqlen_q_: int = 0
    max_seqlen_k_: int = 0
    slot_mapping_: torch.Tensor | None = None
    context_lens_: torch.Tensor | None = None
    block_tables_: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    slot_mapping: torch.Tensor | None,
    context_lens: torch.Tensor | None,
    block_tables: torch.Tensor | None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill_=is_prefill,
        cu_seqlens_q_=cu_seqlens_q,
        cu_seqlens_k_=cu_seqlens_k,
        max_seqlen_q_=max_seqlen_q,
        max_seqlen_k_=max_seqlen_k,
        slot_mapping_=slot_mapping,
        context_lens_=context_lens,
        block_tables_=block_tables,
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()

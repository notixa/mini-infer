from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


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
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()

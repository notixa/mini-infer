import torch

from mini_infer.config import CacheConfig, DeviceConfig
from mini_infer.engine.sequence import Sequence
from mini_infer.utils.context import set_context


class ModelRunner:

    def __init__(self,
                 cache_config: CacheConfig | None = None,
                 device_config: DeviceConfig | None = None):
        if cache_config is None:
            cache_config = CacheConfig()
        if device_config is None:
            device_config = DeviceConfig()
        self.cache_config_ = cache_config
        self.device_config_ = device_config

    def prepare_prefill_context(self, seqs: list[Sequence]):
        sum_seqlens_q = 0
        sum_seqlens_k = 0
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        max_num_blocks = 0
        slot_mapping = []
        for seq in seqs:
            seqlen_q = len(seq.token_ids_) - seq.num_cached_tokens_
            seqlen_k = len(seq.token_ids_)
            sum_seqlens_q += seqlen_q
            sum_seqlens_k += seqlen_k
            cu_seqlens_q.append(sum_seqlens_q)
            cu_seqlens_k.append(sum_seqlens_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            max_num_blocks = max(max_num_blocks, seq.num_blocks)
            for i in range(seq.num_cached_tokens_ // self.cache_config_.block_size_,
                           seq.num_blocks):
                bid = seq.block_table_[i]
                if i != seq.num_blocks - 1:
                    valid_num_tokens = self.cache_config_.block_size_
                else:
                    valid_num_tokens = seq.last_block_num_tokens
                base_offset = bid * self.cache_config_.block_size_
                slot_mapping.extend(range(base_offset, base_offset + valid_num_tokens))

        block_tables = [
            seq.block_table_ + [-1] * (max_num_blocks - seq.num_blocks) for seq in seqs
        ]
        cu_seqlens_q = torch.tensor(cu_seqlens_q,
                                    dtype=torch.int32,
                                    device=self.device_config_.device_)
        cu_seqlens_k = torch.tensor(cu_seqlens_k,
                                    dtype=torch.int32,
                                    device=self.device_config_.device_)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int32,
                                    device=self.device_config_.device_)
        block_tables = torch.tensor(block_tables,
                                    dtype=torch.int32,
                                    device=self.device_config_.device_)
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=block_tables,
        )

    def prepare_decode_context(self, seqs: list[Sequence]):
        slot_mapping = []
        context_lens = []
        max_num_blocks = 0
        for seq in seqs:
            seqlen_k = len(seq.token_ids_)
            context_lens.append(seqlen_k)
            max_num_blocks = max(max_num_blocks, seq.num_blocks)
            slot_mapping.append(seq.block_table_[-1] * self.cache_config_.block_size_ +
                                seq.last_block_num_tokens - 1)

        block_tables = [
            seq.block_table_ + [-1] * (max_num_blocks - seq.num_blocks) for seq in seqs
        ]
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int32,
                                    device=self.device_config_.device_)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int32,
                                    device=self.device_config_.device_)
        block_tables = torch.tensor(block_tables,
                                    dtype=torch.int32,
                                    device=self.device_config_.device_)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

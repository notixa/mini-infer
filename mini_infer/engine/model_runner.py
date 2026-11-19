import os

import torch

from mini_infer.config.engine_config import CacheConfig, DeviceConfig, ScheduleConfig
from mini_infer.config.model_config.model_config import get_config
from mini_infer.engine.sequence import Sequence
from mini_infer.layers.sampler import Sampler
from mini_infer.model.model import get_model
from mini_infer.utils.context import reset_context, set_context
from mini_infer.utils.loader import load_weight


class ModelRunner:

    def __init__(self,
                 model_arch: str,
                 model_path: str | os.PathLike,
                 cache_config: CacheConfig | None = None,
                 schedule_config: ScheduleConfig | None = None,
                 device_config: DeviceConfig | None = None,
                 gpu_memory_utilization=0.85):
        if cache_config is None:
            cache_config = CacheConfig()
        if schedule_config is None:
            schedule_config = ScheduleConfig()
        if device_config is None:
            device_config = DeviceConfig()
        self.cache_config = cache_config
        self.schedule_config = schedule_config
        self.device_config = device_config
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model_config = get_config(model_arch, model_path)
        self.model_dtype: torch.dtype = getattr(torch, self.model_config.torch_dtype)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_dtype)
        try:
            self.model = get_model(model_arch,
                                   self.model_config).to(self.device_config.device)
        finally:
            torch.set_default_dtype(prev_dtype)
        load_weight(self.model, model_path)
        self.model = self.model.eval()
        self.sampler = Sampler()
        self.warmup_model()
        self.num_blocks = self.allocate_kv_cache()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.schedule_config.max_num_batched_tokens, self.schedule_config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len,
                       self.schedule_config.max_num_seqs)
        seqs = [Sequence([0 for _ in range(seq_len)], 0.1) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self) -> int:
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        avail = int(total * self.gpu_memory_utilization - (total - free) - peak +
                    current)
        block_bytes = 2 * self.model_config.num_hidden_layers * self.cache_config.block_size * self.model_config.num_key_value_heads * self.model_config.head_dim * self.model_dtype.itemsize
        num_blocks = avail // block_bytes
        assert num_blocks > 0
        self.kv_cache = torch.empty([
            2, self.model_config.num_hidden_layers, num_blocks,
            self.cache_config.block_size, self.model_config.num_key_value_heads,
            self.model_config.head_dim
        ],
                                    device=self.device_config.device,
                                    dtype=self.model_dtype)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        return num_blocks

    def prepare_prefill_context(self, seqs: list[Sequence]):
        block_size = self.cache_config.block_size
        sum_seqlens_q = 0
        sum_seqlens_k = 0
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        max_num_blocks = 0
        slot_mapping = []
        input_ids = []
        positions = []
        for seq in seqs:
            start_pos = seq.num_computed_tokens
            seqlen_q = seq.num_scheduled_tokens if seq.num_scheduled_tokens > 0 else len(
                seq.token_ids) - start_pos
            end_pos = start_pos + seqlen_q
            seqlen_k = end_pos
            input_ids.extend(seq.token_ids[start_pos:end_pos])
            positions.extend([id for id in range(start_pos, end_pos)])
            sum_seqlens_q += seqlen_q
            sum_seqlens_k += seqlen_k
            cu_seqlens_q.append(sum_seqlens_q)
            cu_seqlens_k.append(sum_seqlens_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            max_num_blocks = max(max_num_blocks, len(seq.block_table))
            if not seq.block_table: continue
            for i in range(start_pos, end_pos):
                bid = seq.block_table[i // block_size]
                base_offset = bid * self.cache_config.block_size
                slot_mapping.append(base_offset + i % block_size)

        block_tables = [
            seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
            for seq in seqs
        ]
        cu_seqlens_q = torch.tensor(cu_seqlens_q,
                                    dtype=torch.int32,
                                    device=self.device_config.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k,
                                    dtype=torch.int32,
                                    device=self.device_config.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int32,
                                    device=self.device_config.device)
        if sum_seqlens_q < sum_seqlens_k:
            block_tables = torch.tensor(block_tables,
                                        dtype=torch.int32,
                                        device=self.device_config.device)
        else:
            block_tables = None
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
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int64,
                                 device=self.device_config.device)
        positions = torch.tensor(positions,
                                 dtype=torch.int64,
                                 device=self.device_config.device)
        return input_ids, positions

    def prepare_decode_context(self, seqs: list[Sequence]):
        slot_mapping = []
        context_lens = []
        input_ids = []
        positions = []
        max_num_blocks = 0
        for seq in seqs:
            seqlen_k = len(seq.token_ids)
            input_ids.append(seq.token_ids[-1])
            positions.append(len(seq.token_ids) - 1)
            context_lens.append(seqlen_k)
            max_num_blocks = max(max_num_blocks, seq.num_blocks)
            slot_mapping.append(seq.block_table[-1] * self.cache_config.block_size +
                                seq.last_block_num_tokens - 1)

        block_tables = [
            seq.block_table + [-1] * (max_num_blocks - seq.num_blocks) for seq in seqs
        ]
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int32,
                                    device=self.device_config.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.int32,
                                    device=self.device_config.device)
        block_tables = torch.tensor(block_tables,
                                    dtype=torch.int32,
                                    device=self.device_config.device)
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
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int64,
                                 device=self.device_config.device)
        positions = torch.tensor(positions,
                                 dtype=torch.int64,
                                 device=self.device_config.device)
        return input_ids, positions

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool):
        input_ids, positions = self.prepare_prefill_context(
            seqs) if is_prefill else self.prepare_decode_context(seqs)
        hidden = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden)
        temperatures = torch.tensor([seq.temperature for seq in seqs],
                                    dtype=torch.float32,
                                    device=self.device_config.device)
        sample_tokens = self.sampler(logits, temperatures)
        reset_context()
        return sample_tokens.tolist()

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheConfig:
    block_size_: int = 16


@dataclass(frozen=True)
class ScheduleConfig:
    max_num_seqs_: int = 512
    max_num_batched_tokens_: int = 1024
    eos_token_id_: int = -1

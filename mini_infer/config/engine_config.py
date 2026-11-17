from dataclasses import dataclass


@dataclass(frozen=True)
class CacheConfig:
    block_size: int = 256


@dataclass(frozen=True)
class ScheduleConfig:
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 1024
    eos_token_id: int = -1
    max_model_len: int = 4096


@dataclass(frozen=True)
class DeviceConfig:
    device: str = "cuda"

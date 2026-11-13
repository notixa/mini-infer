from dataclasses import dataclass


@dataclass(frozen=True)
class CacheConfig:
    block_size_: int = 16

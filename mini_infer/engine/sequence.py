from enum import Enum, auto

from mini_infer.config import CacheConfig, ScheduleConfig
from mini_infer.utils.counter import Counter


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    _seq_id_counter = Counter(start=0)

    def __init__(self,
                 token_ids: list[int],
                 max_tokens: int = 64,
                 ignore_eos: bool = False):
        assert len(token_ids) <= ScheduleConfig.max_num_batched_tokens_
        self.seq_id_: int = Sequence._seq_id_counter()
        self.token_ids_: list[int] = list(token_ids)
        self.num_prompt_tokens_: int = len(token_ids)
        self.status_: SequenceStatus = SequenceStatus.WAITING
        self.block_table_: list[int] = []
        self.num_cached_tokens_: int = 0
        self.num_scheduled_tokens: int = 0
        self.max_tokens_ = max_tokens
        self.ignore_eos_ = ignore_eos

    def append_token(self, token_id: int):
        self.token_ids_.append(token_id)

    def append_block_table(self, block_id: int):
        self.block_table_.append(block_id)

    def block(self, i: int):
        block_size = CacheConfig.block_size_
        return self.token_ids_[i * block_size:(i + 1) * block_size]

    @property
    def num_blocks(self) -> int:
        block_size = CacheConfig.block_size_
        return (len(self.token_ids_) + block_size - 1) // block_size

    @property
    def last_block_num_tokens(self) -> int:
        if not self.token_ids_:
            return 0
        block_size = CacheConfig.block_size_
        return len(self.token_ids_) - (self.num_blocks - 1) * block_size

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids_) - self.num_prompt_tokens_

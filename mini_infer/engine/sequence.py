from enum import Enum, auto

from config import CacheConfig
from utils.counter import Counter


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    _seq_id_counter = Counter(start=0)

    def __init__(self, token_ids: list[int]):
        self.seq_id_: int = Sequence._seq_id_counter()
        self.token_ids_: list[int] = token_ids
        self.status_: SequenceStatus = SequenceStatus.WAITING
        self.block_table_: list[int] = []
        self.num_cached_tokens_: int = 0
        self.num_scheduled_tokens: int = 0

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

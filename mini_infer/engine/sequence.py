from enum import Enum, auto

from mini_infer.config.engine_config import CacheConfig, ScheduleConfig
from mini_infer.utils.counter import Counter


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    _seq_id_counter = Counter(start=0)

    def __init__(self,
                 token_ids: list[int],
                 temperature: float,
                 max_tokens: int = 64,
                 ignore_eos: bool = False):
        assert len(token_ids) <= ScheduleConfig.max_model_len
        self.seq_id: int = Sequence._seq_id_counter()
        self.temperature = temperature
        self.token_ids: list[int] = list(token_ids)
        self.num_prompt_tokens: int = len(token_ids)
        self.status: SequenceStatus = SequenceStatus.WAITING
        self.block_table: list[int] = []
        self.num_hashed_blocks: int = 0
        self.num_scheduled_tokens: int = 0
        self.num_computed_tokens: int = 0
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)

    def append_block_table(self, block_id: int):
        self.block_table.append(block_id)

    def block(self, i: int):
        block_size = CacheConfig.block_size
        return self.token_ids[i * block_size:(i + 1) * block_size]

    @property
    def num_blocks(self) -> int:
        block_size = CacheConfig.block_size
        return (len(self.token_ids) + block_size - 1) // block_size

    @property
    def last_block_num_tokens(self) -> int:
        if not self.token_ids:
            return 0
        block_size = CacheConfig.block_size
        return len(self.token_ids) - (self.num_blocks - 1) * block_size

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

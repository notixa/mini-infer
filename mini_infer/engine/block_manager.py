from collections import deque

from config import CacheConfig
from engine.sequence import Sequence


class Block:

    def __init__(self, block_id: int):
        self.block_id_: int = block_id
        self.ref_count_: int = 0
        self.hash_: int = -1
        self.token_ids_: list[int] = []


class BlockManager:

    def __init__(self, num_blocks: int):
        self.blocks_ = [Block(i) for i in range(num_blocks)]
        self.free_block_ids_: deque[int] = deque(range(num_blocks))
        self.used_block_ids_: set = set()

    def _alloc_block(self) -> int:
        block_id = self.free_block_ids_.popleft()
        self.used_block_ids_.add(block_id)
        self.blocks_[block_id].ref_count_ = 1
        return block_id

    def can_allocate(self, seq: Sequence):
        # TODO(Prefix Cache)
        return len(self.free_block_ids_) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert len(seq.block_table_) == 0
        for _ in range(seq.num_blocks):
            seq.append_block_table(self._alloc_block())

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table_):
            self.blocks_[block_id].ref_count_ -= 1
            if self.blocks_[block_id].ref_count_ == 0:
                self.used_block_ids_.remove(block_id)
                self.free_block_ids_.append(block_id)

        seq.block_table_.clear()

    def can_append(self, seq: Sequence):
        block_size = CacheConfig.block_size_
        if len(seq.token_ids_) % block_size == 1:
            return not len(self.free_block_ids_) <= 0
        return True

    def may_append(self, seq: Sequence):
        block_size = CacheConfig.block_size_
        if len(seq.token_ids_) % block_size == 1:
            seq.block_table_.append(self._alloc_block())

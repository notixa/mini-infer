from collections import deque

from mini_infer.config.engine_config import CacheConfig
from mini_infer.engine.sequence import Sequence


class Block:

    def __init__(self, block_id: int):
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.hash: int = -1
        self.token_ids: list[int] = []


class BlockManager:

    def __init__(self, num_blocks: int, cache_config: CacheConfig | None = None):
        if cache_config is None:
            cache_config = CacheConfig()
        self.cache_config: CacheConfig = cache_config
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set = set()

    def _alloc_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        self.used_block_ids.add(block_id)
        self.blocks[block_id].ref_count = 1
        return block_id

    def can_allocate(self, seq: Sequence):
        # TODO(Prefix Cache)
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert len(seq.block_table) == 0
        for _ in range(seq.num_blocks):
            seq.append_block_table(self._alloc_block())

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)

        seq.block_table.clear()

    def can_append(self, seq: Sequence):
        block_size = self.cache_config.block_size
        if len(seq.token_ids) % block_size == 1:
            return not len(self.free_block_ids) <= 0
        return True

    def may_append(self, seq: Sequence):
        block_size = self.cache_config.block_size
        if len(seq.token_ids) % block_size == 1:
            seq.block_table.append(self._alloc_block())

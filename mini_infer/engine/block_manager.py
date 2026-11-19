from collections import deque

import numpy as np
import xxhash

from mini_infer.config.engine_config import CacheConfig
from mini_infer.engine.sequence import Sequence


class Block:

    def __init__(self, block_id: int):
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.hash: int = -1
        self.token_ids: list[int] = []

    def reset(self) -> int:
        hash = self.hash
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        return hash


class BlockManager:

    def __init__(self, num_blocks: int, cache_config: CacheConfig | None = None):
        if cache_config is None:
            cache_config = CacheConfig()
        self.cache_config: CacheConfig = cache_config
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set = set()
        self.hash_to_block_id: dict[int, int] = {}

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def hash_blocks(self, seq: Sequence):
        block_size = self.cache_config.block_size
        start = seq.num_hashed_blocks
        end = seq.num_computed_tokens // block_size
        if start < end:
            h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
            for i in range(start, end):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block_id = seq.block_table[i]
                block = self.blocks[block_id]
                block.hash = h
                block.token_ids = token_ids
                self.hash_to_block_id[h] = block_id
                seq.num_hashed_blocks += 1

    def _alloc_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        old_hash = self.blocks[block_id].reset()
        if old_hash != -1 and self.hash_to_block_id.get(old_hash) == block_id:
            del self.hash_to_block_id[old_hash]
        self.used_block_ids.add(block_id)
        self.blocks[block_id].ref_count = 1
        return block_id

    def can_allocate(self, seq: Sequence) -> int:
        num_cached_blocks = 0
        num_reused_used_blocks = 0
        h = -1
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                num_cached_blocks += 1
                if block_id in self.used_block_ids:
                    num_reused_used_blocks += 1
            else:
                break

        if seq.num_blocks - num_reused_used_blocks > len(self.free_block_ids):
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence):
        assert len(seq.block_table) == 0
        h = -1
        cached_block_num = 0
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                if block_id in self.used_block_ids:
                    self.blocks[block_id].ref_count += 1
                elif block_id in self.free_block_ids:
                    self.blocks[block_id].ref_count = 1
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                seq.append_block_table(block_id)
                cached_block_num += 1
            else:
                break

        for _ in range(cached_block_num, seq.num_blocks):
            seq.append_block_table(self._alloc_block())

        seq.num_hashed_blocks = cached_block_num
        seq.num_computed_tokens = cached_block_num * self.cache_config.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)

        seq.block_table.clear()
        seq.num_hashed_blocks = 0
        seq.num_computed_tokens = 0
        seq.num_scheduled_tokens = 0

    def can_append(self, seq: Sequence):
        if len(seq.block_table) < seq.num_blocks:
            return len(self.free_block_ids) >= seq.num_blocks - len(seq.block_table)
        return True

    def may_append(self, seq: Sequence):
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self._alloc_block())

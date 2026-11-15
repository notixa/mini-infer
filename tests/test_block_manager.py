from mini_infer.config import CacheConfig
from mini_infer.engine.block_manager import BlockManager
from mini_infer.engine.sequence import Sequence, SequenceStatus


def assert_block_pool_invariant(bm: BlockManager, total_blocks: int) -> None:
    """Verify block pool conservation, disjointness, and ref_count consistency."""
    free_set = set(bm.free_block_ids_)
    used_set = set(bm.used_block_ids_)

    assert len(bm.free_block_ids_) == len(free_set), "Duplicate block_id in free_block_ids_"
    assert len(bm.free_block_ids_) + len(bm.used_block_ids_) == total_blocks
    assert free_set.isdisjoint(used_set), "Block cannot be both free and used"
    assert (free_set | used_set) == set(range(total_blocks))

    for bid in free_set:
        assert bm.blocks_[bid].ref_count_ == 0
    for bid in used_set:
        assert bm.blocks_[bid].ref_count_ >= 1


def test_sequence_id_and_initial_state() -> None:
    s1 = Sequence([1, 2, 3])
    s2 = Sequence([4, 5])

    assert s2.seq_id_ == s1.seq_id_ + 1
    assert s1.status_ == SequenceStatus.WAITING
    assert s1.block_table_ == []
    assert s1.num_cached_tokens_ == 0
    assert s1.num_scheduled_tokens == 0


def test_sequence_block_math_boundaries() -> None:
    bs = CacheConfig.block_size_  # 16

    s_one = Sequence(list(range(1)))
    assert s_one.num_blocks == 1
    assert s_one.last_block_num_tokens == 1
    assert s_one.block(0) == [0]

    s_exact = Sequence(list(range(bs)))
    assert s_exact.num_blocks == 1
    assert s_exact.last_block_num_tokens == bs
    assert s_exact.block(0) == list(range(bs))

    s_cross = Sequence(list(range(bs + 1)))
    assert s_cross.num_blocks == 2
    assert s_cross.last_block_num_tokens == 1
    assert s_cross.block(0) == list(range(bs))
    assert s_cross.block(1) == [bs]

    s_exact.append_token(999)
    assert s_exact.num_blocks == 2
    assert s_exact.last_block_num_tokens == 1
    assert s_exact.block(1) == [999]


def test_can_allocate_exact_and_insufficient_capacity() -> None:
    bs = CacheConfig.block_size_
    bm = BlockManager(num_blocks=2)

    seq_1_block = Sequence(list(range(bs)))          # needs 1 block
    seq_2_blocks = Sequence(list(range(bs * 2)))     # needs 2 blocks (exact match)
    seq_3_blocks = Sequence(list(range(bs * 2 + 1))) # needs 3 blocks (exceeds pool)

    assert bm.can_allocate(seq_1_block) is True
    assert bm.can_allocate(seq_2_blocks) is True
    assert bm.can_allocate(seq_3_blocks) is False


def test_allocate_and_deallocate_conservation_and_reverse_free() -> None:
    bs = CacheConfig.block_size_
    total_blocks = 8
    bm = BlockManager(num_blocks=total_blocks)
    assert_block_pool_invariant(bm, total_blocks)

    seq = Sequence(list(range(bs * 2 + 3)))  # needs 3 blocks
    bm.allocate(seq)

    assert len(seq.block_table_) == 3
    assert seq.block_table_ == [0, 1, 2]
    assert_block_pool_invariant(bm, total_blocks)

    bm.deallocate(seq)

    assert seq.block_table_ == []
    assert_block_pool_invariant(bm, total_blocks)
    # Reverse free appends [2, 1, 0] to the tail of free_block_ids_
    assert list(bm.free_block_ids_)[-3:] == [2, 1, 0]


def test_boundary_crossing_can_append_and_may_append() -> None:
    bs = CacheConfig.block_size_
    total_blocks = 3
    bm = BlockManager(num_blocks=total_blocks)

    seq = Sequence(list(range(bs - 1)))  # 15 tokens -> 1 block
    bm.allocate(seq)
    assert len(seq.block_table_) == 1
    assert_block_pool_invariant(bm, total_blocks)

    # 16th token: still fits in block 0, no new physical block needed
    seq.append_token(100)
    assert bm.can_append(seq) is True
    bm.may_append(seq)
    assert len(seq.block_table_) == 1
    assert_block_pool_invariant(bm, total_blocks)

    # 17th token: crosses into block 1, requires 1 new physical block
    seq.append_token(101)
    assert bm.can_append(seq) is True
    bm.may_append(seq)
    assert len(seq.block_table_) == 2
    assert len(set(seq.block_table_)) == 2
    assert_block_pool_invariant(bm, total_blocks)


def test_can_append_when_pool_exhausted() -> None:
    bs = CacheConfig.block_size_
    total_blocks = 1
    bm = BlockManager(num_blocks=total_blocks)

    seq = Sequence(list(range(bs - 1)))  # uses the only block (block 0)
    bm.allocate(seq)
    assert len(bm.free_block_ids_) == 0

    # 16th token does not require a new block
    seq.append_token(100)
    assert bm.can_append(seq) is True

    # 17th token requires a new block, but pool is empty
    seq.append_token(101)
    assert bm.can_append(seq) is False


def test_multi_sequence_allocation_and_out_of_order_deallocate() -> None:
    bs = CacheConfig.block_size_
    total_blocks = 10
    bm = BlockManager(num_blocks=total_blocks)

    seqs = [
        Sequence(list(range(5))),           # 1 block
        Sequence(list(range(bs))),          # 1 block
        Sequence(list(range(bs * 2 + 1))),  # 3 blocks
    ]

    for seq in seqs:
        assert bm.can_allocate(seq) is True
        bm.allocate(seq)
        assert_block_pool_invariant(bm, total_blocks)

    assert len(bm.used_block_ids_) == 5
    assert len(bm.free_block_ids_) == 5

    # Simulate decode steps on seqs[1] crossing block boundary
    seqs[1].append_token(999)
    assert bm.can_append(seqs[1]) is True
    bm.may_append(seqs[1])
    assert len(seqs[1].block_table_) == 2
    assert len(bm.used_block_ids_) == 6
    assert_block_pool_invariant(bm, total_blocks)

    # Deallocate in scrambled order
    for idx in (2, 0, 1):
        bm.deallocate(seqs[idx])
        assert seqs[idx].block_table_ == []
        assert_block_pool_invariant(bm, total_blocks)

    assert len(bm.free_block_ids_) == total_blocks
    assert len(bm.used_block_ids_) == 0

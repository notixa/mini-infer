import gc
from pathlib import Path

import pytest
import torch

from mini_infer.config.engine_config import CacheConfig, ScheduleConfig
from mini_infer.engine.block_manager import BlockManager
from mini_infer.engine.llm_engine import LLMEngine, SamplingParams
from mini_infer.engine.scheduler import Scheduler
from mini_infer.engine.sequence import Sequence

MODEL_PATH = Path("/home/notixa/huggingface/Qwen3-0.6B")


def assert_block_pool_invariant(bm: BlockManager, total_blocks: int) -> None:
    free_set = set(bm.free_block_ids)
    used_set = set(bm.used_block_ids)
    assert len(bm.free_block_ids) == len(free_set)
    assert len(bm.free_block_ids) + len(bm.used_block_ids) == total_blocks
    assert free_set.isdisjoint(used_set)
    assert (free_set | used_set) == set(range(total_blocks))
    for bid in free_set:
        assert bm.blocks[bid].ref_count == 0
    for bid in used_set:
        assert bm.blocks[bid].ref_count >= 1


def test_prefix_cache_active_reuse_and_last_block_excluded() -> None:
    bs = CacheConfig.block_size
    total_blocks = 8
    bm = BlockManager(num_blocks=total_blocks)

    prefix_2blocks = list(range(bs * 2))
    seq1 = Sequence(prefix_2blocks + [9001, 9002], temperature=0.0)
    assert bm.can_allocate(seq1) == 0
    bm.allocate(seq1)
    assert seq1.num_hashed_blocks == 0
    assert seq1.num_computed_tokens == 0
    seq1.num_computed_tokens = len(seq1.token_ids)
    bm.hash_blocks(seq1)
    assert seq1.num_hashed_blocks == 2
    assert len(bm.hash_to_block_id) == 2
    assert_block_pool_invariant(bm, total_blocks)

    # seq2 shares 2 full blocks and has a different suffix -> hits both full blocks while seq1 is active
    seq2 = Sequence(prefix_2blocks + [9999], temperature=0.0)
    assert bm.can_allocate(seq2) == 2
    bm.allocate(seq2)
    assert seq2.num_hashed_blocks == 2
    assert seq2.num_computed_tokens == bs * 2
    assert seq2.block_table[:2] == seq1.block_table[:2]
    assert seq2.block_table[2] != seq1.block_table[2]
    for bid in seq1.block_table[:2]:
        assert bm.blocks[bid].ref_count == 2
    assert_block_pool_invariant(bm, total_blocks)

    # seq3 has exact 2*bs tokens -> must leave last block uncached (hits only 1 block)
    seq3 = Sequence(prefix_2blocks, temperature=0.0)
    assert bm.can_allocate(seq3) == 1
    bm.allocate(seq3)
    assert seq3.num_hashed_blocks == 1
    assert seq3.num_computed_tokens == bs
    assert seq3.block_table[0] == seq1.block_table[0]
    assert seq3.block_table[1] != seq1.block_table[1]
    assert_block_pool_invariant(bm, total_blocks)

    for s in (seq1, seq2, seq3):
        bm.deallocate(s)
    assert_block_pool_invariant(bm, total_blocks)


def test_prefix_cache_lazy_eviction_and_revival() -> None:
    bs = CacheConfig.block_size
    total_blocks = 4
    bm = BlockManager(num_blocks=total_blocks)

    shared = list(range(bs))
    seq1 = Sequence(shared + [100], temperature=0.0)
    bm.allocate(seq1)
    seq1.num_computed_tokens = len(seq1.token_ids)
    bm.hash_blocks(seq1)
    cached_bid = seq1.block_table[0]

    # Deallocate seq1: cached_bid returns to free_block_ids but keeps its hash
    bm.deallocate(seq1)
    assert cached_bid in bm.free_block_ids
    assert bm.blocks[cached_bid].hash != -1
    assert seq1.num_hashed_blocks == 0
    assert seq1.num_computed_tokens == 0
    assert_block_pool_invariant(bm, total_blocks)

    # seq2 revives cached_bid from free_block_ids
    seq2 = Sequence(shared + [200], temperature=0.0)
    assert bm.can_allocate(seq2) == 1
    bm.allocate(seq2)
    assert seq2.block_table[0] == cached_bid
    assert seq2.num_hashed_blocks == 1
    assert seq2.num_computed_tokens == bs
    assert cached_bid in bm.used_block_ids
    assert cached_bid not in bm.free_block_ids
    assert bm.blocks[cached_bid].ref_count == 1
    assert_block_pool_invariant(bm, total_blocks)


def test_prefix_cache_revival_respects_free_block_capacity() -> None:
    bs = CacheConfig.block_size
    bm = BlockManager(num_blocks=2)

    shared = list(range(bs))
    seq1 = Sequence(shared + [1], temperature=0.0)
    bm.allocate(seq1)
    seq1.num_computed_tokens = len(seq1.token_ids)
    bm.hash_blocks(seq1)
    bm.deallocate(seq1)

    # Occupy 1 of the 2 free blocks with an unrelated 1-block sequence
    # Because of reverse free, free_block_ids is [seq1_b1, seq1_b0], so seq_other takes seq1_b1
    # leaving only 1 free block (seq1_b0, which is cached)
    seq_other = Sequence([777] * 10, temperature=0.0)
    bm.allocate(seq_other)
    assert len(bm.free_block_ids) == 1

    # seq2 needs 2 blocks: block 0 hits cached free block, block 1 needs a new block -> total 2 free blocks needed!
    seq2 = Sequence(shared + [2], temperature=0.0)
    assert bm.can_allocate(seq2) == -1


def test_prefix_cache_fifo_eviction_cleans_hash_table() -> None:
    bs = CacheConfig.block_size
    bm = BlockManager(num_blocks=2)

    shared = list(range(bs))
    seq1 = Sequence(shared + [1], temperature=0.0)
    bm.allocate(seq1)
    seq1.num_computed_tokens = len(seq1.token_ids)
    bm.hash_blocks(seq1)
    old_hash = bm.blocks[seq1.block_table[0]].hash
    assert old_hash in bm.hash_to_block_id
    bm.deallocate(seq1)

    # Allocate an unrelated 2-block sequence to evict both free blocks
    seq_evict = Sequence([555] * (bs + 5), temperature=0.0)
    bm.allocate(seq_evict)
    assert old_hash not in bm.hash_to_block_id
    bm.deallocate(seq_evict)

    # Now shared prefix must miss
    seq2 = Sequence(shared + [2], temperature=0.0)
    assert bm.can_allocate(seq2) == 0


def test_scheduler_batched_token_budget_excludes_cached_prefix() -> None:
    bs = CacheConfig.block_size
    sched = Scheduler(
        num_blocks=8,
        schedule_config=ScheduleConfig(
            max_num_seqs=4,
            max_num_batched_tokens=bs + 64,
            eos_token_id=999999,
        ),
    )

    shared = list(range(bs))
    seq1 = Sequence(shared + [10, 11], temperature=0.0, max_tokens=2)
    sched.add(seq1)
    scheduled, is_prefill = sched.schedule()
    assert is_prefill and scheduled == [seq1]
    sched.postprocess(scheduled, [12])

    # Now block 0 (256 tokens) is cached. Add two sequences each with 256 + 20 tokens.
    # Under mixed scheduling, seq1 (decode: 1 tok) + seq2 (20 new toks) + seq3 (20 new toks) = 41 <= 320!
    seq2 = Sequence(shared + [20] * 20, temperature=0.0, max_tokens=2)
    seq3 = Sequence(shared + [30] * 20, temperature=0.0, max_tokens=2)
    sched.add(seq2)
    sched.add(seq3)
    scheduled, is_prefill = sched.schedule()
    assert is_prefill
    assert scheduled == [seq1, seq2, seq3]
    assert seq2.num_computed_tokens == bs
    assert seq3.num_computed_tokens == bs


@pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL_PATH.exists(),
    reason="Requires CUDA GPU and local Qwen3-0.6B weights",
)
def test_prefix_caching_qwen3_gpu_exact_match() -> None:
    bs = CacheConfig.block_size
    engine = LLMEngine(
        "qwen3",
        MODEL_PATH,
        schedule_config=ScheduleConfig(max_num_seqs=8, max_num_batched_tokens=2048),
    )
    try:
        # Build a shared system prompt longer than 2 full KV blocks (> 512 tokens)
        shared_text = (
            "You are a precise geography and mathematics assistant. "
            "Always answer concisely and directly without extra preamble. "
        ) * 30
        shared_ids = engine.tokenizer.encode(shared_text)
        assert len(shared_ids) > bs * 2

        q1 = engine.tokenizer.encode("\nQuestion: What is the capital of France?\nAnswer:")
        q2 = engine.tokenizer.encode("\nQuestion: Continue the sequence 1, 2, 3, 4, 5, 6,\nAnswer:")
        prompt1 = shared_ids + q1
        prompt2 = shared_ids + q2

        sp = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)

        # 1. Warm the cache with prompt1
        _ = engine.generate([prompt1], [sp])
        assert len(engine.scheduler.block_manager.hash_to_block_id) >= 2

        # 2. Run prompt2 WITH prefix cache and verify cache hit during scheduling
        engine.add_request(prompt2, sp)
        seqs, is_prefill = engine.scheduler.schedule()
        assert is_prefill
        assert seqs[0].num_computed_tokens == (len(shared_ids) // bs) * bs
        token_ids = engine.model_runner.run(seqs, is_prefill)
        engine.scheduler.postprocess(seqs, token_ids)
        while not engine.scheduler.is_finished:
            engine.step()
        cached_output = seqs[0].token_ids[seqs[0].num_prompt_tokens:]

        # 3. Clear prefix cache metadata and run prompt2 WITHOUT prefix cache
        bm = engine.scheduler.block_manager
        bm.hash_to_block_id.clear()
        for blk in bm.blocks:
            blk.hash = -1
            blk.token_ids = []

        engine.add_request(prompt2, sp)
        seqs_cold, is_prefill_cold = engine.scheduler.schedule()
        assert is_prefill_cold
        assert seqs_cold[0].num_computed_tokens == 0
        token_ids_cold = engine.model_runner.run(seqs_cold, is_prefill_cold)
        engine.scheduler.postprocess(seqs_cold, token_ids_cold)
        while not engine.scheduler.is_finished:
            engine.step()
        uncached_output = seqs_cold[0].token_ids[seqs_cold[0].num_prompt_tokens:]

        assert cached_output == uncached_output
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()

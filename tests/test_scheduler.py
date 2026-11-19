from mini_infer.config.engine_config import CacheConfig, ScheduleConfig
from mini_infer.engine.scheduler import Scheduler
from mini_infer.engine.sequence import Sequence, SequenceStatus
from tests.test_block_manager import assert_block_pool_invariant


def run_dummy_engine(
    scheduler: Scheduler,
    total_blocks: int,
    next_token_fn=None,
) -> list[tuple[list[int], bool]]:
    """Drive scheduler to completion with a dummy token generator, asserting invariants each step."""
    if next_token_fn is None:
        next_token_fn = lambda seq: 1000 + seq.seq_id  # noqa: E731

    trace: list[tuple[list[int], bool]] = []
    max_steps = 200
    steps = 0
    while not scheduler.is_finished:
        steps += 1
        assert steps <= max_steps, "Scheduler loop did not terminate (possible deadlock)"
        seqs, is_prefill = scheduler.schedule()
        assert seqs, "schedule() returned empty batch while scheduler is not finished"
        assert_block_pool_invariant(scheduler.block_manager, total_blocks)
        trace.append(([s.seq_id for s in seqs], is_prefill))

        tokens = [next_token_fn(s) for s in seqs]
        scheduler.postprocess(seqs, tokens)
        assert_block_pool_invariant(scheduler.block_manager, total_blocks)

    assert len(scheduler.block_manager.used_block_ids) == 0
    assert len(scheduler.block_manager.free_block_ids) == total_blocks
    return trace


def test_scheduler_single_sequence_lifecycle() -> None:
    total_blocks = 4
    scheduler = Scheduler(num_blocks=total_blocks)
    seq = Sequence([1, 2, 3, 4], max_tokens=3, temperature=0.0)

    assert scheduler.is_finished is True
    scheduler.add(seq)
    assert scheduler.is_finished is False

    # Step 1: Prefill
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert seqs == [seq]
    assert seq.status == SequenceStatus.RUNNING
    assert len(seq.block_table) == 1
    assert_block_pool_invariant(scheduler.block_manager, total_blocks)

    scheduler.postprocess(seqs, [10])
    assert seq.token_ids == [1, 2, 3, 4, 10]
    assert seq.num_completion_tokens == 1
    assert seq.status == SequenceStatus.RUNNING

    # Step 2: Decode
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill is False
    assert seqs == [seq]
    scheduler.postprocess(seqs, [11])
    assert seq.num_completion_tokens == 2
    assert seq.status == SequenceStatus.RUNNING

    # Step 3: Decode reaches max_tokens=3 -> FINISHED & freed
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill is False
    scheduler.postprocess(seqs, [12])

    assert seq.status == SequenceStatus.FINISHED
    assert seq.token_ids == [1, 2, 3, 4, 10, 11, 12]
    assert seq.block_table == []
    assert scheduler.is_finished is True
    assert_block_pool_invariant(scheduler.block_manager, total_blocks)


def test_scheduler_max_num_seqs_limit() -> None:
    total_blocks = 8
    cfg = ScheduleConfig(max_num_seqs=2, max_num_batched_tokens=1024)
    scheduler = Scheduler(num_blocks=total_blocks, schedule_config=cfg)

    s0 = Sequence([1, 2, 3], max_tokens=2, temperature=0.0)
    s1 = Sequence([4, 5, 6], max_tokens=2, temperature=0.0)
    s2 = Sequence([7, 8, 9], max_tokens=2, temperature=0.0)
    for s in (s0, s1, s2):
        scheduler.add(s)

    # Step 1: Prefill capped at max_num_seqs=2
    seqs1, is_prefill1 = scheduler.schedule()
    assert is_prefill1 is True
    assert seqs1 == [s0, s1]
    scheduler.postprocess(seqs1, [100, 101])

    # Step 2: Running takes priority; decode capped at max_num_seqs=2 so s2 stays waiting
    seqs2, is_prefill2 = scheduler.schedule()
    assert is_prefill2 is False
    assert seqs2 == [s0, s1]
    assert list(scheduler.waiting) == [s2]
    scheduler.postprocess(seqs2, [200, 201])

    # s0 and s1 reached max_tokens=2 and finished; now s2 can enter prefill
    assert s0.status == SequenceStatus.FINISHED
    assert s1.status == SequenceStatus.FINISHED
    assert list(scheduler.running) == []

    # Step 3: s2 prefills
    seqs3, is_prefill3 = scheduler.schedule()
    assert is_prefill3 is True
    assert seqs3 == [s2]
    scheduler.postprocess(seqs3, [102])

    # Step 4: s2 decodes and finishes
    seqs4, is_prefill4 = scheduler.schedule()
    assert is_prefill4 is False
    assert seqs4 == [s2]
    scheduler.postprocess(seqs4, [202])
    assert scheduler.is_finished is True
    assert_block_pool_invariant(scheduler.block_manager, total_blocks)


def test_scheduler_max_num_batched_tokens_limit() -> None:
    total_blocks = 8
    cfg = ScheduleConfig(max_num_seqs=10, max_num_batched_tokens=20)
    scheduler = Scheduler(num_blocks=total_blocks, schedule_config=cfg)

    s0 = Sequence(list(range(10)), max_tokens=1, temperature=0.0)
    s1 = Sequence(list(range(10)), max_tokens=1, temperature=0.0)
    s2 = Sequence(list(range(5)), max_tokens=1, temperature=0.0)
    for s in (s0, s1, s2):
        scheduler.add(s)

    # Step 1: 10 + 10 == 20 fits budget; adding s2 (5) would exceed 20
    seqs1, is_prefill1 = scheduler.schedule()
    assert is_prefill1 is True
    assert seqs1 == [s0, s1]
    scheduler.postprocess(seqs1, [1, 1])

    assert s0.status == SequenceStatus.FINISHED
    assert s1.status == SequenceStatus.FINISHED

    # Step 2: s2 prefills and finishes
    seqs2, is_prefill2 = scheduler.schedule()
    assert is_prefill2 is True
    assert seqs2 == [s2]
    scheduler.postprocess(seqs2, [1])
    assert scheduler.is_finished is True
    assert_block_pool_invariant(scheduler.block_manager, total_blocks)


def test_scheduler_eos_and_ignore_eos() -> None:
    eos = 999
    total_blocks = 4
    cfg = ScheduleConfig(eos_token_id=eos)
    scheduler = Scheduler(num_blocks=total_blocks, schedule_config=cfg)

    s_stop_on_eos = Sequence([1, 2, 3], max_tokens=5, ignore_eos=False, temperature=0.0)
    s_ignore_eos = Sequence([4, 5, 6], max_tokens=3, ignore_eos=True, temperature=0.0)
    scheduler.add(s_stop_on_eos)
    scheduler.add(s_ignore_eos)

    # Both receive EOS on step 1
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill is True
    scheduler.postprocess(seqs, [eos, eos])

    assert s_stop_on_eos.status == SequenceStatus.FINISHED
    assert s_stop_on_eos.num_completion_tokens == 1
    assert s_stop_on_eos.block_table == []

    assert s_ignore_eos.status == SequenceStatus.RUNNING
    assert s_ignore_eos.num_completion_tokens == 1

    # s_ignore_eos keeps running until max_tokens=3
    run_dummy_engine(scheduler, total_blocks, next_token_fn=lambda _: eos)
    assert s_ignore_eos.status == SequenceStatus.FINISHED
    assert s_ignore_eos.num_completion_tokens == 3


def test_scheduler_continuous_batching_dynamic_join_and_free() -> None:
    bs = CacheConfig.block_size
    total_blocks = 4
    scheduler = Scheduler(num_blocks=total_blocks)

    s_short = Sequence(list(range(10)), max_tokens=2, temperature=0.0)          # 1 block
    s_long = Sequence(list(range(bs + 4)), max_tokens=4, temperature=0.0)       # 2 blocks
    s_wait = Sequence([10000 + i for i in range(bs + 4)], max_tokens=2, temperature=0.0)  # 2 blocks (unique prefix)

    for s in (s_short, s_long, s_wait):
        scheduler.add(s)

    # Step 1: s_short (1) + s_long (2) = 3 blocks; s_wait needs 2 > 1 free, waits
    seqs1, is_prefill1 = scheduler.schedule()
    assert is_prefill1 is True
    assert seqs1 == [s_short, s_long]
    assert list(scheduler.waiting) == [s_wait]
    scheduler.postprocess(seqs1, [101, 201])

    # Step 2: decode s_short & s_long; s_short finishes and frees 1 block -> 2 free
    seqs2, is_prefill2 = scheduler.schedule()
    assert is_prefill2 is False
    assert seqs2 == [s_short, s_long]
    scheduler.postprocess(seqs2, [102, 202])
    assert s_short.status == SequenceStatus.FINISHED
    assert len(scheduler.block_manager.free_block_ids) == 2

    # Step 3: Mixed batch! s_long decodes while s_wait prefills in the same step
    seqs3, is_prefill3 = scheduler.schedule()
    assert is_prefill3 is True
    assert seqs3 == [s_long, s_wait]
    scheduler.postprocess(seqs3, [203, 301])

    # Finish remaining decode steps for s_long and s_wait
    run_dummy_engine(scheduler, total_blocks)
    assert s_long.num_completion_tokens == 4
    assert s_wait.num_completion_tokens == 2


def test_scheduler_preemption_evicts_tail_and_recomputes() -> None:
    bs = CacheConfig.block_size
    total_blocks = 2  # Tight pool: fits both prefills (1 block each), forces preempt on block cross
    scheduler = Scheduler(num_blocks=total_blocks)

    s0 = Sequence(list(range(bs)), max_tokens=3, temperature=0.0)
    s1 = Sequence(list(range(bs)), max_tokens=3, temperature=0.0)
    scheduler.add(s0)
    scheduler.add(s1)

    # Step 1: Both prefill (2 blocks used, 0 free)
    seqs1, is_prefill1 = scheduler.schedule()
    assert is_prefill1 is True
    assert seqs1 == [s0, s1]
    scheduler.postprocess(seqs1, [500, 600])

    # Step 2: Decode s0 needs 2nd block -> preempts tail (s1) to free 1 block
    seqs2, is_prefill2 = scheduler.schedule()
    assert is_prefill2 is False
    assert seqs2 == [s0]
    assert s1.status == SequenceStatus.WAITING
    assert s1.block_table == []
    assert list(scheduler.waiting) == [s1]
    assert len(s1.token_ids) == bs + 1
    assert s1.token_ids[-1] == 600
    assert_block_pool_invariant(scheduler.block_manager, total_blocks)

    scheduler.postprocess(seqs2, [501])

    trace = run_dummy_engine(scheduler, total_blocks)

    assert s0.status == SequenceStatus.FINISHED
    assert s1.status == SequenceStatus.FINISHED
    assert s0.num_completion_tokens == 3
    assert s1.num_completion_tokens == 3
    assert any(ids == [s1.seq_id] and is_p for ids, is_p in trace)


def test_scheduler_preempt_self_when_last_running_seq() -> None:
    bs = CacheConfig.block_size
    total_blocks = 3
    scheduler = Scheduler(num_blocks=total_blocks)

    s0 = Sequence(list(range(bs)), max_tokens=3, temperature=0.0)
    s1 = Sequence(list(range(bs)), max_tokens=3, temperature=0.0)
    scheduler.add(s0)
    scheduler.add(s1)

    # Step 1: Both prefill (2 used, 1 free)
    seqs1, _ = scheduler.schedule()
    scheduler.postprocess(seqs1, [700, 800])

    # Step 2: s0 takes the last free block (3 used, 0 free);
    # s1 has no other running seq behind it, so s1 preempts itself!
    seqs2, is_prefill2 = scheduler.schedule()
    assert is_prefill2 is False
    assert seqs2 == [s0]
    assert s1.status == SequenceStatus.WAITING
    assert list(scheduler.waiting) == [s1]
    assert list(scheduler.running) == [s0]
    assert_block_pool_invariant(scheduler.block_manager, total_blocks)

    scheduler.postprocess(seqs2, [701])
    run_dummy_engine(scheduler, total_blocks)

    assert s0.num_completion_tokens == 3
    assert s1.num_completion_tokens == 3

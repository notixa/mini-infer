import gc
from pathlib import Path

import pytest
import torch

from mini_infer.config.engine_config import CacheConfig, ScheduleConfig
from mini_infer.engine.llm_engine import LLMEngine, SamplingParams
from mini_infer.engine.scheduler import Scheduler
from mini_infer.engine.sequence import Sequence, SequenceStatus

MODEL_PATH = Path("/home/notixa/huggingface/Qwen3-0.6B")


def test_chunked_prefill_single_long_sequence_and_incremental_hashing() -> None:
    bs = CacheConfig.block_size  # 256
    sched = Scheduler(
        num_blocks=8,
        schedule_config=ScheduleConfig(
            max_num_seqs=4,
            max_num_batched_tokens=300,  # non-block-aligned!
            eos_token_id=999999,
        ),
    )

    # 700 tokens -> 3 chunks: 300, 300, 100
    seq = Sequence(list(range(700)), temperature=0.0, max_tokens=2)
    sched.add(seq)

    # Chunk 1: [0, 300) -> block 0 (0..255) completed and hashed; no token appended
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert seqs == [seq]
    assert seq.num_scheduled_tokens == 300
    sched.postprocess(seqs, [1001])
    assert seq.num_computed_tokens == 300
    assert seq.num_hashed_blocks == 1
    assert len(seq.token_ids) == 700  # intermediate chunk must NOT append token
    assert seq.num_completion_tokens == 0

    # Chunk 2: [300, 600) -> block 1 (256..511) completed and hashed; still no token appended
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert seqs == [seq]
    assert seq.num_scheduled_tokens == 300
    sched.postprocess(seqs, [1002])
    assert seq.num_computed_tokens == 600
    assert seq.num_hashed_blocks == 2
    assert len(seq.token_ids) == 700
    assert seq.num_completion_tokens == 0

    # Chunk 3 (final): [600, 700) -> prompt complete, appends first generated token
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert seqs == [seq]
    assert seq.num_scheduled_tokens == 100
    sched.postprocess(seqs, [1003])
    assert seq.num_computed_tokens == 700
    assert seq.num_hashed_blocks == 2
    assert len(seq.token_ids) == 701
    assert seq.token_ids[-1] == 1003
    assert seq.num_completion_tokens == 1

    # Step 4: Pure Decode (1 token) -> finishes (max_tokens=2)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is False
    assert seqs == [seq]
    assert seq.num_scheduled_tokens == 1
    sched.postprocess(seqs, [1004])
    assert seq.status == SequenceStatus.FINISHED
    assert seq.token_ids[-2:] == [1003, 1004]
    assert sched.is_finished


def test_mixed_scheduling_decode_piggybacks_with_chunked_prefill() -> None:
    sched = Scheduler(
        num_blocks=8,
        schedule_config=ScheduleConfig(
            max_num_seqs=4,
            max_num_batched_tokens=200,
            eos_token_id=999999,
        ),
    )

    # Start a short sequence and finish its prefill so it enters Decode
    seq_short = Sequence(list(range(10)), temperature=0.0, max_tokens=4)
    sched.add(seq_short)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True and seqs == [seq_short]
    sched.postprocess(seqs, [501])
    assert seq_short.num_completion_tokens == 1

    # Now a long 500-token prompt arrives while seq_short is decoding
    seq_long = Sequence([1000 + i for i in range(500)], temperature=0.0, max_tokens=2)
    sched.add(seq_long)

    # Mixed Step 1: seq_short (decode: 1 tok) + seq_long (chunk 1: 199 toks) = 200
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert seqs == [seq_short, seq_long]
    assert seq_short.num_scheduled_tokens == 1
    assert seq_long.num_scheduled_tokens == 199
    sched.postprocess(seqs, [502, 9999])
    assert seq_short.num_completion_tokens == 2
    assert seq_short.token_ids[-1] == 502
    assert seq_long.num_computed_tokens == 199
    assert seq_long.num_completion_tokens == 0

    # Mixed Step 2: seq_short (decode: 1 tok) + seq_long (chunk 2: 199 toks) = 200
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert seqs == [seq_short, seq_long]
    assert seq_short.num_scheduled_tokens == 1
    assert seq_long.num_scheduled_tokens == 199
    sched.postprocess(seqs, [503, 9999])
    assert seq_short.num_completion_tokens == 3
    assert seq_long.num_computed_tokens == 398
    assert seq_long.num_completion_tokens == 0

    # Mixed Step 3: seq_short (decode: 1 tok, finishes!) + seq_long (final chunk: 102 toks)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert seqs == [seq_short, seq_long]
    assert seq_short.num_scheduled_tokens == 1
    assert seq_long.num_scheduled_tokens == 102
    sched.postprocess(seqs, [504, 601])
    assert seq_short.status == SequenceStatus.FINISHED
    assert seq_long.num_computed_tokens == 500
    assert seq_long.num_completion_tokens == 1
    assert seq_long.token_ids[-1] == 601

    # Pure Decode Step 4: seq_long finishes
    seqs, is_prefill = sched.schedule()
    assert is_prefill is False
    assert seqs == [seq_long]
    sched.postprocess(seqs, [602])
    assert seq_long.status == SequenceStatus.FINISHED
    assert sched.is_finished


def test_chunked_prefill_block_boundary_no_block_leak() -> None:
    bs = CacheConfig.block_size  # 256
    total_blocks = 4
    sched = Scheduler(
        num_blocks=total_blocks,
        schedule_config=ScheduleConfig(
            max_num_seqs=4,
            max_num_batched_tokens=100,
            eos_token_id=999999,
        ),
    )

    # Prompt length 257 -> len % block_size == 1, needs exactly 2 blocks
    seq = Sequence(list(range(bs + 1)), temperature=0.0, max_tokens=2)
    sched.add(seq)

    for step_idx, expected_chunk in enumerate((100, 100, 57)):
        seqs, is_prefill = sched.schedule()
        assert is_prefill is True
        assert seqs == [seq]
        assert seq.num_scheduled_tokens == expected_chunk
        # Must stay at exactly 2 blocks across all chunks (no extra block allocated!)
        assert len(seq.block_table) == 2
        sched.postprocess(seqs, [700 + step_idx])

    assert len(seq.token_ids) == bs + 2
    assert len(seq.block_table) == 2


def test_preempted_sequence_chunked_recomputes_all_tokens() -> None:
    bs = CacheConfig.block_size
    sched = Scheduler(
        num_blocks=3,
        schedule_config=ScheduleConfig(
            max_num_seqs=4,
            max_num_batched_tokens=200,
            eos_token_id=999999,
        ),
    )

    # Simulate a sequence that had 250 prompt tokens + 5 generated tokens before being preempted
    seq = Sequence(list(range(250)), temperature=0.0, max_tokens=10)
    sched.add(seq)
    # Chunk 1 (200) + Chunk 2 (50)
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [1])
    seqs, _ = sched.schedule()
    sched.postprocess(seqs, [10])
    for tok in (11, 12, 13, 14):
        seqs, is_prefill = sched.schedule()
        assert is_prefill is False
        sched.postprocess(seqs, [tok])
    assert len(seq.token_ids) == 255
    assert seq.num_completion_tokens == 5

    # Preempt seq back to waiting (clears blocks and resets num_computed_tokens=0)
    sched.running.remove(seq)
    sched.preempt(seq)
    assert seq.num_computed_tokens == 0
    assert seq.block_table == []

    # Re-admit with budget=200: chunk 1 recomputes 200 tokens (0..199)
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True and seq.num_scheduled_tokens == 200
    sched.postprocess(seqs, [999])
    assert len(seq.token_ids) == 255  # must NOT append yet
    assert seq.num_computed_tokens == 200

    # Chunk 2 recomputes remaining 55 tokens (200..254) -> NOW appends token 15
    seqs, is_prefill = sched.schedule()
    assert is_prefill is True and seq.num_scheduled_tokens == 55
    sched.postprocess(seqs, [15])
    assert seq.num_computed_tokens == 255
    assert len(seq.token_ids) == 256
    assert seq.token_ids[-1] == 15


@pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL_PATH.exists(),
    reason="Requires CUDA GPU and local Qwen3-0.6B weights",
)
def test_mixed_chunked_prefill_qwen3_gpu_exact_match() -> None:
    # 1. Ground truth: full unchunked, unmixed runs with max_num_batched_tokens=2048
    engine_ref = LLMEngine(
        "qwen3",
        MODEL_PATH,
        schedule_config=ScheduleConfig(max_num_seqs=4, max_num_batched_tokens=2048),
    )
    try:
        long_text = (
            "You are a precise geography and mathematics assistant. "
            "Always answer concisely and directly without extra preamble. "
        ) * 35 + "\nQuestion: What is the capital of France?\nAnswer:"
        short_text = "1, 2, 3, 4, 5, 6,"

        long_ids = engine_ref.tokenizer.encode(long_text)
        assert len(long_ids) > 650

        sp_long = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        sp_short = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)

        expected_long = engine_ref.generate([long_ids], [sp_long])[0]["token_ids"]
        expected_short = engine_ref.generate([short_text], [sp_short])[0]["token_ids"]
    finally:
        del engine_ref
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Mixed + Chunked Prefill run with non-block-aligned max_num_batched_tokens=300
    engine = LLMEngine(
        "qwen3",
        MODEL_PATH,
        schedule_config=ScheduleConfig(max_num_seqs=4, max_num_batched_tokens=300),
    )
    try:
        short_rid = engine.add_request(short_text, sp_short)
        # Advance short request into active Decode (1 prefill + 1 decode step)
        for _ in range(2):
            engine.step()

        # Now inject long prompt while short request is actively decoding!
        long_rid = engine.add_request(long_ids, sp_long)

        observed_mixed_step = False
        outputs: dict[int, list[int]] = {}
        while not engine.scheduler.is_finished:
            seqs, is_prefill = engine.scheduler.schedule()
            if (
                is_prefill
                and len(seqs) == 2
                and seqs[0].seq_id == short_rid
                and seqs[1].seq_id == long_rid
                and seqs[0].num_scheduled_tokens == 1
                and seqs[1].num_scheduled_tokens == 299
            ):
                observed_mixed_step = True

            token_ids = engine.model_runner.run(seqs, is_prefill)
            engine.scheduler.postprocess(seqs, token_ids)
            for seq in seqs:
                if seq.status == SequenceStatus.FINISHED:
                    outputs[seq.seq_id] = seq.token_ids[seq.num_prompt_tokens:]

        assert observed_mixed_step, "Expected a true mixed Decode(1) + ChunkedPrefill(299) step"
        assert outputs[short_rid] == expected_short
        assert outputs[long_rid] == expected_long
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()

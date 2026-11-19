import argparse
import time
from pathlib import Path

import torch

from mini_infer.config.engine_config import ScheduleConfig
from mini_infer.engine.llm_engine import LLMEngine, SamplingParams


def clear_prefix_cache(engine: LLMEngine) -> None:
    bm = engine.scheduler.block_manager
    bm.hash_to_block_id.clear()
    for blk in bm.blocks:
        blk.hash = -1
        blk.token_ids = []


def run_pass(
    engine: LLMEngine,
    prompts: list[list[int]],
    output_len: int,
    label: str,
) -> dict[str, float]:
    sp = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    for seq_prompt in prompts:
        engine.add_request(seq_prompt, sp)

    total_prompt_tokens = sum(len(p) for p in prompts)
    computed_prefill_tokens = 0
    prefill_time = 0.0
    decode_tokens = 0
    decode_time = 0.0

    torch.cuda.synchronize()
    total_start = time.perf_counter()

    while not engine.scheduler.is_finished:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        seqs, is_prefill = engine.scheduler.schedule()
        if is_prefill:
            step_prefill_tokens = sum(
                s.num_scheduled_tokens
                for s in seqs
                if s.num_computed_tokens < s.num_prompt_tokens
            )
            step_decode_tokens = sum(
                1 for s in seqs if s.num_computed_tokens >= s.num_prompt_tokens
            )
        else:
            step_prefill_tokens = 0
            step_decode_tokens = len(seqs)

        token_ids = engine.model_runner.run(seqs, is_prefill)
        engine.scheduler.postprocess(seqs, token_ids)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if is_prefill:
            computed_prefill_tokens += step_prefill_tokens
            decode_tokens += step_decode_tokens
            prefill_time += dt
        else:
            decode_tokens += step_decode_tokens
            decode_time += dt

    total_time = time.perf_counter() - total_start

    effective_prefill_tps = total_prompt_tokens / prefill_time if prefill_time > 0 else 0.0
    decode_tps = decode_tokens / decode_time if decode_time > 0 else 0.0
    total_tps = (total_prompt_tokens + decode_tokens) / total_time if total_time > 0 else 0.0
    cache_hit_ratio = (
        1.0 - computed_prefill_tokens / total_prompt_tokens
        if total_prompt_tokens > 0
        else 0.0
    )

    print(f"--- {label} ---")
    print(
        f"Prefill : {total_prompt_tokens:6d} prompt tokens "
        f"(computed {computed_prefill_tokens:6d}, cache hit {cache_hit_ratio * 100:5.1f}%) "
        f"in {prefill_time * 1000:8.2f} ms -> {effective_prefill_tps:9.2f} eff tok/s"
    )
    print(
        f"Decode  : {decode_tokens:6d} tokens "
        f"in {decode_time * 1000:8.2f} ms -> {decode_tps:9.2f} tok/s"
    )
    print(
        f"Total   : {total_prompt_tokens + decode_tokens:6d} tokens "
        f"in {total_time * 1000:8.2f} ms -> {total_tps:9.2f} tok/s"
    )

    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "effective_prefill_tps": effective_prefill_tps,
        "cache_hit_ratio": cache_hit_ratio,
    }


def run_benchmark(
    model_path: Path,
    num_seqs: int,
    input_len: int,
    output_len: int,
    max_num_batched_tokens: int,
    shared_prefix_len: int = 0,
    compare_prefix_cache: bool = False,
) -> None:
    schedule_config = ScheduleConfig(
        max_num_seqs=max(num_seqs, 64),
        max_num_batched_tokens=max(max_num_batched_tokens, input_len),
    )
    engine = LLMEngine("qwen3", model_path, schedule_config=schedule_config)

    # Warmup Triton JIT kernels before timing
    engine.generate(
        ["Warmup prompt for Triton kernels."],
        [SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)],
    )
    clear_prefix_cache(engine)

    print(f"=== mini-infer Benchmark ({model_path.name}) ===")
    print(f"Requests: {num_seqs} | Input len: {input_len} | Output len: {output_len}")
    print(
        f"KV Blocks: {engine.model_runner.num_blocks} "
        f"(block_size={engine.model_runner.cache_config.block_size})"
    )

    base_tokens = [100 + (i % 1000) for i in range(input_len)]

    if compare_prefix_cache:
        prefix_len = shared_prefix_len if shared_prefix_len > 0 else max(0, input_len - 64)
        # 1. Baseline: unique prefix per request (0% prefix cache hit)
        unique_prompts = [
            [2000 + idx] + base_tokens[1:] for idx in range(num_seqs)
        ]
        clear_prefix_cache(engine)
        res_cold = run_pass(engine, unique_prompts, output_len, "No Shared Prefix (Baseline)")

        # 2. Shared prefix: warm 1st request to populate prefix cache, then run all requests
        shared_prompts = [
            base_tokens[:prefix_len] + [3000 + idx] + base_tokens[prefix_len + 1:]
            for idx in range(num_seqs)
        ]
        clear_prefix_cache(engine)
        # Warm the shared system prompt with a single short request
        engine.generate(
            [shared_prompts[0]],
            [SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)],
        )
        res_warm = run_pass(
            engine,
            shared_prompts,
            output_len,
            f"Shared Prefix ({prefix_len} shared + {input_len - prefix_len} unique tokens)",
        )
        speedup = (
            res_cold["prefill_time"] / res_warm["prefill_time"]
            if res_warm["prefill_time"] > 0
            else 0.0
        )
        print(f"Prefill Speedup with Prefix Caching: {speedup:.2f}x")
    else:
        if shared_prefix_len > 0:
            prompts = [
                base_tokens[:shared_prefix_len]
                + [3000 + idx]
                + base_tokens[shared_prefix_len + 1:]
                for idx in range(num_seqs)
            ]
            label = f"Shared Prefix ({shared_prefix_len} tokens)"
        else:
            prompts = [[2000 + idx] + base_tokens[1:] for idx in range(num_seqs)]
            label = "Unique Prompts"
        run_pass(engine, prompts, output_len, label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mini-infer prefill/decode throughput")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/notixa/huggingface/Qwen3-0.6B"),
    )
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument("--compare-prefix-cache", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        num_seqs=args.num_seqs,
        input_len=args.input_len,
        output_len=args.output_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        shared_prefix_len=args.shared_prefix_len,
        compare_prefix_cache=args.compare_prefix_cache,
    )


if __name__ == "__main__":
    main()

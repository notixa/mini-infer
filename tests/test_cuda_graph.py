import gc
from pathlib import Path

import pytest
import torch

from mini_infer.config.engine_config import ScheduleConfig
from mini_infer.engine.llm_engine import LLMEngine, SamplingParams

MODEL_PATH = Path("/home/notixa/huggingface/Qwen3-0.6B")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL_PATH.exists(),
    reason="Requires CUDA GPU and local Qwen3-0.6B weights",
)
def test_cuda_graph_decode_matches_eager_exact_and_padded_batch() -> None:
    """Verify CUDA Graph decode outputs match eager mode bit-for-bit,
    including non-exact batch size bs=3 padded into graph_bs=4."""
    prompts = [
        "1, 2, 3, 4, 5, 6,",
        "The capital of France is",
        "2 + 2 = 4, 3 + 3 = 6, 4 + 4 =",
    ]
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
        SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True),
        SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True),
    ]
    cfg = ScheduleConfig(max_num_seqs=16, max_num_batched_tokens=2048)

    # 1. Eager reference
    engine_eager = LLMEngine("qwen3", MODEL_PATH, schedule_config=cfg, enforce_eager=True)
    try:
        assert engine_eager.model_runner.graphs == {}
        eager_outputs = engine_eager.generate(prompts, sampling_params)
    finally:
        del engine_eager
        gc.collect()
        torch.cuda.empty_cache()

    # 2. CUDA Graph run (captures graph_bs=[1, 2, 4, 8, 16]; bs=3 pads into graph_bs=4)
    engine_graph = LLMEngine("qwen3", MODEL_PATH, schedule_config=cfg, enforce_eager=False)
    try:
        assert engine_graph.model_runner.graph_bs == [1, 2, 4, 8, 16]
        assert set(engine_graph.model_runner.graphs.keys()) == {1, 2, 4, 8, 16}
        graph_outputs = engine_graph.generate(prompts, sampling_params)
    finally:
        del engine_graph
        gc.collect()
        torch.cuda.empty_cache()

    for eager_out, graph_out in zip(eager_outputs, graph_outputs):
        assert graph_out["token_ids"] == eager_out["token_ids"]
        assert graph_out["text"] == eager_out["text"]

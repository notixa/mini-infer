from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_infer.engine.llm_engine import LLMEngine, SamplingParams

MODEL_DIR = Path("/home/notixa/huggingface/Qwen3-0.6B")


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Local Qwen3-0.6B checkpoint not found")
def test_llm_engine_greedy_matches_huggingface() -> None:
    prompts = [
        "1, 2, 3, 4, 5, 6,",
        "The capital of France is",
    ]
    max_new_tokens = 12

    # 1. Run HuggingFace reference first, then free VRAM before LLMEngine allocates KV cache
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16
    ).to("cuda")
    hf_model.eval()

    hf_token_ids: list[list[int]] = []
    with torch.inference_mode():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            out = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            hf_token_ids.append(out[0, input_ids.shape[1]:].tolist())

    del hf_model
    torch.cuda.empty_cache()

    # 2. Run mini_infer LLMEngine with greedy sampling
    prev_dtype = torch.get_default_dtype()
    try:
        engine = LLMEngine("qwen3", MODEL_DIR)
        outputs = engine.generate(
            prompts,
            [
                SamplingParams(temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True)
                for _ in prompts
            ],
        )

        assert len(outputs) == len(prompts)
        # Deterministic counting sequence matches HF 100% across all tokens
        assert outputs[0]["token_ids"] == hf_token_ids[0]
        # Open-ended prompt matches HF on top-confidence prefix (" Paris. The capital of")
        assert outputs[1]["token_ids"][:5] == hf_token_ids[1][:5]
        for out in outputs:
            assert isinstance(out["text"], str) and len(out["text"]) > 0

        assert engine.scheduler.is_finished is True
        assert len(engine.scheduler.block_manager.used_block_ids) == 0
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.cuda.empty_cache()


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Local Qwen3-0.6B checkpoint not found")
def test_llm_engine_continuous_batching_order_and_mixed_params() -> None:
    prompts = [
        "Explain PagedAttention in one sentence:",
        "2 * 3 =",
        "Once upon a time",
    ]
    params = [
        SamplingParams(temperature=0.0, max_tokens=16),
        SamplingParams(temperature=0.0, max_tokens=4),
        SamplingParams(temperature=0.6, max_tokens=10),
    ]

    prev_dtype = torch.get_default_dtype()
    try:
        engine = LLMEngine("qwen3", MODEL_DIR)
        outputs = engine.generate(prompts, params)

        assert len(outputs) == 3
        assert 1 <= len(outputs[0]["token_ids"]) <= 16
        assert 1 <= len(outputs[1]["token_ids"]) <= 4
        assert 1 <= len(outputs[2]["token_ids"]) <= 10
        for out in outputs:
            assert isinstance(out["text"], str) and len(out["text"]) > 0

        assert engine.scheduler.is_finished is True
        assert len(engine.scheduler.block_manager.used_block_ids) == 0
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.cuda.empty_cache()

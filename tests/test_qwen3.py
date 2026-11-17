import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_infer.config.engine_config import CacheConfig
from mini_infer.config.model_config import Qwen3Config
from mini_infer.engine.block_manager import BlockManager
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.sequence import Sequence
from mini_infer.layers.attention import Attention
from mini_infer.model.qwen3 import Qwen3ForCausalLM
from mini_infer.utils.context import reset_context
from mini_infer.utils.loader import load_weight

MODEL_DIR = Path("/home/notixa/huggingface/Qwen3-0.6B")


def _build_and_load_mini_qwen3() -> tuple[Qwen3ForCausalLM, Qwen3Config]:
    with open(MODEL_DIR / "config.json", encoding="utf-8") as f:
        cfg = Qwen3Config.from_dict(json.load(f))

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(getattr(torch, cfg.torch_dtype))
    try:
        model = Qwen3ForCausalLM(cfg).to("cuda")
        load_weight(model, MODEL_DIR)
        model.eval()
    finally:
        torch.set_default_dtype(prev_dtype)
    return model, cfg


def _bind_kv_cache(model: Qwen3ForCausalLM, cfg: Qwen3Config, num_blocks: int) -> torch.Tensor:
    bs = CacheConfig.block_size
    kv_cache = torch.empty(
        (2, cfg.num_hidden_layers, num_blocks, bs, cfg.num_key_value_heads, cfg.head_dim),
        dtype=getattr(torch, cfg.torch_dtype),
        device="cuda",
    )
    layer_id = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.k_cache = kv_cache[0, layer_id]
            module.v_cache = kv_cache[1, layer_id]
            layer_id += 1
    assert layer_id == cfg.num_hidden_layers
    return kv_cache


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Local Qwen3-0.6B checkpoint not found")
def test_load_qwen3_0_6b_weights_and_tied_embeddings() -> None:
    model, cfg = _build_and_load_mini_qwen3()

    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert len(model.model.layers) == cfg.num_hidden_layers
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN found in {name}"
        assert not torch.isinf(param).any(), f"Inf found in {name}"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Local Qwen3-0.6B checkpoint not found")
@torch.inference_mode()
def test_qwen3_prefill_and_decode_matches_huggingface() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16
    ).to("cuda")
    hf_model.eval()

    model, cfg = _build_and_load_mini_qwen3()
    num_blocks = 4
    _kv_cache = _bind_kv_cache(model, cfg, num_blocks=num_blocks)
    bm = BlockManager(num_blocks=num_blocks)
    runner = ModelRunner()

    prompts = ["The capital of France is", "1 + 1 ="]
    seqs = [Sequence(tokenizer.encode(p)) for p in prompts]
    for s in seqs:
        bm.allocate(s)

    # Ground-truth HF greedy outputs per prompt
    hf_inputs = [torch.tensor([s.token_ids], dtype=torch.long, device="cuda") for s in seqs]
    hf_past = [None, None]
    hf_next_tokens: list[int] = []
    hf_prefill_logits: list[torch.Tensor] = []
    for i, inp in enumerate(hf_inputs):
        out = hf_model(input_ids=inp, use_cache=True)
        hf_past[i] = out.past_key_values
        last_logit = out.logits[0, -1]
        hf_prefill_logits.append(last_logit)
        hf_next_tokens.append(int(last_logit.argmax().item()))

    try:
        # 1. Prefill batch in mini_infer
        input_ids = torch.tensor(
            [tok for s in seqs for tok in s.token_ids],
            dtype=torch.int64,
            device="cuda",
        )
        positions = torch.tensor(
            [pos for s in seqs for pos in range(len(s.token_ids))],
            dtype=torch.int64,
            device="cuda",
        )
        runner.prepare_prefill_context(seqs)
        hidden = model(input_ids, positions)
        logits = model.compute_logits(hidden)
        reset_context()

        assert logits.shape == (2, cfg.vocab_size)
        mini_next_tokens = logits.argmax(dim=-1).tolist()
        assert mini_next_tokens == hf_next_tokens
        for i in range(2):
            torch.testing.assert_close(logits[i], hf_prefill_logits[i], atol=0.5, rtol=2e-2)

        # 2. Decode 3 steps and verify greedy token-by-token match
        for _step in range(3):
            for i, s in enumerate(seqs):
                s.append_token(mini_next_tokens[i])
                bm.may_append(s)

            # HF step
            hf_step_tokens: list[int] = []
            hf_step_logits: list[torch.Tensor] = []
            for i in range(2):
                tok_t = torch.tensor([[mini_next_tokens[i]]], dtype=torch.long, device="cuda")
                out = hf_model(input_ids=tok_t, past_key_values=hf_past[i], use_cache=True)
                hf_past[i] = out.past_key_values
                last_logit = out.logits[0, -1]
                hf_step_logits.append(last_logit)
                hf_step_tokens.append(int(last_logit.argmax().item()))

            # mini_infer step
            dec_input_ids = torch.tensor(
                [s.token_ids[-1] for s in seqs], dtype=torch.int64, device="cuda"
            )
            dec_positions = torch.tensor(
                [len(s.token_ids) - 1 for s in seqs], dtype=torch.int64, device="cuda"
            )
            runner.prepare_decode_context(seqs)
            dec_hidden = model(dec_input_ids, dec_positions)
            dec_logits = model.compute_logits(dec_hidden)
            reset_context()

            mini_next_tokens = dec_logits.argmax(dim=-1).tolist()
            assert mini_next_tokens == hf_step_tokens
            for i in range(2):
                torch.testing.assert_close(dec_logits[i], hf_step_logits[i], atol=0.5, rtol=2e-2)
    finally:
        reset_context()
        for s in seqs:
            bm.deallocate(s)

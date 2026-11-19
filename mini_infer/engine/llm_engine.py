import os
from dataclasses import dataclass, replace
from typing import cast

from transformers import AutoTokenizer

from mini_infer.config.engine_config import ScheduleConfig
from mini_infer.engine.model_runner import ModelRunner
from mini_infer.engine.scheduler import Scheduler, SequenceStatus
from mini_infer.engine.sequence import Sequence


@dataclass
class SamplingParams:
    temperature: float = 0.1
    max_tokens: int = 64
    ignore_eos: bool = False


class LLMEngine:

    def __init__(self,
                 model_arch: str,
                 model_path: str | os.PathLike,
                 schedule_config: ScheduleConfig | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_runner = ModelRunner(model_arch, model_path)
        if schedule_config is None:
            schedule_config = ScheduleConfig()
        schedule_config = replace(schedule_config,
                                  eos_token_id=self.tokenizer.eos_token_id)
        self.scheduler = Scheduler(self.model_runner.num_blocks,
                                   schedule_config=schedule_config)

    def add_request(self, prompt: str | list[int],
                    sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        prompt = cast(list[int], prompt)
        seq = Sequence(prompt,
                       sampling_params.temperature,
                       max_tokens=sampling_params.max_tokens,
                       ignore_eos=sampling_params.ignore_eos)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self) -> list[tuple[int, list[int]]]:
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.token_ids[seq.num_prompt_tokens:]) for seq in seqs
                   if seq.status == SequenceStatus.FINISHED]
        return outputs

    def generate(self, prompts: list[str],
                 sampling_params: list[SamplingParams | None]) -> list[dict]:
        req_ids: list[int] = []
        outputs: dict[int, list[int]] = {}
        for prompt, sampling_param in zip(prompts, sampling_params):
            if sampling_param is None:
                sampling_param = SamplingParams()
            req_ids.append(self.add_request(prompt, sampling_param))

        while not self.scheduler.is_finished:
            for seq_id, token_ids in self.step():
                outputs[seq_id] = token_ids

        return [{
            "text": self.tokenizer.decode(outputs[rid], skip_special_tokens=True),
            "token_ids": outputs[rid],
        } for rid in req_ids]

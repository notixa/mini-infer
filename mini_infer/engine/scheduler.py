from collections import deque

from mini_infer.config.engine_config import ScheduleConfig
from mini_infer.engine.block_manager import BlockManager
from mini_infer.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, num_blocks: int, schedule_config: ScheduleConfig | None = None):
        if schedule_config is None:
            schedule_config = ScheduleConfig()
        self.schedule_config: ScheduleConfig = schedule_config
        self.max_num_seqs = schedule_config.max_num_seqs
        self.max_num_batched_tokens = schedule_config.max_num_batched_tokens
        self.block_manager = BlockManager(num_blocks)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        budget = self.max_num_batched_tokens
        is_pure_decode = True
        preempted = False
        while self.running and len(scheduled_seqs) < self.max_num_seqs and budget > 0:
            seq = self.running.popleft()
            is_prefill = (seq.num_computed_tokens
                          < seq.num_prompt_tokens) or (len(seq.token_ids) -
                                                       seq.num_computed_tokens > 1)
            if is_prefill:
                remaining = len(seq.token_ids) - seq.num_computed_tokens
                chunk_size = min(budget, remaining)
                seq.num_scheduled_tokens = chunk_size
                scheduled_seqs.append(seq)
                is_pure_decode = False
                budget -= chunk_size
            else:
                while not self.block_manager.can_append(seq):
                    preempted = True
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break

                else:
                    seq.num_scheduled_tokens = 1
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
                    budget -= 1

        self.running.extendleft(reversed(scheduled_seqs))

        if not preempted:
            while self.waiting and len(
                    scheduled_seqs) < self.max_num_seqs and budget > 0:
                seq = self.waiting[0]
                if not seq.block_table:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        break
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)

                remaining = len(seq.token_ids) - seq.num_computed_tokens
                chunk_size = min(budget, remaining)
                seq.num_scheduled_tokens = chunk_size
                scheduled_seqs.append(seq)
                is_pure_decode = False
                budget -= chunk_size

        return scheduled_seqs, not is_pure_decode

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            self.block_manager.hash_blocks(seq)
            if seq.num_computed_tokens < len(seq.token_ids):
                continue
            seq.append_token(token_id)
            if not seq.ignore_eos and token_id == self.schedule_config.eos_token_id or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    @property
    def is_finished(self) -> bool:
        return not self.waiting and not self.running

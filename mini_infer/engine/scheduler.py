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
        num_batched_tokens = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq.token_ids) > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += len(seq.token_ids)
        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break

            else:
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if not seq.ignore_eos and token_id == self.schedule_config.eos_token_id or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    @property
    def is_finished(self) -> bool:
        return not self.waiting and not self.running

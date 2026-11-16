from collections import deque

from mini_infer.config.engine_config import ScheduleConfig
from mini_infer.engine.block_manager import BlockManager
from mini_infer.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, num_blocks: int, schedule_config: ScheduleConfig | None = None):
        if schedule_config is None:
            schedule_config = ScheduleConfig()
        self.schedule_config_: ScheduleConfig = schedule_config
        self.max_num_seqs_ = schedule_config.max_num_seqs_
        self.max_num_batched_tokens_ = schedule_config.max_num_batched_tokens_
        self.block_manager_ = BlockManager(num_blocks)
        self.waiting_: deque[Sequence] = deque()
        self.running_: deque[Sequence] = deque()

    def add(self, seq: Sequence):
        self.waiting_.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0
        while self.waiting_ and len(scheduled_seqs) < self.max_num_seqs_:
            seq = self.waiting_[0]
            if num_batched_tokens + len(seq.token_ids_) > self.max_num_batched_tokens_:
                break
            if not self.block_manager_.can_allocate(seq):
                break
            self.block_manager_.allocate(seq)
            seq.status_ = SequenceStatus.RUNNING
            self.waiting_.popleft()
            self.running_.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += len(seq.token_ids_)
        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running_ and len(scheduled_seqs) < self.max_num_seqs_:
            seq = self.running_.popleft()
            while not self.block_manager_.can_append(seq):
                if self.running_:
                    self.preempt(self.running_.pop())
                else:
                    self.preempt(seq)
                    break

            else:
                self.block_manager_.may_append(seq)
                scheduled_seqs.append(seq)

        self.running_.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status_ = SequenceStatus.WAITING
        self.block_manager_.deallocate(seq)
        self.waiting_.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if not seq.ignore_eos_ and token_id == self.schedule_config_.eos_token_id_ or seq.num_completion_tokens >= seq.max_tokens_:
                seq.status_ = SequenceStatus.FINISHED
                self.block_manager_.deallocate(seq)
                self.running_.remove(seq)

    @property
    def is_finished(self) -> bool:
        return not self.waiting_ and not self.running_

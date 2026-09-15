"""Priority admission queue with FCFS ordering inside each class."""
from collections import deque


class AdmissionQueue:
    def __init__(self, priorities=(0, 1)):
        self.queues = {priority: deque() for priority in priorities}

    def push(self, request):
        self.queues.setdefault(request.priority, deque()).append(request)

    def resume(self, request):
        self.queues.setdefault(request.priority, deque()).appendleft(request)

    def pop(self):
        # FCFS within a class is deliberate. Under load it can cause cache
        # thrashing; a prefix-aware policy is an explicit future tradeoff.
        for priority in sorted(self.queues, reverse=True):
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None

    def pop_ready(self, now):
        for priority in sorted(self.queues, reverse=True):
            if self.queues[priority] and self.queues[priority][0].arrival_time <= now:
                return self.queues[priority].popleft()
        return None

    def peek(self):
        for priority in sorted(self.queues, reverse=True):
            if self.queues[priority]:
                return self.queues[priority][0]
        return None

    def next_arrival(self):
        values = [r.arrival_time for queue in self.queues.values() for r in queue]
        return min(values) if values else None

    def prefix_hashes(self):
        return {r.prefix_hash for queue in self.queues.values() for r in queue if r.prefix_hash}

    def __bool__(self):
        return any(self.queues.values())

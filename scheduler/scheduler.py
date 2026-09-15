"""Main continuous-batching scheduler."""
from dataclasses import dataclass
from .admission import AdmissionQueue
from .batching import build_batch
from .kv_cache import PagedKVCache

@dataclass
class Request:
    id: str; prompt_tokens: int; max_output_tokens: int; priority: int; arrival_time: float; prefix_hash: str|None=None
    state: str = "WAITING"; prefetched: int = 0; output_tokens: int = 0; first_token_time: float|None = None; done_time: float|None = None
    def __post_init__(self):
        if self.prompt_tokens < 0 or self.max_output_tokens < 1:
            raise ValueError("prompt_tokens must be non-negative and max_output_tokens must be positive")
    @property
    def remaining_prefill(self): return self.prompt_tokens-self.prefetched

@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 32; max_num_seqs: int = 8; ttft_slo: float = .1; tpot_slo: float = .02
    cache_blocks: int = 512; block_size: int = 16

class Scheduler:
    def __init__(self, engine, config=None):
        config = config or SchedulerConfig()
        self.engine, self.config, self.waiting, self.running, self.now = engine, config, AdmissionQueue(), [], 0.0
        self.cache = PagedKVCache(config.cache_blocks, config.block_size); self.logs=[]; self.peak_kv_utilization = 0.0
    def submit(self, request):
        if request.state != "WAITING":
            raise ValueError(f"only WAITING requests may be submitted, got {request.state}")
        if any(request.id == existing.id for existing in self.running) or any(
            request.id == existing.id for queue in self.waiting.queues.values() for existing in queue
        ):
            raise ValueError(f"duplicate request id: {request.id}")
        self.waiting.push(request)

    def cancel(self, request_id):
        """Terminate a queued or running request and return whether it existed."""
        for queue in self.waiting.queues.values():
            for request in list(queue):
                if request.id == request_id:
                    queue.remove(request); request.state = "ABORTED"; request.done_time = self.now
                    self._release_engine(request.id)
                    return True
        for request in list(self.running):
            if request.id == request_id:
                self.running.remove(request); self.cache.free_request(request.id)
                request.state = "ABORTED"; request.done_time = self.now; self._release_engine(request.id)
                self.cache.assert_invariants(); return True
        return False
    def _preempt(self, incoming):
        victims = [r for r in self.running if r.priority < incoming.priority]
        if not victims: return False
        victim = min(victims, key=lambda r: r.priority)
        self.running.remove(victim); victim.state="PREEMPTED"; cost=victim.prefetched+victim.output_tokens
        self.cache.free_request(victim.id); self._release_engine(victim.id); victim.prefetched=0; victim.output_tokens=0; victim.state="WAITING"; self.waiting.push(victim)
        self.logs.append(f"preempt request={victim.id} for={incoming.id} recompute_tokens={cost}")
        return True
    def step(self):
        batch = build_batch(self.running, self.waiting, self.config.max_num_batched_tokens, self.config.max_num_seqs, self.now)
        if not batch:
            next_arrival = self.waiting.next_arrival()
            if next_arrival is not None and next_arrival > self.now:
                self.now = next_arrival
                return True
            return False
        # Reserve logical pages before issuing real engine work. Otherwise a
        # failed admission could leave an engine cache that the scheduler does
        # not own. This is especially important for TransformersEngine.
        prepared = []
        for item in batch:
            if item.kind != "prefill" or self._reserve_prefill(item.request):
                prepared.append(item)
            else:
                self.waiting.push(item.request)
        # Reserving a high-priority prefill can preempt a decode that was
        # tentatively selected at the start of this same scheduling step.
        prepared = [item for item in prepared if item.kind != "decode" or item.request in self.running]
        if not prepared:
            return False
        batch = prepared
        duration = self.engine.step(batch)
        completed_at = self.now + duration
        for item in batch:
            r=item.request
            if item.kind == "prefill":
                r.state="PREFILLING"; r.prefetched += min(item.tokens, r.remaining_prefill)
                if r.remaining_prefill == 0:
                    r.state="DECODING"; self.running.append(r)
                else:
                    # Chunked prefill yields back to admission so decode work
                    # can run in the next iteration instead of being blocked.
                    r.state="WAITING"; self.waiting.resume(r)
            else:
                r.output_tokens += 1; self.cache.touch(r.id)
                if r.first_token_time is None: r.first_token_time=completed_at
                if r.output_tokens >= r.max_output_tokens:
                    r.state="DONE"; r.done_time=completed_at; self.running.remove(r); self.cache.free_request(r.id, keep_prefix=True)
                    self._release_engine(r.id, completed=True)
        self.now = completed_at; self.cache.assert_invariants()
        self.peak_kv_utilization = max(self.peak_kv_utilization, self.cache.utilization())
        self.logs.append("batch="+",".join(f"{x.request.id}:{x.kind}:{x.tokens}" for x in batch))
        return True
    def run(self):
        while self.step(): pass

    def _release_engine(self, request_id, completed=False):
        release = getattr(self.engine, "release", None)
        if release:
            release(request_id, completed=completed)

    def _reserve_prefill(self, request):
        """Reserve logical pages before the engine creates request-local KV."""
        if request.id in self.cache.by_request:
            return True
        reused = self.cache.prefix_tokens(request.prefix_hash) if getattr(self.engine, "supports_prefix_reuse", True) else 0
        request.prefetched = reused
        needed = (request.remaining_prefill + self.cache.block_size - 1) // self.cache.block_size
        while len(self.cache.free) < needed and self._preempt(request):
            pass
        if not self.cache.allocate(request.id, request.remaining_prefill, request.prefix_hash, bool(request.prefix_hash)):
            hashes = self.waiting.prefix_hashes() | ({request.prefix_hash} if request.prefix_hash else set())
            self.cache.evict_unused_prefixes(hashes)
        if request.id not in self.cache.by_request:
            return self.cache.allocate(request.id, request.remaining_prefill, request.prefix_hash, bool(request.prefix_hash))
        return True

    def assert_invariants(self):
        self.cache.assert_invariants()
        running_ids = [request.id for request in self.running]
        waiting_ids = [request.id for queue in self.waiting.queues.values() for request in queue]
        assert len(running_ids) == len(set(running_ids)), "duplicate running request"
        assert not set(running_ids) & set(waiting_ids), "request is running and waiting"
        assert all(request.state == "DECODING" for request in self.running)
        assert all(request.state == "WAITING" for queue in self.waiting.queues.values() for request in queue)

    @property
    def preemption_events(self):
        return [line for line in self.logs if line.startswith("preempt ")]

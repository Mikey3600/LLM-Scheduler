"""Incremental optional Transformers engine with per-request *real* KV state.

The scheduler's :class:`PagedKVCache` remains a logical allocator.  This
engine owns independent framework ``past_key_values`` tensors and never maps
logical pages to model tensor addresses.
"""
from dataclasses import dataclass, field
import time


@dataclass
class _RequestState:
    prefill_started_at: float | None = None
    past_key_values: object | None = None
    next_input_id: int | None = None
    generated_ids: list[int] = field(default_factory=list)
    prefill_finished_at: float | None = None
    decode_times: list[float] = field(default_factory=list)
    decode_finished_at: list[float] = field(default_factory=list)
    completed_at: float | None = None


class TransformersEngine:
    """Run prefill chunks and one-token decode calls using retained HF caches.

    Work is deliberately executed request-by-request because separate
    ``past_key_values`` objects cannot be naively stacked into a Transformers
    batch.  The scheduler interface remains ``step(items) -> seconds``.
    """

    supports_prefix_reuse = False

    def __init__(self, model_name="sshleifer/tiny-gpt2", device="auto", local_files_only=False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Real backend requires optional dependencies. Run: pip install -r requirements-real.txt"
            ) from exc
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable; use device='cpu' or 'auto'.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only).to(device).eval()
        except OSError as exc:
            where = "local cache" if local_files_only else "the Hugging Face model source"
            raise RuntimeError(f"Could not load {model_name!r} from {where}. Download it first or check network access.") from exc
        self.torch, self.device = torch, device
        encoded = self.tokenizer.encode(" scheduler", add_special_tokens=False)
        self.prompt_token_id = encoded[0] if encoded else self.tokenizer.eos_token_id
        self.states: dict[str, _RequestState] = {}
        self.completed: dict[str, _RequestState] = {}
        self.peak_memory_bytes = 0
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    def _ids(self, token_id, count):
        return self.torch.full((1, count), token_id, dtype=self.torch.long, device=self.device)

    def _update_peak_memory(self):
        if self.device.startswith("cuda"):
            self.peak_memory_bytes = max(self.peak_memory_bytes, self.torch.cuda.max_memory_allocated(self.device))

    def step(self, items):
        """Execute item work, retaining true ``past_key_values`` by request id."""
        started = time.perf_counter()
        with self.torch.inference_mode():
            for item in items:
                state = self.states.setdefault(item.request.id, _RequestState())
                if item.kind == "prefill":
                    if state.prefill_started_at is None:
                        state.prefill_started_at = time.perf_counter()
                    output = self.model(
                        input_ids=self._ids(self.prompt_token_id, item.tokens),
                        past_key_values=state.past_key_values,
                        use_cache=True,
                    )
                    state.past_key_values = output.past_key_values
                    state.next_input_id = int(output.logits[0, -1].argmax())
                    if item.tokens == item.request.remaining_prefill:
                        state.prefill_finished_at = time.perf_counter()
                elif item.kind == "decode":
                    if state.past_key_values is None or state.next_input_id is None:
                        raise RuntimeError(f"Decode for {item.request.id!r} has no real model KV state.")
                    token_started = time.perf_counter()
                    output = self.model(
                        input_ids=self._ids(state.next_input_id, 1),
                        past_key_values=state.past_key_values,
                        use_cache=True,
                    )
                    state.past_key_values = output.past_key_values
                    state.next_input_id = int(output.logits[0, -1].argmax())
                    state.generated_ids.append(state.next_input_id)
                    state.decode_times.append(time.perf_counter() - token_started)
                    state.decode_finished_at.append(time.perf_counter())
                else:
                    raise ValueError(f"Unknown engine work kind: {item.kind}")
        self._update_peak_memory()
        return time.perf_counter() - started

    def release(self, request_id, completed=False):
        """Drop tensor references on completion, cancellation, or preemption."""
        state = self.states.pop(request_id, None)
        if state is not None and completed:
            state.completed_at = time.perf_counter()
            self.completed[request_id] = state
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def generated_ids(self, request_id):
        state = self.states.get(request_id) or self.completed.get(request_id)
        return list(state.generated_ids) if state else []

    def real_metrics(self):
        """Wall-clock request measurements from actual model calls only."""
        rows = []
        for request_id, state in self.completed.items():
            if not state.prefill_started_at or not state.decode_finished_at:
                continue
            first = state.decode_finished_at[0]
            inter_token = [b - a for a, b in zip(state.decode_finished_at, state.decode_finished_at[1:])]
            rows.append({
                "id": request_id,
                "ttft_s": first - state.prefill_started_at,
                "mean_inter_token_s": sum(inter_token) / len(inter_token) if inter_token else 0.0,
                "completion_latency_s": (state.completed_at or first) - state.prefill_started_at,
                "generated_tokens": len(state.generated_ids),
            })
        elapsed = sum(row["completion_latency_s"] for row in rows)
        return {"requests": len(rows), "rows": rows,
                "generated_tokens_s": sum(row["generated_tokens"] for row in rows) / elapsed if elapsed else 0.0,
                "peak_memory_bytes": self.peak_memory_bytes}

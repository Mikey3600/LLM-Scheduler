"""Deterministic zero-dependency engine latency model."""

class SimEngine:
    supports_prefix_reuse = True
    def __init__(self, prefill_ms_per_token=0.2, decode_ms_per_token=1.0):
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token

    def step(self, items):
        return sum((self.prefill_ms_per_token if i.kind == "prefill" else self.decode_ms_per_token) * i.tokens for i in items) / 1000

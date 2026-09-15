"""Iteration-level token-budget batch construction."""
from dataclasses import dataclass


@dataclass
class BatchItem:
    request: object
    tokens: int
    kind: str


def build_batch(running, waiting, token_budget, max_seqs, now=float("inf")):
    """Schedule decode first, then chunked-prefill waiting requests."""
    items, remaining = [], token_budget
    for request in list(running):
        if len(items) == max_seqs or remaining < 1:
            break
        items.append(BatchItem(request, 1, "decode"))
        remaining -= 1
    while waiting and len(items) < max_seqs and remaining:
        request = waiting.pop_ready(now)
        if request is None:
            break
        requested = request.remaining_prefill
        if requested == 0:
            break
        chunk = min(requested, remaining)
        items.append(BatchItem(request, chunk, "prefill"))
        remaining -= chunk
    return items

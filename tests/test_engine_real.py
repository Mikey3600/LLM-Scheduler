"""Optional integration test: runs only when a cached Transformers model exists."""
import pytest

from scheduler.batching import BatchItem
from scheduler.engine_real import TransformersEngine
from scheduler.scheduler import Request


def local_engine_or_skip():
    try:
        return TransformersEngine(local_files_only=True, device="cpu")
    except RuntimeError as exc:
        pytest.skip(f"real engine unavailable locally: {exc}")


def test_chunked_prefill_decode_matches_uninterrupted_continuation():
    engine = local_engine_or_skip()
    full = Request("full", 6, 3, 0, 0)
    chunked = Request("chunked", 6, 3, 0, 0)

    engine.step([BatchItem(full, 6, "prefill")])
    for _ in range(3):
        engine.step([BatchItem(full, 1, "decode")])

    engine.step([BatchItem(chunked, 2, "prefill")])
    engine.step([BatchItem(chunked, 4, "prefill")])
    for _ in range(3):
        engine.step([BatchItem(chunked, 1, "decode")])

    assert engine.generated_ids("full") == engine.generated_ids("chunked")

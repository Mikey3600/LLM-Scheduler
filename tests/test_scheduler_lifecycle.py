import pytest

from benchmark.load_gen import generate
from scheduler.engine_sim import SimEngine
from scheduler.scheduler import Request, Scheduler, SchedulerConfig


def make_scheduler(**overrides):
    return Scheduler(SimEngine(), SchedulerConfig(**overrides))


def test_cancel_removes_running_pages_and_marks_terminal_state():
    scheduler = make_scheduler(cache_blocks=2, block_size=4)
    request = Request("cancel-me", 4, 3, 0, 0)
    scheduler.submit(request)
    scheduler.step()  # prefill completes and request enters decoding

    assert scheduler.cancel("cancel-me")
    assert request.state == "ABORTED"
    assert request.done_time == scheduler.now
    assert not scheduler.running
    assert len(scheduler.cache.free) == 2
    scheduler.assert_invariants()
    assert not scheduler.cancel("missing")


def test_same_priority_fcfs_requests_complete_without_starvation():
    scheduler = make_scheduler(max_num_batched_tokens=4, max_num_seqs=1)
    requests = [Request(str(index), 2, 2, 0, 0) for index in range(4)]
    for request in requests:
        scheduler.submit(request)
    scheduler.run()

    assert [request.state for request in requests] == ["DONE"] * 4
    assert [request.done_time for request in requests] == sorted(request.done_time for request in requests)
    scheduler.assert_invariants()


def replay(seed):
    scheduler = make_scheduler()
    requests = generate(25, count=12, seed=seed)
    for request in requests:
        scheduler.submit(request)
    scheduler.run()
    return scheduler.logs, [(request.id, request.state, request.first_token_time, request.done_time) for request in requests]


def test_seeded_workload_replays_identically():
    assert replay(19) == replay(19)


def test_rejects_invalid_token_count_and_duplicate_request_id():
    with pytest.raises(ValueError):
        Request("invalid", 1, 0, 0, 0)
    scheduler = make_scheduler()
    scheduler.submit(Request("duplicate", 1, 1, 0, 0))
    with pytest.raises(ValueError, match="duplicate request id"):
        scheduler.submit(Request("duplicate", 1, 1, 0, 0))


@pytest.mark.parametrize("priority", [-1, 0, 1, 5])
def test_priority_values_are_deterministic(priority):
    scheduler = make_scheduler()
    request = Request("request", 1, 1, priority, 0)
    scheduler.submit(request)
    scheduler.run()
    assert request.state == "DONE"

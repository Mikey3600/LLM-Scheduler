from scheduler.scheduler import Scheduler, SchedulerConfig, Request
from scheduler.engine_sim import SimEngine
def test_higher_priority_request_preempts_and_logs_cost():
    s=Scheduler(SimEngine(),SchedulerConfig(cache_blocks=1,block_size=4)); low=Request('low',4,3,0,0); high=Request('high',4,2,1,0)
    s.submit(low); s.step(); s.submit(high); s.step()
    # Admission happens before the engine executes the tentative decode, so
    # only the four prefetched tokens must be recomputed.
    assert any('preempt request=low for=high recompute_tokens=4' in x for x in s.logs)

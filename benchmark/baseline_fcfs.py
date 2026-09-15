"""Naive sequential FCFS comparison baseline."""
from scheduler.engine_sim import SimEngine
from scheduler.metrics import summarize

def run(requests, config):
    engine=SimEngine(); now=0.
    for r in requests:
        now=max(now,r.arrival_time)+r.prompt_tokens*engine.prefill_ms_per_token/1000
        r.first_token_time=now+engine.decode_ms_per_token/1000
        now += r.max_output_tokens*engine.decode_ms_per_token/1000
        r.output_tokens=r.max_output_tokens; r.done_time=now; r.state="DONE"
    return summarize(requests, config.ttft_slo, config.tpot_slo, now)

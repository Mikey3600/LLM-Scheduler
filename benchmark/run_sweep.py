"""Run reproducible simulator or optional real-engine SLO sweeps."""
import argparse, csv, os
from benchmark.load_gen import generate
from benchmark.baseline_fcfs import run as baseline_run
from scheduler.engine_sim import SimEngine
from scheduler.engine_real import TransformersEngine
from scheduler.metrics import summarize, goodput
from scheduler.scheduler import Scheduler, SchedulerConfig

def main():
    p=argparse.ArgumentParser(); p.add_argument('--backend',default='sim',choices=['sim','real']); p.add_argument('--output',default='benchmark/results'); args=p.parse_args()
    cfg=SchedulerConfig(); rows=[]
    engine_factory=SimEngine if args.backend=='sim' else TransformersEngine
    for rate in (10,25,50,75):
        reqs=generate(rate); s=Scheduler(engine_factory(),cfg)
        for r in reqs: s.submit(r)
        s.run(); result=summarize(reqs,cfg.ttft_slo,cfg.tpot_slo,s.now)
        if args.backend == 'real':
            # These are wall-clock engine measurements, distinct from the
            # scheduler's simulated-time SLO summary above.
            result.update({f"real_{key}": value for key, value in s.engine.real_metrics().items() if key != "rows"})
        costs = [int(event.rsplit('=', 1)[1]) for event in s.preemption_events]
        result.update(rate=rate, scheduler='continuous', preemption_count=len(costs),
                      preemption_recompute_tokens=sum(costs), peak_kv_utilization=s.peak_kv_utilization)
        rows.append(result)
        # FCFS uses the deterministic simulator and is only a like-for-like
        # comparison for the simulator sweep. Do not label it as real-model data.
        if args.backend == 'sim':
            base=baseline_run(generate(rate),cfg)
            rows.append(dict(base,rate=rate,scheduler='fcfs'))
    os.makedirs(args.output,exist_ok=True); path=os.path.join(args.output,f'{args.backend}_sweep.csv')
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['backend','scheduler','rate','requests','slo_attainment','throughput_tokens_s','generated_tokens_s','mean_ttft','p50_ttft','p50_latency','p90_latency','p95_latency','p99_latency','preemption_count','preemption_recompute_tokens','peak_kv_utilization','real_requests','real_generated_tokens_s','real_peak_memory_bytes'], lineterminator='\n'); writer.writeheader()
        for r in rows: writer.writerow({k:(args.backend if k=='backend' else r.get(k,'')) for k in writer.fieldnames})
    continuous=[r for r in rows if r['scheduler']=='continuous']
    if args.backend == 'sim':
        print(f"backend=sim goodput_rps={goodput(continuous)} csv={path}")
    else:
        print(f"backend=real csv={path} (wall-clock measurements only; no simulated FCFS comparison)")
if __name__=='__main__': main()

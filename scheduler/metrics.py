"""Per-request SLO metrics and sweep-derived goodput."""
import math

def percentile(values, p):
    if not values: return 0.0
    values = sorted(values); return values[min(len(values)-1, math.ceil(p * len(values))-1)]

def summarize(requests, ttft_slo, tpot_slo, elapsed):
    finished = [r for r in requests if r.done_time is not None]
    rows = []
    for r in finished:
        ttft = r.first_token_time - r.arrival_time
        tpot = (r.done_time - r.first_token_time) / max(1, r.output_tokens - 1)
        rows.append({"id": r.id, "ttft": ttft, "tpot": tpot, "latency": r.done_time-r.arrival_time,
                     "slo_ok": ttft <= ttft_slo and tpot <= tpot_slo})
    attainment = sum(x["slo_ok"] for x in rows) / len(rows) if rows else 0
    return {"requests": len(rows), "slo_attainment": attainment,
            "throughput_tokens_s": sum(r.prompt_tokens+r.output_tokens for r in finished)/elapsed if elapsed else 0,
            "generated_tokens_s": sum(r.output_tokens for r in finished)/elapsed if elapsed else 0,
            "mean_ttft": sum(x["ttft"] for x in rows)/len(rows) if rows else 0,
            "p50_ttft": percentile([x["ttft"] for x in rows], .5),
            "p95_latency": percentile([x["latency"] for x in rows], .95),
            "p50_latency": percentile([x["latency"] for x in rows], .5),
            "p90_latency": percentile([x["latency"] for x in rows], .9),
            "p99_latency": percentile([x["latency"] for x in rows], .99), "rows": rows}

def goodput(results):
    eligible = [r["rate"] for r in results if r["slo_attainment"] >= .90]
    return max(eligible, default=0)

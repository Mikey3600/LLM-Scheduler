# LLM Inference Scheduler

This project explores a narrow systems question: when several generation
requests share one model, what happens to latency when prompt work, decode
work, and limited KV capacity compete? The code is intentionally a small
single-node experiment rather than a service.

I started with a deterministic simulator so that scheduling decisions can be
replayed exactly. An optional Transformers adapter then checks one important
assumption against a real model: prefill can retain a model cache and decode
can advance it one token at a time without rerunning the prompt.

## Question and model

The scheduler gives existing decodes one token each before admitting waiting
prefill. Long prompts are split by a token budget, so they yield between
iterations. Requests have FCFS ordering within priority classes; a higher
priority allocation may preempt a lower priority decode and record how many
logical tokens must be recomputed.

```
client -> priority admission -> waiting queue
                              |
                    per-step scheduler
                     /       |       \
              running   logical KV   token budget
                     \       |       /
                        engine step
```

The **logical KV cache** is fixed-size block accounting used to explore
admission, eviction, prefix reuse, and preemption. It is not memory owned by a
model framework. Prefix blocks are protected from LRU eviction while a waiting
request may still use that prefix.

The optional **real model KV cache** is the Transformers `past_key_values`
object kept per request. It is separate from logical blocks: the engine owns
and releases those tensors at completion, cancellation, or preemption. Logical
prefix reuse is disabled for the real engine because a logical prefix hit is
not a tensor-cache hit.

## What is implemented

* Continuous iteration-level batching with decode-first ordering and chunked
  prefill.
* Deterministic simulator latency, priority admission, logical-page LRU,
  protected prefixes, cancellation, and preemption logging.
* TTFT, TPOT, joint-SLO attainment, latency percentiles, generated-token
  throughput, preemption cost, and logical-KV utilization.
* An optional CPU/CUDA Transformers adapter that performs incremental prefill
  and single-token decode with retained `past_key_values`. Separate request
  caches are executed serially within a scheduler step; this is a correctness
  adapter, not a batched model-cache implementation.

## Reproduction

The simulator path needs only Python and pytest:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m benchmark.run_sweep --backend sim
```

The simulator sweep is seeded (`seed=7`), uses 30 Poisson-arrival requests at
each offered rate, and writes the checked-in canonical result
`benchmark/results/sim_sweep.csv`. `config.yaml` records the matching defaults;
it is not parsed by the current command-line runner.

For real-model validation, install the optional stack and run the cached-model
continuation test and smoke sweep. The default model is
`sshleifer/tiny-gpt2`; it must be downloadable or already present locally.

```bash
python -m pip install -r requirements-real.txt
python -m pytest tests/test_engine_real.py -q
python -m benchmark.run_sweep --backend real
```

An unavailable CUDA device produces a clear error; `device="auto"` selects
CPU when CUDA is absent. The real sweep records wall-clock engine measurements
only. It deliberately does not attach the simulator FCFS baseline or derive
simulated-time SLO goodput from real executions.

## Simulator results

The checked-in artifact was produced by `python -m benchmark.run_sweep
--backend sim` in this repository. The offered-rate sweep reached the
configured 90% joint-SLO threshold at every tested rate, so its simulator
goodput is **75 req/s**. These are clock-model results, not model-serving
performance measurements.

| offered req/s | continuous SLO attainment | continuous P99 s | FCFS P99 s |
|---:|---:|---:|---:|
| 10 | 1.00 | 0.03377 | 0.02805 |
| 25 | 1.00 | 0.04360 | 0.03798 |
| 50 | 1.00 | 0.05860 | 0.04433 |
| 75 | 1.00 | 0.10862 | 0.09039 |

The result was less dramatic than a feature list might suggest: both policies
meet this loose, small simulated workload. No preemptions occurred in this
sweep, so it demonstrates neither a preemption advantage nor a throughput
advantage over FCFS. The CSV contains the supporting TTFT, latency,
throughput, preemption, and logical-KV columns.

## Real model results

No real model result is checked in. In the verification environment used for
this release, `torch` and `transformers` were not installed, so the real smoke
command could not load a model and the optional continuation test was skipped.
No real latency, throughput, or memory number is claimed here.

## Limits and next questions

The simulator does not model GPU kernels or tensor allocation. The real
adapter uses synthetic repeated token IDs because requests carry token counts,
not prompt text; it does not stream text, batch independent framework caches,
or map logical pages to tensor memory. FCFS within a priority class can also
starve lower priorities under an unbounded high-priority stream.

The three next questions worth answering are: (1) load runtime settings from
the config file and add a bounded contention workload that actually exercises
preemption; (2) batch compatible real model cache states while retaining the
same continuation test; and (3) measure real CPU/GPU memory and latency on a
named, pinned model and machine before making any performance comparison.

# Phase 0 Results: Offline Benchmark

Oct 8, 2026. All numbers below come from the dev host GPU, an **NVIDIA
GeForce RTX 5090** (driver 580.173.02, torch 2.14.1+cu130). They are not
RTX 3060 results; the 3060 measurement is still to be done.

## Summary

LM rescoring of libchewing candidates beats the libchewing 1-best on the
held out test set. With the smallest model (Qwen2.5-0.5B, FP16) and the
proposal's top-10 candidates:

- Full zhuyin with tone: CER 10.02% -> 6.06% (39.5% relative reduction,
  95% CI 36.9% to 42.2%), sentence accuracy 47.0% -> 66.2%.
- No tone: CER 29.60% -> 23.02% (22.2% relative reduction).
- Initials only: CER 82.33% -> 78.48% (4.7%). Candidate generation, not
  rescoring, is the bottleneck here: even Oracle@30 is 65.87% CER.

The proposal's accuracy criterion (>= 20% relative CER reduction) is met for
`full` and `notone` with every model tried, and not met for `initial`.
Latency on the 5090 is p95 20.4 ms (0.5B, k=10), well inside the 50 ms
target, but the 3060 and the utilization budget are not measured yet.

## Setup

- **Data.** 2,000 dev and 2,000 test clauses of 5 to 20 Han characters from
  Taiwan news (HF `liswei/news-collection-zhtw`, shard 2 of 5), split by
  article (1,000 articles per split, at most 2 clauses per article). Clauses
  are delimited by punctuation, which is roughly what a user types before
  committing. Context is the preceding 80 characters of the article; the LM
  keeps the last 64 tokens. Readings come from g2pW (Taiwan readings, no
  example dropped). Only 2 of 2,000 dev sentences contain a (character,
  reading) pair that libchewing does not know, so baseline errors are real
  selection errors. Built by `scripts/build_dataset.py`.
- **Input conditions.** `full` (tone marks), `notone`, `initial` (first
  symbol of each syllable). libchewing v0.13.1 with the default dictionary
  and learning disabled; the strict engine for `full`, the fuzzy engine
  (open tone, syllable prefix match) for `notone` and `initial`.
- **Candidates** (`src/zhuyin_rescore/candidates.py`). Pool of up to 30:
  the libchewing 1-best, then the engine's own whole sentence n-best, then
  single substitutions from the per-position candidate lists (frequency
  sorted). Alternatives with characters outside Big5 (cp950) are dropped,
  which removes Simplified only forms such as 头; only 6 of 2,000 dev gold
  sentences contain such characters. The 1-best is never filtered.
- **Scoring** (`src/zhuyin_rescore/scorer.py`). Base (not instruct) models
  in FP16. Score = sum of token log-probs of the candidate given
  `<|endoftext|>` + context, with context and candidate tokenized separately.
  The KV cached path and the full sequence path agree within 1e-3 in FP32
  (unit tests).
- **Fusion.** `score = lm - mu * rank - beta * [not the 1-best]`, where rank
  is the position in the libchewing pool. mu and beta are grid searched on
  dev, then frozen for test. mu = beta = 0 is pure LM selection.
- **Statistics.** 95% CI of the relative CER reduction by a paired bootstrap
  over test sentences (2,000 resamples).

## Results

### Candidate coverage (test)

| condition | 1-best CER | 1-best SentAcc | Oracle@5 | Oracle@10 | Oracle@20 | Oracle@30 |
| --- | --- | --- | --- | --- | --- | --- |
| full | 10.02% | 47.0% | 6.25% | 4.98% | 4.20% | 4.00% |
| notone | 29.60% | 17.1% | 23.25% | 20.71% | 18.26% | 17.46% |
| initial | 82.33% | 0.2% | 76.26% | 73.19% | 67.68% | 65.87% |

The libchewing Tab n-best and the substitutions were also compared alone:
at k=10 the Tab n-best reaches Oracle@10 CER 4.98% on test (full), the
substitutions only 6.38%, which is why the pool puts the n-best first.

### Main result (test CER, relative reduction vs libchewing in parentheses)

| condition | model | libchewing 1-best | LM @10 | LM + fusion @10 | LM + fusion @30 | Oracle@10 / @30 |
| --- | --- | --- | --- | --- | --- | --- |
| full | Qwen2.5-0.5B | 10.02% | 6.02% (39.9%) | 6.06% (39.5%) | 5.26% (47.5%) | 4.98% / 4.00% |
| full | Qwen3-0.6B-Base | 10.02% | 5.95% (40.6%) | 5.94% (40.7%) | 5.08% (49.3%) | 4.98% / 4.00% |
| full | Qwen2.5-1.5B | 10.02% | 5.75% (42.6%) | 5.77% (42.4%) | 4.92% (50.9%) | 4.98% / 4.00% |
| notone | Qwen2.5-0.5B | 29.60% | 23.02% (22.2%) | 23.02% (22.2%) | 21.08% (28.8%) | 20.71% / 17.46% |
| notone | Qwen3-0.6B-Base | 29.60% | 23.14% (21.8%) | 23.18% (21.7%) | 21.24% (28.2%) | 20.71% / 17.46% |
| notone | Qwen2.5-1.5B | 29.60% | 22.74% (23.2%) | 22.74% (23.2%) | 20.68% (30.1%) | 20.71% / 17.46% |
| initial | Qwen2.5-0.5B | 82.33% | 78.54% (4.6%) | 78.48% (4.7%) | 77.49% (5.9%) | 73.19% / 65.87% |
| initial | Qwen3-0.6B-Base | 82.33% | 78.54% (4.6%) | 78.69% (4.4%) | 77.54% (5.8%) | 73.19% / 65.87% |
| initial | Qwen2.5-1.5B | 82.33% | 78.20% (5.0%) | 78.15% (5.1%) | 77.23% (6.2%) | 73.19% / 65.87% |

All three models clear the 20% bar on `full` and `notone`. Going from 0.5B
to 1.5B buys about 3 points of relative reduction on `full` and 1 point on
`notone`; Qwen3-0.6B-Base is on par with Qwen2.5-0.5B. At k=10 the 0.5B model already recovers about 79% of
the errors that are recoverable within the top 10 on `full`.

### Sentence accuracy and stability (test, Qwen2.5-0.5B)

| condition | system | SentAcc | 95% CI of rel. CER reduction | recovered vs Oracle@k | sentences fixed / broken |
| --- | --- | --- | --- | --- | --- |
| full | libchewing 1-best | 47.0% | - | - | - |
| full | LM@10 (mu=0, beta=0) | 66.6% | 37.2% to 42.6% | 79.4% | 431 / 40 |
| full | LM + fusion@10 (mu=0, beta=2) | 66.2% | 36.9% to 42.2% | 78.6% | 395 / 12 |
| full | LM + fusion@30 (mu=0, beta=2) | 68.8% | 44.9% to 50.0% | 79.1% | 458 / 22 |
| notone | libchewing 1-best | 17.1% | - | - | - |
| notone | LM@10 (mu=0, beta=0) | 30.8% | 20.8% to 23.7% | 74.0% | 293 / 19 |
| notone | LM + fusion@10 (mu=0, beta=0) | 30.8% | 20.8% to 23.7% | 74.0% | 293 / 19 |
| notone | LM + fusion@30 (mu=0, beta=0) | 33.0% | 27.3% to 30.3% | 70.2% | 345 / 26 |
| initial | libchewing 1-best | 0.2% | - | - | - |
| initial | LM@10 (mu=0, beta=0) | 0.9% | 4.0% to 5.2% | 41.5% | 13 / 1 |
| initial | LM + fusion@10 (mu=0.1, beta=0) | 0.9% | 4.1% to 5.3% | 42.1% | 13 / 1 |
| initial | LM + fusion@30 (mu=0, beta=0) | 0.9% | 5.2% to 6.5% | 29.4% | 15 / 1 |

The 1-best bonus picked on dev for `full` (beta=2) gives the same CER as
pure LM selection but breaks 12 instead of 40 sentences that libchewing had
right. This matters for the "jumping" preedit risk in the proposal, so the
fused setting is the recommended default.

### Context ablation (test CER, LM + fusion @10, fusion tuned on dev separately)

| condition | model | with context | without context |
| --- | --- | --- | --- |
| full | Qwen2.5-0.5B | 6.06% (39.5%) | 7.22% (27.9%) |
| full | Qwen3-0.6B-Base | 5.94% (40.7%) | 7.54% (24.8%) |
| full | Qwen2.5-1.5B | 5.77% (42.4%) | 6.87% (31.4%) |
| notone | Qwen2.5-0.5B | 23.02% (22.2%) | 25.53% (13.7%) |
| notone | Qwen3-0.6B-Base | 23.18% (21.7%) | 25.52% (13.8%) |
| notone | Qwen2.5-1.5B | 22.74% (23.2%) | 25.06% (15.3%) |
| initial | Qwen2.5-0.5B | 78.48% (4.7%) | 80.96% (1.7%) |
| initial | Qwen3-0.6B-Base | 78.69% (4.4%) | 80.97% (1.7%) |
| initial | Qwen2.5-1.5B | 78.15% (5.1%) | 80.86% (1.8%) |

The committed context is worth about a third of the gain on `full` and
`notone`. Without context the LM still beats libchewing on both.

### Excluding clauses that overlap their context (test, Qwen2.5-0.5B, LM + fusion @10)

| condition | examples kept | libchewing 1-best | LM + fusion @10 |
| --- | --- | --- | --- |
| full | 1907 | 10.02% | 6.09% (39.2%) |
| notone | 1907 | 29.56% | 23.02% (22.1%) |
| initial | 1907 | 82.32% | 78.56% (4.6%) |

About 5% of test clauses share at least half of their text with the
preceding context. Dropping them leaves the gains unchanged, so they do not
come from copying the context.

### Focus characters (test, Qwen2.5-0.5B, LM + fusion @30): gold occurrences, errors libchewing -> rescored

| condition | 在/再 | 的/得/地 | 他/她/它 | 裡/裏 | 著/着 |
| --- | --- | --- | --- | --- | --- |
| full | 164: 12 -> 5 | 530: 19 -> 10 | 76: 24 -> 16 | 17: 0 -> 0 | 16: 0 -> 0 |
| notone | 164: 20 -> 14 | 530: 157 -> 100 | 76: 27 -> 19 | 17: 8 -> 8 | 16: 4 -> 2 |
| initial | 164: 140 -> 132 | 530: 487 -> 464 | 76: 56 -> 51 | 17: 17 -> 16 | 16: 14 -> 14 |

The focus homophones improve. The Taiwan form pairs 裡/裏 and 著/着 show no
regression.

### Latency (RTX 5090 only, not the RTX 3060 target)

| config | scoring p50 ms | scoring p95 ms | context cache p50 ms | VRAM MB |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-float16-k10 | 19.1 | 20.4 | 15.2 | 1226 |
| Qwen2.5-0.5B-float16-k30 | 21.2 | 22.4 | 15.7 | 1748 |
| Qwen2.5-1.5B-float16-k10 | 23.1 | 24.1 | 18.4 | 3236 |
| Qwen2.5-1.5B-float16-k30 | 24.3 | 25.7 | 18.1 | 3813 |
| Qwen3-0.6B-Base-float16-k10 | 28.1 | 28.9 | 23.4 | 1524 |

The scoring time is one batched forward over the candidates with the
context KV cache already resident (built at commit time, "context cache"
column). On the 5090 the 1.5B model is barely slower than the 0.5B model,
so these numbers are dominated by kernel launch and Python overhead, not by
compute; CUDA Graphs are the obvious next step. GPU and CPU utilization
under a replayed typing load is not measured yet.

## Findings That Change the Plan

- **libchewing does have a whole sentence n-best.** Pressing Tab at the end
  of the preedit cycles through the conversion engine's n-best (exactly 10
  sentences including the 1-best, then it wraps). It reaches a better
  Oracle@10 than 1-best plus local substitution, so the proposal's risk "no
  n-best API" is mostly resolved; substitutions are still useful to go past
  10.
- **Initials only input needs a different candidate generator.** The fuzzy
  engine's 1-best is 82% CER and even Oracle@30 is 66%, so rescoring cannot
  help much. A lattice over the dictionary with LM guided beam search is the
  likely fix.
- **Remaining rescoring errors are often acceptable variants.** Examples of
  sentences the LM changed from correct to "wrong": 台階 -> 臺階,
  計畫 -> 計劃, 反覆 -> 反復, 蛋捲 -> 蛋卷, 興奮的 -> 興奮地. Both forms are
  used in Taiwan, so CER slightly understates the system. Real LM mistakes
  are mostly rare idioms (兼容並蓄 -> 兼容並續).

## Caveats and Next Steps

- 5090 only. Repeat latency on the RTX 3060 and run the utilization
  benchmark (`resource-benchmark` skill) before claiming the resource budget.
- Possible pretraining contamination: much of the news text covers events
  from around 2024 and the Qwen models may have seen it. Evaluate on text newer than the model
  cutoff, and on PTT and Wikipedia as the proposal lists.
- g2pW readings are not manually checked yet (the proposal asks for about
  200 sentences).
- Candidate generation cost: the substitution enumeration is slow in the
  fuzzy engine (test p50 91 ms for `notone`, 403 ms for `initial`, measured
  with 16 parallel workers), while the Tab n-best alone is cheap. The online
  path should use the Tab n-best first.
- Not done yet: INT8 weight only, Qwen3-1.7B-Base, the scoring server and
  trigger policy, the gather only logits and shared prefix optimizations.

## Reproduce

Inside the `phonetic-candidate-dev` container, from the repo root:

```bash
python scripts/build_dataset.py                      # data/news/{dev,test}.jsonl
python scripts/gen_candidates.py                     # outputs/candidates/
python scripts/oracle_report.py --split test
python scripts/score_candidates.py --model Qwen/Qwen2.5-0.5B
python scripts/evaluate.py --tag Qwen2.5-0.5B-float16
python scripts/evaluate.py --tag Qwen2.5-0.5B-float16 --exclude-overlap
python scripts/bench_latency.py --model Qwen/Qwen2.5-0.5B --k 10
python -m pytest -q
```

# CPU Reranker: Small Character LM

Oct 9, 2026. Accuracy numbers are test CER over the five phase 1 domains
(see [results-phase1.md](results-phase1.md)); CPU latency was measured on the
dev host CPU, an **AMD Ryzen Threadripper PRO 3955WX** (Zen 2), not the
proposal's i5-14500; training used the RTX 5090.

## Summary

- A 0.5B LM cannot rerank on the CPU: 0.24 to 1.4 s per call even with int8.
- A **16M parameter character LM**, int8, with prefix sharing, reranks the
  decoder's 10-best in **18 ms (p50) / 34 ms (p95)** on one CPU thread, using
  20 ms of CPU time per call (about 3.4% of one core at 1.7 calls per second).
- It cuts the CPU-only CER from 3.36% to 2.48% (toned) and from 7.53% to
  5.77% (toneless), about two thirds of the gain the zh-TW Qwen 0.5B gets on
  the GPU from the same pool (2.03% / 5.08%). The pool itself is now the
  ceiling: its Oracle@10 is 1.00% / 3.48%.
- In the IME (typing 200 Common Voice sentences), CPU reranking with it gives
  82.5% / 75.5% sentences correct (toned / toneless) vs 80.0% / 72.0% for
  the decoder alone and 83.0% / 76.0% with the GPU Qwen reranker.
- CPU budget, estimated from per-call costs at the proposal's 100 characters
  per minute (about 1.7 reranks and 1.7 syllables per second): reranker 1.7 x
  20 ms = 3.4% of one core, decoder 1.7 x about 3 ms = 0.5%, together about
  4% of one core, which is 0.2% of the proposal's 20 thread i5-14500 against
  a 10% budget. The proposal's replayed typing load benchmark (NVML and
  psutil sampling) has not been run yet.

## Models

Pre-norm causal transformers over characters (`src/zhuyin_rescore/charlm.py`):
the 14,079 n-gram characters, clause punctuation and four special tokens
(14,125 tokens), tied embeddings, learned positions up to 128. Trained from
scratch on the same eval-filtered documents as the Qwen continued
pretraining (300M characters, 75M per source), one pass, sequence length
128, AdamW, bf16, 4 to 12 minutes each on the RTX 5090.

| model | layers x width | parameters | held-out perplexity per character |
| --- | --- | --- | --- |
| tiny | 4 x 256 | 6.8M | 35.1 |
| small | 6 x 384 | 16.1M | 26.9 |
| medium | 8 x 512 | 32.5M | 23.6 |

Context: the last 64 characters of committed text, normalized exactly as the
training text.

## Accuracy

"4-gram top 10" is the decoder's 10-best, the pool the IME reranks; "merged
@30" adds libchewing 0.14's candidates (needs libchewing 0.14 too). Fusion
weights `lm + a * ngram - beta * [not first] - mu * rank` are tuned on the dev
sets per condition and pool.

| system | runs on | full mean | full cv | notone mean | notone cv | initial mean |
| --- | --- | --- | --- | --- | --- | --- |
| libchewing 0.14 1-best | CPU | 4.88% | 3.01% | 16.55% | 12.37% | 68.52% |
| char 4-gram decoder 1-best | CPU | 3.36% | 2.51% | 7.53% | 5.60% | 58.52% |
| Oracle@10 of the decoder pool (upper bound of any reranker on it) | - | 1.00% | 0.41% | 3.48% | 1.71% | 44.65% |
| n-gram only fusion, merged @30 | CPU | 2.96% | 2.05% | 8.63% | 6.47% | 58.19% |
| char LM 6.8M, homophone softmax, 4-gram top 10 | CPU | 3.12% | 2.78% | 6.79% | 6.01% | 55.66% |
| char LM 6.8M, full softmax, 4-gram top 10 | CPU | 2.75% | 2.19% | 6.17% | 5.08% | 54.47% |
| char LM 6.8M, full softmax, merged @30 | CPU | 2.46% | 2.16% | 7.02% | 6.12% | 53.29% |
| char LM 16M, homophone softmax, 4-gram top 10 | CPU | 2.92% | 2.62% | 6.45% | 5.71% | 55.12% |
| char LM 16M, full softmax, 4-gram top 10 | CPU | 2.48% | 2.06% | 5.78% | 4.84% | 53.75% |
| char LM 16M int8, full softmax, 4-gram top 10 | CPU | 2.48% | 2.08% | 5.77% | 4.82% | 53.77% |
| char LM 16M, full softmax, merged @30 | CPU | 2.15% | 2.00% | 6.53% | 5.78% | 52.09% |
| char LM 32.5M, homophone softmax, 4-gram top 10 | CPU | 2.84% | 2.46% | 6.37% | 5.75% | 54.78% |
| char LM 32.5M, full softmax, 4-gram top 10 | CPU | 2.35% | 1.96% | 5.53% | 4.64% | 53.22% |
| char LM 32.5M int8, full softmax, 4-gram top 10 | CPU | 2.37% | 1.96% | 5.53% | 4.58% | 53.14% |
| char LM 32.5M, full softmax, merged @30 | CPU | 2.03% | 1.85% | 6.02% | 5.49% | 51.33% |
| Qwen2.5-0.5B base, merged @30 | GPU | 2.13% | 2.15% | 6.30% | 6.70% | 52.12% |
| Qwen2.5-0.5B zh-TW, 4-gram top 10 | GPU | 2.03% | 1.74% | 5.08% | 4.27% | 52.22% |
| Qwen2.5-0.5B zh-TW, merged @30 | GPU | 1.57% | 1.57% | 5.12% | 4.99% | 49.59% |

- **Full softmax beats the homophone softmax** for every size and condition
  (for example 2.48% vs 2.92% toned, 16M). Scoring P(characters) keeps
  information that the per position conditional P(character | reading) throws
  away, so the cheaper output layer is not worth it.
- **int8 costs no accuracy**: every condition stays within 0.06 points of fp32.
- On toneless input the merged pool is worse than the decoder's own pool,
  because libchewing's toneless candidates are weak; the IME uses the
  decoder pool.
- Bigger is better but flattening: 6.8M -> 16M gains 0.27 points (toned),
  16M -> 32.5M another 0.13.

## CPU latency

One rerank call for the decoder 10-best (about 95 candidate characters),
context cache resident, 1 thread, 300 dev clauses:

| model | scoring | p50 | p95 | CPU time per call |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B zh-TW | fp32 | 1400 ms | 2104 ms | 1416 ms |
| Qwen2.5-0.5B zh-TW | int8 | 909 ms | 1474 ms | 931 ms |
| char LM 6.8M | flat, full softmax | 24 ms | 40 ms | 25 ms |
| char LM 6.8M | trie, full softmax | 12 ms | 21 ms | 13 ms |
| char LM 16M | flat, full softmax | 51 ms | 82 ms | 52 ms |
| char LM 16M | trie, full softmax | 24 ms | 42 ms | 26 ms |
| char LM 16M | trie, full softmax, int8 | 18 ms | 34 ms | 20 ms |
| char LM 32.5M | flat, full softmax | 97 ms | 158 ms | 100 ms |
| char LM 32.5M | trie, full softmax | 46 ms | 80 ms | 50 ms |
| char LM 32.5M | trie, full softmax, int8 | 31 ms | 61 ms | 34 ms |

- **Prefix sharing (trie):** the 10 candidates share prefixes; computing each
  unique prefix once (about 40 of 95 tokens) with tree attention halves the
  cost and gives the same scores.
- **int8 dynamic quantization** of the Linear layers saves another 25 to 35%.
- The 16M int8 model meets the proposal's p95 < 50 ms on this CPU; the
  32.5M int8 model does not (61 ms) but may on a faster core.
- Two threads lower wall time by about 25% at higher total CPU time; one
  thread is the low impact setting.

## IME, CPU only

`scripts/bench_ime_keys.py`, 200 Common Voice dev sentences typed key by
key, reranker results awaited before Enter, GPU hidden for the CPU rows:

| server | fusion a, beta, mu | with tones | toneless | key latency p95 |
| --- | --- | --- | --- | --- |
| decoder only | - | 80.0% | 72.0% | 2.9 ms |
| + char LM 16M int8 (CPU) | 0.3, 1.0, 0.1 | 82.5% | 75.5% | 5.5 ms |
| + char LM 32.5M int8 (CPU) | 0.3, 1.0, 0.1 | 80.5% | 74.0% | 5.5 ms |
| + Qwen2.5-0.5B zh-TW, CUDA graph (GPU) | 0.0, 2.0, 0.05 | 83.0% | 76.0% | 5.5 ms |

The IME uses the same fusion as the evaluation (`lm + a * ngram - beta *
[not first] - mu * rank`) with the dev tuned weights for the decoder pool.
With n = 200 the binomial standard error is about 2.8 points, so the rows
other than "decoder only" are not distinguishable here; the 6,000 sentence
offline table above is the evidence for ordering the models (32.5M slightly
ahead of 16M). Latency does not explain the 32.5M row: its rerank call is
31 ms, far inside the 400 ms settle window.
Keystrokes never wait for the reranker (it runs in its own thread and
pushes its result).

Recommended CPU only server:

```bash
python -m zhuyin_ime.server --reranker outputs/charlm/small --reranker-type charlm --int8
```

(The fusion weights are now chosen per candidate pool type, see
[ime.md](ime.md#candidate-pool); the pool also grew since this
measurement, which moves the IME numbers to 82.5% / 77.0%.)

## Discriminative fine-tuning on candidate pools (no gain)

Can the 16M model learn to pick the correct sentence from the pool directly,
instead of scoring candidates as a plain LM? Short answer: not measurably.

**Data.** 60,002 training and 3,001 dev clauses (5 to 20 characters, about
15k per source), sampled from news, wiki, PTT and web documents that come
after the n-gram's 150M character cut in each training file, so neither the
n-gram nor the LMs saw them. g2pW readings, 80 characters of preceding
context, 15% of contexts blanked. Pools are composed exactly as evaluated and
as the IME builds them, with the evaluation's decoder weights: typed with
tones, the decoder's 10-best plus libchewing 0.14's n-best (at most 20);
toneless, the decoder's 30-best. The correct sentence is in the pool for
96.4% (toned) and 84.7% (toneless) of the training clauses; the decoder's
1-best is correct for 81.8% / 67.8%.

**Loss** (`src/zhuyin_rescore/listwise.py`, `scripts/train_rerank.py`). The
candidate score is the inference score: full softmax log P(candidate |
context), computed as one batched forward over [BOS, context, candidate]
rows (a test checks it equals `CharLMScorer.score_cached` within 1e-3), fused
with the 4-gram, first-position bonus and rank penalty. Per pool: expected
normalized error under the pool softmax (MWER), cross entropy on the correct
sentence when the pool has it, and an LM anchor (NLL per character of the
correct sentence). A learned per pool scale on the softmax absorbs pressure
to change the network's temperature. Fusion weights frozen at the preset.

**Dev runs** (dev CER, mean of toned and toneless, with the best fusion on
a grid as in `evaluate_pools.py`; NLL of the correct sentence in nats per
character):

| run | dev CER | NLL |
| --- | --- | --- |
| base char LM 16M | 3.53% | 3.631 |
| LM fine-tuning only on the training clauses (control, 0.15 epoch) | 3.53% | 3.535 |
| anchor 0.1, no scale, lr 1e-4, learned fusion (stopped at step 1000) | 3.74% (1) | 3.961 |
| anchor 1, scale, lr 3e-5, 2 epochs (kept) | 3.50% | 3.584 |
| same, lr 1e-4, learned fusion (stopped at step 1500) | 3.58% | 3.637 |

(1) with its learned fusion; the base scores 3.56% with the preset fusion.

The first configuration degraded the LM: with no free scale the pool loss
moves the network's own temperature, which raises its NLL and does not change
any argmax. The learned scale stayed below 1 (0.5 to 0.6 during the first
epoch, 0.8 at the end), i.e. the base model's margins between candidates are
too wide for the pool softmax, not too narrow.

**Test** (five domains, kept checkpoint vs base, same pools, fusion tuned on
the dev sets for each; 95% CI from a paired bootstrap over sentences):

| pool | base | tuned | change, 95% CI | fixed / broken sentences |
| --- | --- | --- | --- | --- |
| toned: decoder 10-best + libchewing n-best, at most 20 | 2.17% | 2.16% | -0.01 [-0.08, +0.05] | 71 / 55 |
| toneless: decoder 30-best | 5.69% | 5.69% | 0.00 [-0.08, +0.08] | 64 / 63 |

Per domain the changes go both ways (toned: web 1.96% -> 1.86% and wiki
2.34% -> 2.25%, Common Voice 1.97% -> 2.08%). The model is not deployed.

**Why there is little to gain.** On the dev pools, of the base model's
2.03% toned CER only 1.23% comes from pools that contain the correct
sentence; 0.79% comes from pools that do not. Toneless: 1.35% of 5.10%
is fixable by reranking, 3.75% is the pool. Among the wrong picks with the
correct sentence in the pool, the fused score prefers the wrong one by a
median of about 3 nats, and a sample of them is dominated by cases context
cannot decide: two accepted spellings of the same word (variant characters
for "what", "self-made", "need", "to order"), he vs she (same reading), and
names. What is left after that is small relative to the noise of 60k
training pools.

## Next steps

- Widen the IME pool (decoder k = 20, or merge libchewing 0.14's n-best;
  done, see [ime.md](ime.md#candidate-pool)):
  1.00% toned and 3.48% toneless CER is the Oracle@10 floor, sentences whose
  correct form is not in the decoder 10-best at all, which no reranker can
  fix. Toneless input is mostly pool limited.
- Discriminative fine-tuning on the pools: tried, no measurable gain (see
  above). The pool is the larger lever, especially toneless.
- Distillation from the zh-TW Qwen reranker.
- Measure on the i5-14500 and under the replayed typing load.

## Reproduce

```bash
python scripts/prepare_charlm_data.py
python scripts/train_charlm.py --out outputs/charlm/small --d-model 384 --layers 6 --heads 6 --d-ff 1536
python scripts/score_union.py --scorer charlm --model outputs/charlm/small --tag charlm-small \
    --dtype float32 --sources chewing-0.14 beam-o4 --chunk 64 --device cuda
python scripts/evaluate_pools.py --tag charlm-small            # full softmax
python scripts/evaluate_pools.py --tag charlm-small --lm-field lm_homo
OMP_NUM_THREADS=1 python scripts/bench_cpu_charlm.py --models outputs/charlm/small --int8 --variants full
CUDA_VISIBLE_DEVICES="" python scripts/bench_ime_keys.py --settle-ms 400 \
    --server-args "--reranker outputs/charlm/small --reranker-type charlm --int8"

# Discriminative fine-tuning (no gain, see above)
python scripts/build_rerank_train.py                     # -> data/rerank_train/{train,dev}.jsonl
python scripts/gen_beam.py --data-dirs data/rerank_train --splits dev train --conditions full \
    --w-prior 0.3 --w-edge -1 --w-mismatch -3
python scripts/gen_beam.py --data-dirs data/rerank_train --splits dev train --conditions notone \
    --w-prior 0 --w-edge -1 --w-mismatch -6
python scripts/gen_candidates.py --data-dir data/rerank_train --out-dir outputs/cand/chewing-0.14/rerank_train \
    --splits dev train --conditions full --lib outputs/libchewing-0.14/lib/libchewing.so.3 \
    --syspath outputs/libchewing-0.14/share/libchewing
python scripts/build_rerank_pools.py                     # -> data/rerank_train/pools.{train,dev}.jsonl
python scripts/train_rerank.py --out outputs/charlm/disc-a1-2ep --freeze-fusion --epochs 2
python scripts/score_union.py --scorer charlm --model outputs/charlm/disc-a1-2ep --tag charlm-disc-a1-2ep \
    --dtype float32 --sources chewing-0.14 beam-o4 --conditions full notone --chunk 64 --device cuda
python scripts/evaluate_pools.py --tag charlm-disc-a1-2ep --pools beam-o4 b4+c14tab --ks 10 20 30 \
    --conditions full notone --save-errors --out outputs/reports/rerank
```

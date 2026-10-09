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
  the GPU from the same pool (2.03% / 5.08%).
- In the IME (typing 200 Common Voice sentences), CPU reranking with it gives
  83.0% / 75.0% sentences correct (toned / toneless) vs 80.0% / 72.0% for
  the decoder alone and 83.5% / 76.0% with the GPU Qwen reranker.

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

| server | with tones | toneless | key latency p95 |
| --- | --- | --- | --- |
| decoder only | 80.0% | 72.0% | 2.9 ms |
| + char LM 16M int8 (CPU) | 83.0% | 75.0% | 5.4 ms |
| + char LM 32.5M int8 (CPU) | 81.0% | 73.5% | 5.5 ms |
| + Qwen2.5-0.5B zh-TW, CUDA graph (GPU) | 83.5% | 76.0% | 5.5 ms |

200 sentences give about +-2.5 points, so 16M vs 32.5M is within noise
here; the 6,000 sentence offline numbers above favor 32.5M slightly.
Keystrokes never wait for the reranker (it runs in its own thread and
pushes its result).

Recommended CPU only server:

```bash
python -m zhuyin_ime.server --reranker outputs/charlm/small --reranker-type charlm --int8 \
    --fusion-a 0.3 --fusion-beta 1.0
```

## Next steps

- Discriminative fine-tuning on decoder pools (listwise loss against the
  gold sentence) to close the remaining gap to the 0.5B reranker.
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
    --server-args "--reranker outputs/charlm/small --reranker-type charlm --int8 --fusion-a 0.3 --fusion-beta 1.0"
```

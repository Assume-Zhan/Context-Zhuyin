# Phase 1 Results: Lattice Decoder, zh-TW Reranker, libchewing 0.14 Baseline

Oct 8, 2026. All GPU numbers come from the dev host GPU, an **NVIDIA GeForce
RTX 5090**; none are RTX 3060 results.

## Summary

Test CER, mean over five domains (news, Taiwan web 2025, PTT, newest
zh-TW Wikipedia articles, Common Voice), relative to **libchewing 0.14**:

| system | full | notone | initial | runs on |
| --- | --- | --- | --- | --- |
| libchewing 0.13.1 1-best (unigram) | 10.09% | 29.84% | 81.55% | CPU |
| libchewing 0.14 1-best (word bigram), baseline | 4.88% | 16.55% | 68.52% | CPU |
| our char 4-gram lattice decoder, 1-best | 3.36% (-31%) | 7.53% (-55%) | 58.52% (-15%) | CPU, 2 to 6 ms per keystroke |
| + Qwen2.5-0.5B base reranking the merged pool | 2.13% (-56%) | 6.30% (-62%) | 52.12% (-24%) | GPU |
| + Qwen2.5-0.5B zh-TW reranking the merged pool | **1.57% (-68%)** | **5.12% (-69%)** | **49.59% (-28%)** | GPU, 7.5 ms per call with a CUDA graph |

- Continued pretraining on 300M tokens of zh-TW text cuts the reranked
  character errors by another 26% on toned input (95% CI 23% to 30%), and
  helps on every domain, including Common Voice, which no model trained on
  (full 2.15% -> 1.57%).
- The decoder's own gain on toned input depends on its training data
  covering the domain; on toneless input it is robust (see below).
- A 0.5B reranker is not usable on the CPU: 0.24 to 1.4 s per call on this
  host. The CPU-only path is the n-gram decoder today.


## What changed since phase 0

- **Baseline.** libchewing 0.14.0-alpha.4 (data v2026.10.7) replaces 0.13.1
  as the reference engine. 0.14 implements the libchewing architecture RFC: a
  word lattice decoded with an interpolated word bigram + unigram LM (trained
  on CC-100) and a 10-best k-best Viterbi. It is a pre-release, built from
  source next to 0.13.1. On news it already halves the 0.13.1 CER, so the
  phase 0 gains (measured against 0.13.1) no longer describe the state of the
  art.
- **Decoder (step 1).** A lattice beam search over libchewing dictionary
  phrases with their readings, scored by a character n-gram trained on
  Taiwan text, with a penalty for character windows that form a dictionary
  word read differently from the input. Pure Python and numpy on the CPU.
- **Reranker (step 2).** Qwen2.5-0.5B with continued pretraining on 300M
  tokens of zh-TW text, compared with the base model.
- **Data.** Four new evaluation domains next to the phase 0 news sets, all
  from sources newer than or disjoint from the training text.

## Data

| domain | source | license | dev / test clauses | context |
| --- | --- | --- | --- | --- |
| news | liswei/news-collection-zhtw shard 2 (about 2024), phase 0 sets | none stated | 2000 / 2000 | preceding 80 chars |
| web | Common Crawl 2025-33, Taiwan domains (jed351/Traditional-Chinese-Common-Crawl-Filtered), heldout pages | none stated | 1000 / 1000 | preceding 80 chars |
| ptt | yuhuanstudio/PTT-pretrain-zhtw, posts from 2022 to 2024, heldout | Apache-2.0 | 1000 / 1000 | preceding 80 chars |
| wiki | yuhuanstudio/wikipedia-zh-tw, 2026-08 dump, the 2% newest articles by page id | CC BY-SA 4.0 | 1000 / 1000 | preceding 80 chars |
| cv | OpenFormosa/common_voice_25_zh-TW validated sentence list | CC0 | 1000 / 1000 | none |

Clauses are pure Han spans of 5 to 20 characters, readings from g2pW.
Clauses repeated across documents (boilerplate) are dropped, and clause text
is unique across all domains. Dev and test are split by document.

Training text (n-gram and LM) comes from the train side of the same sources
(news shards 0, 1, 3, 4; Wikipedia minus the heldout articles; PTT; the
Taiwan web pages minus the heldout ones). Every training document that
contains any evaluation clause of 5 or more characters, or any Common Voice
sentence, is dropped (27,349 clauses; 12% of news documents, mostly pages
that repeat the same "related articles" link lists, and 2 to 4% elsewhere).
Common Voice is never trained on.

## Models

- **Char n-gram:** order 4, stupid backoff, 150M Han characters from each of
  news, wiki, PTT and web (600M total), 3- and 4-grams with count 1 pruned:
  14,079 characters, 5.1M bigrams, 26M trigrams, 47.5M 4-grams. Bigram and
  trigram decoders use the first 2 or 3 count tables of the same model.
- **Decoder weights** (lexicon prior, per edge cost, reading mismatch
  penalty) were grid searched per condition and per n-gram order on 300 news
  dev and 300 web dev clauses only, optimizing 1-best CER.
- **Reranker:** score = sum of token log-probs given `<|endoftext|>` plus up
  to 64 tokens of context, fused as
  `lm + a * ngram - beta * [not pool 1-best] - mu * rank`, with (a, beta, mu)
  grid searched on the dev sets of all five domains together, per condition
  and pool, then frozen for test.
- **Continued pretraining:** Qwen2.5-0.5B, full parameters, 300M tokens (75M
  each from news, wiki, PTT, web), sequence length 512, bf16 autocast, AdamW
  (lr 3e-5, warmup 200, cosine to 10%), one pass.

## Decoder: n-gram order and training data

Test 1-best CER, mean over the five domains (per domain numbers are in the
tables below). No neural LM is involved; everything runs on the CPU.

| decoder | full | notone | initial |
| --- | --- | --- | --- |
| libchewing 0.13.1 (unigram) | 10.09% | 29.84% | 81.55% |
| libchewing 0.14 (word bigram, CC-100) | 4.88% | 16.55% | 68.52% |
| ours, char 2-gram | 6.69% | 16.15% | 77.78% |
| ours, char 3-gram | 3.54% | 8.18% | 63.92% |
| ours, char 4-gram | 3.36% | 7.53% | 58.52% |
| ours, char 4-gram, n-gram trained on PTT + wiki only | 4.28% | 9.29% | 62.59% |

- A character bigram is not enough: it is worse than the libchewing 0.14
  word bigram on toned input. A character trigram already beats it, and the
  4-gram is best in every condition.
- **In-corpus advantage, measured.** The main n-gram saw other documents from
  the news and web sources. Trained on PTT and Wikipedia only, so news, web
  and Common Voice are unseen sources, the 4-gram decoder is *worse* than
  libchewing 0.14 on toned input there (news 4.32% vs 3.93%, web 5.19% vs
  4.03%, cv 3.30% vs 3.01%). With all four sources it beats 0.14 on cv
  (2.51%), which no model was trained on. For toned input the gain of the
  decoder alone therefore comes from training data coverage, not from the
  model form.
- For toneless input the gain is robust: even the PTT + wiki n-gram beats
  0.14 on the unseen sources by a wide margin (news 10.29% vs 15.57%, cv
  7.31% vs 12.37%).
- Initials only remains hard for every decoder (best 1-best about 58% CER).

### Decode latency

Whole clause decode on the dev host CPU (AMD Ryzen Threadripper PRO 3955WX,
not the proposal's i5-14500), pure Python and numpy, measured inside a pool
of 12 to 14 concurrent workers, test sets of all domains:

| decoder | full p50 / p95 | notone p50 / p95 | initial p50 / p95 |
| --- | --- | --- | --- |
| char 2-gram | 12 / 24 ms | 15 / 31 ms | 34 / 73 ms |
| char 3-gram | 14 / 29 ms | 19 / 38 ms | 51 / 109 ms |
| char 4-gram | 25 / 53 ms | 33 / 68 ms | 73 / 163 ms |
| libchewing 0.14, 1-best + Tab n-best + substitution lists | 6 / 22 ms | 15 / 49 ms | 50 / 141 ms |

The decoder has not been optimized: no native code, no incremental decoding
as syllables arrive, no caching across keystrokes.

## Results per domain

Test CER per domain. "Merged pool" is the libchewing 0.14 1-best and
n-best followed by the 4-gram decoder k-best and the libchewing
substitutions, cut at 30. Rerank fusion weights are tuned on the dev sets
of all domains together. "fixed / broken" counts test sentences that
libchewing 0.14 got wrong / right and the system gets right / wrong.

### full

| system | news | web | ptt | wiki | cv | mean | vs 0.14 | fixed / broken vs 0.14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libchewing 0.13.1 1-best (unigram) | 10.02% | 9.26% | 10.77% | 13.54% | 6.88% | 10.09% | +106.7% | 207 / 1553 |
| libchewing 0.14 1-best (word bigram), baseline | 3.93% | 4.03% | 6.20% | 7.24% | 3.01% | 4.88% | - | - |
| Ours: char 2-gram decoder 1-best | 5.69% | 6.63% | 7.76% | 8.19% | 5.16% | 6.69% | +36.9% | 421 / 904 |
| Ours: char 3-gram decoder 1-best | 2.64% | 3.62% | 4.59% | 4.03% | 2.81% | 3.54% | -27.5% | 844 / 360 |
| Ours: char 4-gram decoder 1-best | 2.41% | 3.55% | 4.64% | 3.69% | 2.51% | 3.36% | -31.2% | 891 / 335 |
| Ours: char 4-gram decoder 1-best, n-gram from PTT+wiki only | 4.32% | 5.19% | 4.51% | 4.07% | 3.30% | 4.28% | -12.4% | 754 / 520 |
| Oracle@10, libchewing 0.14 pool | 1.15% | 1.47% | 2.15% | 3.01% | 0.59% | 1.67% | -65.8% | - |
| Oracle@30, merged pool | 0.25% | 0.57% | 0.52% | 0.82% | 0.21% | 0.47% | -90.3% | - |
| Rerank Qwen2.5-0.5B base, 0.14 top 10 | 1.95% | 2.43% | 3.78% | 3.96% | 2.26% | 2.88% | -41.1% | 751 / 67 |
| Rerank Qwen2.5-0.5B base, 4-gram top 10 | 1.64% | 2.41% | 3.71% | 2.34% | 2.37% | 2.49% | -49.0% | 1064 / 209 |
| Rerank Qwen2.5-0.5B base, merged pool @30 | 1.43% | 1.79% | 3.17% | 2.10% | 2.15% | 2.13% | -56.4% | 983 / 91 |
| Rerank Qwen2.5-0.5B base, merged pool @30, no context | 2.08% | 2.65% | 4.08% | 3.11% | 2.03% | 2.79% | -42.9% | 815 / 131 |
| Rerank Qwen2.5-0.5B zh-TW, 0.14 top 10 | 1.72% | 2.13% | 3.28% | 3.77% | 1.73% | 2.52% | -48.3% | 867 / 61 |
| Rerank Qwen2.5-0.5B zh-TW, 4-gram top 10 | 1.28% | 2.11% | 2.93% | 2.07% | 1.74% | 2.03% | -58.5% | 1161 / 134 |
| Rerank Qwen2.5-0.5B zh-TW, merged pool @30 | 0.96% | 1.45% | 2.11% | 1.77% | 1.57% | 1.57% | -67.8% | 1165 / 72 |
| Rerank Qwen2.5-0.5B zh-TW, merged pool @30, no context | 1.66% | 2.07% | 3.04% | 2.62% | 1.56% | 2.19% | -55.1% | 1013 / 151 |

### notone

| system | news | web | ptt | wiki | cv | mean | vs 0.14 | fixed / broken vs 0.14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libchewing 0.13.1 1-best (unigram) | 29.60% | 28.19% | 32.01% | 33.80% | 25.60% | 29.84% | +80.2% | 141 / 1505 |
| libchewing 0.14 1-best (word bigram), baseline | 15.57% | 14.21% | 19.38% | 21.25% | 12.37% | 16.55% | - | - |
| Ours: char 2-gram decoder 1-best | 15.45% | 15.76% | 18.36% | 18.68% | 12.50% | 16.15% | -2.4% | 657 / 741 |
| Ours: char 3-gram decoder 1-best | 7.04% | 7.94% | 9.89% | 9.65% | 6.40% | 8.18% | -50.6% | 1616 / 250 |
| Ours: char 4-gram decoder 1-best | 6.24% | 7.55% | 9.41% | 8.85% | 5.60% | 7.53% | -54.5% | 1761 / 213 |
| Ours: char 4-gram decoder 1-best, n-gram from PTT+wiki only | 10.29% | 10.61% | 9.24% | 9.01% | 7.31% | 9.29% | -43.9% | 1451 / 331 |
| Oracle@10, libchewing 0.14 pool | 9.96% | 9.30% | 12.77% | 15.49% | 6.65% | 10.84% | -34.5% | - |
| Oracle@30, merged pool | 2.18% | 3.07% | 3.48% | 4.28% | 1.32% | 2.86% | -82.7% | - |
| Rerank Qwen2.5-0.5B base, 0.14 top 10 | 11.87% | 10.67% | 15.98% | 17.56% | 10.68% | 13.35% | -19.4% | 662 / 97 |
| Rerank Qwen2.5-0.5B base, 4-gram top 10 | 4.17% | 5.17% | 7.24% | 6.31% | 5.51% | 5.68% | -65.7% | 2138 / 140 |
| Rerank Qwen2.5-0.5B base, merged pool @30 | 4.62% | 5.28% | 8.46% | 6.43% | 6.70% | 6.30% | -62.0% | 2016 / 143 |
| Rerank Qwen2.5-0.5B base, merged pool @30, no context | 6.85% | 7.32% | 10.30% | 8.92% | 6.05% | 7.89% | -52.4% | 1650 / 176 |
| Rerank Qwen2.5-0.5B zh-TW, 0.14 top 10 | 11.42% | 10.48% | 15.07% | 17.23% | 9.58% | 12.76% | -22.9% | 754 / 58 |
| Rerank Qwen2.5-0.5B zh-TW, 4-gram top 10 | 3.79% | 4.93% | 6.25% | 6.18% | 4.27% | 5.08% | -69.3% | 2254 / 102 |
| Rerank Qwen2.5-0.5B zh-TW, merged pool @30 | 3.63% | 4.71% | 6.29% | 6.00% | 4.99% | 5.12% | -69.0% | 2220 / 55 |
| Rerank Qwen2.5-0.5B zh-TW, merged pool @30, no context | 5.62% | 6.25% | 8.60% | 7.96% | 4.89% | 6.66% | -59.7% | 1903 / 147 |

### initial

| system | news | web | ptt | wiki | cv | mean | vs 0.14 | fixed / broken vs 0.14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libchewing 0.13.1 1-best (unigram) | 82.33% | 82.13% | 83.78% | 83.68% | 75.85% | 81.55% | +19.0% | 9 / 223 |
| libchewing 0.14 1-best (word bigram), baseline | 69.13% | 65.78% | 72.03% | 72.29% | 63.36% | 68.52% | - | - |
| Ours: char 2-gram decoder 1-best | 78.58% | 77.74% | 78.00% | 80.86% | 73.71% | 77.78% | +13.5% | 19 / 214 |
| Ours: char 3-gram decoder 1-best | 64.85% | 62.89% | 67.50% | 64.93% | 59.45% | 63.92% | -6.7% | 275 / 90 |
| Ours: char 4-gram decoder 1-best | 59.81% | 57.49% | 62.38% | 59.15% | 53.74% | 58.52% | -14.6% | 552 / 59 |
| Ours: char 4-gram decoder 1-best, n-gram from PTT+wiki only | 67.57% | 66.83% | 61.39% | 59.64% | 57.51% | 62.59% | -8.7% | 403 / 91 |
| Oracle@10, libchewing 0.14 pool | 60.20% | 57.28% | 61.88% | 63.78% | 51.07% | 58.84% | -14.1% | - |
| Oracle@30, merged pool | 40.03% | 39.50% | 41.50% | 43.05% | 30.82% | 38.98% | -43.1% | - |
| Rerank Qwen2.5-0.5B base, 0.14 top 10 | 66.67% | 63.01% | 69.90% | 69.14% | 62.85% | 66.31% | -3.2% | 84 / 22 |
| Rerank Qwen2.5-0.5B base, 4-gram top 10 | 53.24% | 51.68% | 57.85% | 53.25% | 52.57% | 53.72% | -21.6% | 743 / 41 |
| Rerank Qwen2.5-0.5B base, merged pool @30 | 50.85% | 49.47% | 56.48% | 51.49% | 52.34% | 52.12% | -23.9% | 787 / 29 |
| Rerank Qwen2.5-0.5B base, merged pool @30, no context | 58.60% | 56.40% | 63.67% | 57.98% | 51.74% | 57.68% | -15.8% | 497 / 31 |
| Rerank Qwen2.5-0.5B zh-TW, 0.14 top 10 | 66.35% | 63.20% | 68.93% | 68.87% | 61.26% | 65.72% | -4.1% | 92 / 12 |
| Rerank Qwen2.5-0.5B zh-TW, 4-gram top 10 | 52.10% | 50.65% | 55.00% | 53.10% | 50.25% | 52.22% | -23.8% | 838 / 32 |
| Rerank Qwen2.5-0.5B zh-TW, merged pool @30 | 49.11% | 47.53% | 51.73% | 50.26% | 49.33% | 49.59% | -27.6% | 912 / 21 |
| Rerank Qwen2.5-0.5B zh-TW, merged pool @30, no context | 56.47% | 54.53% | 59.46% | 57.50% | 49.00% | 55.39% | -19.2% | 678 / 29 |

### Fine-tuned vs base reranker on the same test sentences

| condition | system | sentences fixed by fine-tuning | broken by fine-tuning | char errors base -> zh-TW | 95% CI of relative change |
| --- | --- | --- | --- | --- | --- |
| full | LM rerank c14+beam4@30 | 264 | 63 | 1163 -> 855 | -30.1% to -22.8% |
| full | LM rerank beam-o4@10 | 239 | 67 | 1360 -> 1108 | -22.0% to -15.1% |
| full | LM rerank chewing-0.14@10 | 174 | 52 | 1600 -> 1412 | -14.1% to -9.4% |
| notone | LM rerank c14+beam4@30 | 387 | 95 | 3486 -> 2847 | -20.7% to -16.0% |
| notone | LM rerank beam-o4@10 | 237 | 83 | 3164 -> 2859 | -11.7% to -7.5% |
| notone | LM rerank chewing-0.14@10 | 169 | 38 | 7714 -> 7396 | -5.0% to -3.3% |
| initial | LM rerank c14+beam4@30 | 180 | 47 | 30346 -> 28991 | -5.1% to -3.8% |
| initial | LM rerank beam-o4@10 | 132 | 28 | 31391 -> 30593 | -3.1% to -2.0% |
| initial | LM rerank chewing-0.14@10 | 28 | 10 | 38925 -> 38634 | -1.0% to -0.5% |

The relative change is in total character errors, 95% CI by a paired
bootstrap over the 6,000 test sentences.

### Reranker latency (RTX 5090 only, not the RTX 3060 target)

One scoring call for the k-best of a clause, context KV cache resident:

| reranker | k | p50 | p95 | context cache (once per commit) |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B zh-TW, eager | 10 | 20.4 ms | 21.1 ms | 16.3 ms |
| Qwen2.5-0.5B zh-TW, CUDA graph | 10 | 7.5 ms | 7.6 ms | 16.6 ms |
| Qwen2.5-0.5B zh-TW, eager | 30 | 22.0 ms | 23.2 ms | 16.3 ms |
| Qwen2.5-1.5B, eager | 10 | 23.7 ms | 24.7 ms | 18.9 ms |
| Qwen2.5-1.5B, CUDA graph | 10 | 12.7 ms | 13.0 ms | 19.8 ms |

The eager forward is launch bound; replaying it as one CUDA graph
(`src/zhuyin_rescore/graph_scorer.py`, same best candidate as the eager
scorer on 199 of 199 dev clauses) cuts the call to about a third and also
removes most of the CPU time spent launching kernels.

CPU only (AMD Ryzen Threadripper PRO 3955WX), Qwen2.5-0.5B zh-TW, k=10:

| variant | 1 thread | 2 threads | 4 threads | 8 threads | same best as fp32 |
| --- | --- | --- | --- | --- | --- |
| fp32 | 1400 ms | 944 ms | 552 ms | 360 ms | 100% |
| int8 dynamic quantization | 909 ms | 529 ms | 324 ms | 236 ms | 97.5% |

A call costs about 120 GFLOP (about 120 candidate tokens through 24 layers
and a 151k vocabulary output layer) and about 1 s of CPU time, far over the
10% CPU budget at 1.7 calls per second. A CPU reranker needs a much smaller
model.

### IME keystroke latency

Typing 200 Common Voice dev sentences through the conversion server
(`scripts/bench_ime_keys.py`, decoder only, CPU): p50 0.07 ms, p95 3.0 ms,
max 5.5 ms per key with tones; p95 3.6 ms without tones. Sentence accuracy
of the decoder alone in the IME: 79.5% with tones, 69.0% without.

## Caveats

- 5090 and Threadripper numbers only; the i5-14500 + RTX 3060 target and
  the utilization under load benchmark are still to be measured.
- libchewing 0.14 is a pre-release (v0.14.0-alpha.4, data v2026.10.7)
  built from source; its bigram LM is trained on CC-100, so Taiwan specific
  text is partly out of its distribution too.
- Decoder weights were tuned on news and web dev only; fusion weights on
  all dev sets. The no-context ablation's n-gram weight sits at the grid
  edge (1.5), and its n-gram feature still sees the tail of the context when
  the context ends in Han characters (usually it ends in punctuation).
- The merged pool reuses libchewing 0.14's substitution lists, so the full
  system currently needs both libchewing 0.14 and our decoder.
- Sources without a stated license (the news collection and the Common
  Crawl derivative) are used for research only.
- Possible pretraining contamination of the Qwen base model on news, web
  and Wikipedia; Common Voice and the newest Wikipedia articles are the
  least exposed and show the same direction.

## Reproduce

```bash
python scripts/fetch_corpora.py && python scripts/build_corpus.py
python scripts/build_eval_sets.py && python scripts/build_train_text.py
python scripts/train_ngram.py
python scripts/gen_candidates.py --data-dir data/eval/cv --out-dir outputs/cand/chewing-0.14/cv \
    --lib outputs/libchewing-0.14/lib/libchewing.so.3 --syspath outputs/libchewing-0.14/share/libchewing
python scripts/gen_beam.py --orders 4 --conditions full --w-prior 0.3 --w-edge -1 --w-mismatch -3
python scripts/prepare_lm_data.py && python scripts/train_lm.py
python scripts/score_union.py --model outputs/lm/qwen2.5-0.5b-zhtw --tag qwen2.5-0.5b-zhtw-float16 \
    --sources chewing-0.14 beam-o4 --chunk 48
python scripts/evaluate_pools.py --tag qwen2.5-0.5b-zhtw-float16 --save-errors
python scripts/bench_latency.py --model outputs/lm/qwen2.5-0.5b-zhtw --graph \
    --candidates outputs/cand/chewing-0.14/news/dev.full.jsonl
```

(gen_candidates and gen_beam run per domain and condition; see the script
help for all options.)

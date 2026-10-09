# Zhuyin LM Input Method (Linux, IBus)

Status: first working version. Typing, conversion, candidate selection,
commit and asynchronous LM reranking work end to end through IBus
(`tests/test_ibus_e2e.py` drives a real `ibus-daemon` headless).

## Architecture

```
host desktop session                      dev container (or any machine with the models)
+------------------------------+          +-------------------------------------------+
| application (GTK, Qt, ...)   |          | python -m zhuyin_ime.server               |
|        |  IBus protocol      |          |   Session per input context:              |
| ibus-daemon                  |          |     composer (Dai Chien bopomofo)         |
|        |                     |  JSON    |     incremental lattice beam decoder      |
| zhuyin_ime.ibus_engine       |<-------->|       (libchewing dictionary + char       |
|   thin client, stdlib + gi   |  lines   |        4-gram, CPU)                       |
+------------------------------+  over a  |   Reranker thread (optional): Qwen2.5     |
                                 Unix     |     0.5B zh-TW, debounced, pushes a       |
                                 socket   |     better preedit when it finds one      |
                                          +-------------------------------------------+
```

- The IBus engine must run in the desktop session (where `ibus-daemon`
  runs), so it is kept tiny: it forwards key events and draws the preedit and
  the candidate window. All conversion state lives in the server.
- They meet at `outputs/run/zhuyin-ime.sock`, which the bind mount makes
  visible on the host at the same path in the repo.
- If the server is not running, the engine lets every key through.
- libchewing is used for its dictionary (readings and phrase frequencies),
  read through `chewing-cli dump`; no libchewing source code is copied, so
  the project stays Apache-2.0 (calling or loading an LGPL library is fine).

## Running

In the dev container (from the repo root):

```bash
python -m zhuyin_ime.server                                                 # decoder only, CPU
python -m zhuyin_ime.server --reranker outputs/charlm/small --reranker-type charlm --int8  # CPU reranker
python -m zhuyin_ime.server --reranker outputs/lm/qwen2.5-0.5b-zhtw --graph  # GPU reranker
```

- `--graph` replays the Qwen reranker as a CUDA graph.
- Fusion weights come from dev tuning, per reranker and pool type
  (`--fusion-preset`, default follows `--reranker-type`).
- libchewing 0.14 adds its n-best to the pool for input typed with tones.
  The dev image sets `CHEWING14_LIB` and `CHEWING14_PATH`, which the server
  uses by default; in a container built before that stage, pass
  `--chewing-lib outputs/libchewing-0.14/lib/libchewing.so.3
  --chewing-syspath outputs/libchewing-0.14/share/libchewing`.
- `--pool-k 10` keeps the CPU reranker under 50 ms p95 on this host at a
  small cost in toneless accuracy (see below).

## Candidate pool

The reranker does not only reorder the decoder's 1-best alternatives; it
picks from a pool that depends on how the user types:

- **typed with tones** (at least half of the syllables have a tone mark):
  the decoder's 10-best followed by libchewing 0.14's whole sentence n-best
  (its word bigram decoder makes different mistakes). libchewing gets the
  same keys, where Space is tone 1;
- **otherwise**: the decoder's 30-best (libchewing's toneless n-best is weak
  and hurts here).

Offline test CER over the five domains for the reranked pools:

| pool | Qwen zh-TW, tones | Qwen zh-TW, toneless | char LM 16M, tones | char LM 16M, toneless |
| --- | --- | --- | --- | --- |
| decoder 10-best | 2.03% | 5.08% | 2.48% | 5.78% |
| decoder 30-best | 1.94% | **4.84%** | 2.48% | **5.69%** |
| decoder 10-best + libchewing n-best | **1.64%** | 5.07% | **2.17%** | 6.15% |

## Measurements

Typing 200 Common Voice dev sentences through the server
(`scripts/bench_ime_keys.py --settle-ms 400`, reranker results awaited
before Enter, libchewing 0.14 n-best in the toned pool), default settings
(compact n-gram, ONNX Runtime for the character LM):

| server | sentences correct, with tones | toneless | rerank call p50 / p95 (tones; toneless) | key latency p95 (tones; toneless) | resident memory (anonymous) | GPU memory |
| --- | --- | --- | --- | --- | --- | --- |
| decoder only (CPU) | 80.0% | 72.0% | - | 4.9; 5.8 ms | 0.82 GB (0.12) | - |
| + char LM 16M, ONNX Runtime int8 (CPU, 1 thread) | 82.5% | 76.5% | 36 / 66 ms; 42 / 65 ms | 5.0; 5.7 ms | 0.94 GB (0.22) | - |
| + char LM 16M, ONNX Runtime fp32 (no `--int8`) | 82.5% | 76.5% | 38 / 67 ms; 43 / 69 ms | 4.7; 5.8 ms | 1.05 GB (0.33) | - |
| + char LM 16M, torch int8 (`--charlm-backend torch`) | 82.5% | 77.0% | 41 / 72 ms; 47 / 73 ms | 5.0; 5.9 ms | 1.45 GB (0.52) | - |
| + Qwen2.5-0.5B zh-TW, CUDA graph (RTX 5090) | 85.0% | 78.0% | 19 / 33 ms; 15 / 17 ms | 5.3; 6.4 ms | 2.62 GB (1.44) | 1.92 GB |

With `--pool-k 10` the ONNX int8 reranker takes 25 / 44 ms toneless (76.5%
correct). 200 sentences: standard error about 2.8 points. CPU numbers are from the
dev host's Threadripper 3955WX; GPU memory is the server process's own
(the GPU was shared with an unrelated job during these runs). The reranker
runs in its own thread and pushes its result about 100 ms (debounce) after
the last syllable, so keystrokes never wait for it; after each commit it
builds the context cache for the new text in the background. Enter commits
and also ends the line in the context history, so the next sentence starts
fresh.

## Memory

Same benchmark, before and after the memory work (Oct 9, 2026):

| server | resident memory | anonymous | GPU memory | rerank p50 / p95 (tones; toneless) | sentences correct |
| --- | --- | --- | --- | --- | --- |
| decoder only | 1.05 -> 0.82 GB | 171 -> 122 MB | - | - | 80.0 / 72.0 -> 80.0 / 72.0 |
| + char LM 16M int8 | 1.71 -> 0.94 GB | 599 -> 222 MB | - | 43 / 73; 49 / 74 -> 36 / 66; 42 / 65 ms | 82.5 / 77.0 -> 82.5 / 76.5 |
| + Qwen2.5-0.5B, CUDA graph | 3.48 -> 2.62 GB | 2127 -> 1443 MB | 2.93 -> 1.92 GB (1) | 21 / 37; 17 / 23 -> 19 / 33; 15 / 17 ms | 85.0 / 78.0 -> 85.0 / 78.0 |

(1) Measured with the scorer alone (`LMScorer` plus `GraphScorer`, 50
calls), before and after; the server number after is 1.92 GB.

What changed:

- **Compact n-gram** (`scripts/compact_ngram.py`, the server's default
  `outputs/ngram/zhtw-o4-q16`): the stupid backoff score of every n-gram is
  precomputed and stored as a 16 bit code into a per order codebook instead
  of two int32 counts, and bigram keys are uint32. 901 -> 732 MiB, one
  lookup per order instead of two, so decoding is also faster (dev decode
  p50 24.6 -> 20.9 ms toned). Decoding is unchanged: dev 1-best CER 3.04% /
  7.04% either way, the 1-best is identical for 99.95% / 99.80% of the 6,000
  dev clauses. 8 bit codes (656 MiB) cost +0.02 / +0.05 CER points;
  dropping 3- and 4-grams seen fewer than 3 times (395 MiB) costs +0.13 /
  +0.36, so neither is the default.
- **Character LM on ONNX Runtime** (`src/zhuyin_rescore/charlm_ort.py`,
  `scripts/export_charlm_onnx.py`): the same tree attention scoring as two
  exported graphs (context cache; candidates with the output projection and
  log softmax), int8 dynamic quantization, one thread, spinning disabled.
  The server no longer imports torch (about 450 MB), and the calls are
  faster: on the decoder 10-best of news dev, p50 / p95 13.4 / 29.5 ms vs
  17.6 / 32.8 ms for torch int8; its pick agrees with the fp32 model's on
  99.3% of pools (torch int8: 98.0%). Test CER on the five domains against
  the fp32 model: with tones (decoder 10-best + libchewing n-best, at most
  20) 2.17% -> 2.16%, toneless (decoder 30-best) 5.69% -> 5.76% (95% CI of
  the change +0.01 to +0.13 points). Without `--int8` the ONNX scores are
  the fp32 model's, at about the same speed in the IME and 0.1 GB more
  memory (table above).
- **Qwen reranker**: the CUDA graph computed the log softmax over all 32 x
  24 candidate positions at once (152k vocabulary, fp32, about 1 GB of
  buffers); it now does 4 candidates at a time (graph and scores unchanged).
  The weights load straight onto the GPU, and the server returns freed heap
  memory to the OS after loading (glibc kept the 0.7 GB of a CPU side copy
  resident).
- **Lexicon**: syllable and phrase strings are interned (-38 MB), and the
  span lookup cache is capped at 20k entries instead of 200k.

What did not help: `madvise(MADV_RANDOM)` on the n-gram maps (typing touches
nearly every page anyway); a numpy forward of the character LM (no torch,
but fp32 BLAS is 25% slower than int8); splitting the 4-gram keys into
uint32 halves (another -170 MB, but lookups in numpy were 3 to 6 times
slower).

What is left: the memory mapped n-gram (about 0.7 GB, page cache the kernel
can reclaim, at the cost of slower lookups when it faults back in); for
the GPU reranker, torch itself (about 0.5 GB), CUDA kernels loaded into host
memory on first use (about 0.33 GB), the CUDA context (0.58 GB of GPU
memory) and the fp16 weights (0.99 GB).

The server lowers its own priority (`--nice 5`), uses one CPU thread for
the reranker, and sleeps in `recv` when idle.

On the host (needs `ibus`, `python3-gi`, `gir1.2-ibus-1.0`):

```bash
scripts/ibus/install.sh            # writes ~/.local/bin/zhuyin-ime-ibus
~/.local/bin/zhuyin-ime-ibus &     # registers the engine on the running bus
ibus engine zhuyin-lm              # switch to it (back: ibus engine xkb:us::eng)
```

`scripts/ibus/install.sh --system` additionally installs
`/usr/share/ibus/component/zhuyin-lm.xml` (sudo) so the engine appears in the
desktop input source settings and ibus-daemon starts it on demand.

## Keys (Dai Chien layout)

| key | action |
| --- | --- |
| bopomofo keys | compose a syllable; a symbol of a filled slot replaces it |
| `6` `3` `4` `7` | finish the syllable with tone 2, 3, 4, neutral |
| Space | finish the syllable without a tone (matches any tone); with nothing being composed, open the candidate window |
| Down | open the candidate window at the cursor (at the end: phrases ending there) |
| `1` to `9` | pick a candidate; the choice is locked for later conversions |
| Space, Down / Up, Page keys | next / previous candidate page |
| Enter | commit the preedit |
| BackSpace / Delete | delete a symbol or a syllable |
| Left / Right / Home / End | move the cursor between syllables |
| Esc | close the candidate window, drop the syllable being composed, or clear |
| Shift + `,` `.` `/` `1` `;` and `[` `]` `'` `\` | full width punctuation, commits first |

Committed text becomes the context for the next conversion (the n-gram sees
its tail, the reranker the last 64 tokens).

## Limitations (next steps)

- Only the Dai Chien layout; no user dictionary or learning yet.
- No initials only (abbreviated) input mode.
- Typing inserts at the end; the cursor is for candidate selection only.
- The reranker pool is the decoder's 10-best; libchewing 0.14's n-best is
  not merged in yet.
- Fcitx5 front end not written; the server protocol is framework independent.

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
(`scripts/bench_ime_keys.py`, reranker results awaited before Enter):

| server | sentences correct, with tones | toneless | rerank call p50 / p95 (tones; toneless) | key latency p95 |
| --- | --- | --- | --- | --- |
| decoder only (CPU) | 80.0% | 72.0% | - | 2.9 ms |
| + char LM 16M int8 (CPU, 1 thread) | 82.5% | 77.0% | 42 / 74 ms; 49 / 73 ms | 5.8 ms |
| + char LM 16M int8, `--pool-k 10` | 82.5% | 75.0% | 42 / 74 ms; 32 / 49 ms | 6.9 ms |
| + Qwen2.5-0.5B zh-TW, CUDA graph (RTX 5090) | 85.0% | 78.0% | 20 / 36 ms; 16 / 18 ms | 6.5 ms |

200 sentences: standard error about 2.8 points. CPU numbers are from the
dev host's Threadripper 3955WX. The reranker runs in its own thread and
pushes its result about 100 ms (debounce) after the last syllable, so
keystrokes never wait for it; after each commit it builds the context cache
for the new text in the background. Enter commits and also ends the line in
the context history, so the next sentence starts fresh.

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

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
python -m zhuyin_ime.server --reranker outputs/lm/qwen2.5-0.5b-zhtw --graph  # plus GPU reranker
```

`--graph` replays the reranker as a CUDA graph (7.5 ms per call instead of
20 ms on the RTX 5090). The default fusion weights (`--fusion-a 0 --fusion-beta 2
--fusion-mu 0.05`) are the dev tuned ones for the zh-TW Qwen model on the
decoder 10-best.

CPU only (no GPU), with the 16M character LM reranker
([results-cpu-reranker.md](results-cpu-reranker.md)):

```bash
python -m zhuyin_ime.server --reranker outputs/charlm/small --reranker-type charlm --int8 \
    --fusion-a 0.3 --fusion-beta 1.0 --fusion-mu 0.1
```

Measured by typing 200 Common Voice dev sentences through the server
(`scripts/bench_ime_keys.py`):

| server | key latency p95 | sentences correct, with tones | toneless |
| --- | --- | --- | --- |
| decoder only (CPU) | 2.9 ms | 80.0% | 72.0% |
| + char LM 16M int8 reranker (CPU, 1 thread) | 5.5 ms | 82.5% | 75.5% |
| + Qwen2.5-0.5B zh-TW reranker, CUDA graph (RTX 5090) | 5.5 ms | 83.0% | 76.0% |

200 sentences: standard error about 2.8 points.

The reranker runs in its own thread and pushes its result about 100 ms
(debounce) after the last syllable, so keystrokes never wait for it. Enter
commits and also ends the line in the context history, so the next sentence
starts fresh.

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

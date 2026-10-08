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
python -m zhuyin_ime.server                                         # decoder only, CPU
python -m zhuyin_ime.server --reranker outputs/lm/qwen2.5-0.5b-zhtw  # plus GPU reranker
```

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

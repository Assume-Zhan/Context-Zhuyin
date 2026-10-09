# Context Zhuyin

Zhuyin (bopomofo) input with context aware candidate selection. A lattice
decoder over the libchewing dictionary, scored by a character n-gram trained
on Taiwan text, proposes candidate sentences on the CPU; a small language
model reranks them using the text you already committed. It ships as a Linux
input method (IBus).

On five Taiwan test domains it cuts the character error rate of libchewing
0.14 from 4.88% to 1.57% (typed with tones) and from 16.55% to 5.12%
(without tones) with a GPU reranker, and to 2.48% / 5.77% on the CPU alone.
See [docs/results-phase1.md](docs/results-phase1.md) and
[docs/results-cpu-reranker.md](docs/results-cpu-reranker.md).

## Install the input method on Linux

The input method has two parts that talk over a Unix socket in the repo
(`outputs/run/zhuyin-ime.sock`):

- the **conversion server** (`python -m zhuyin_ime.server`): decoder,
  dictionary and optional reranker; runs in the dev container (GPU or CPU)
  or directly on the host (CPU);
- the **IBus front end** (`zhuyin_ime.ibus_engine`): a thin engine in your
  desktop session that forwards keys and draws the preedit and candidates.
  It needs only the system Python with PyGObject. If the server is not
  running, it lets every key through.

Details of the design and the key bindings: [docs/ime.md](docs/ime.md).

### Requirements

- Linux desktop with IBus (GNOME uses it by default). On Debian / Ubuntu:
  `sudo apt install ibus python3-gi gir1.2-ibus-1.0`
- For the dev container: Docker with the compose plugin; for the GPU
  reranker also an NVIDIA GPU and `nvidia-container-toolkit`.
- About 25 GB of disk (plus the Docker image) to rebuild the models from
  scratch; about 2 GB if `outputs/` already holds them.

### 1. Clone and start the dev container

```bash
git clone <this repo> context-zhuyin && cd context-zhuyin
bash docker/host_setup.sh                                   # writes docker/.env with your UID/GID
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d phonetic-candidate-dev
```

The repo is bind mounted at `/workspace` in the container, so files the
container writes under `outputs/` (models, the socket) appear in your clone.
Steps 2 and 3 run inside the container; open a shell there (it starts in
`/workspace`):

```bash
docker compose -f docker/docker-compose.yml exec phonetic-candidate-dev bash
```

### 2. Prepare the models (once)

Skip this step if `outputs/ngram/zhtw-o4-q16` (and the rerankers you want)
already exist.

```bash
# Data: Taiwan news, PTT, zh-TW Wikipedia, 2025 Taiwan web pages, Common Voice
python scripts/build_dataset.py          # news dev/test sets (used to dedupe training text)
python scripts/fetch_corpora.py          # about 6 GB of downloads
python scripts/build_corpus.py
python scripts/build_eval_sets.py        # g2pW readings, about 10 minutes
python scripts/build_train_text.py       # drops training documents that contain test sentences

# Required: the character 4-gram used by the decoder (about 3 minutes, CPU)
python scripts/train_ngram.py            # -> outputs/ngram/zhtw-o4
python scripts/compact_ngram.py --out outputs/ngram/zhtw-o4-q16   # smaller form the server loads

# Optional: CPU reranker, 16M character LM (about 6 minutes on a GPU)
python scripts/prepare_charlm_data.py
python scripts/train_charlm.py --out outputs/charlm/small --d-model 384 --layers 6 --heads 6 --d-ff 1536
python scripts/export_charlm_onnx.py outputs/charlm/small   # ONNX Runtime graphs for the server

# Optional: GPU reranker, Qwen2.5-0.5B with zh-TW continued pretraining
# (about 3 hours on an RTX 5090)
python scripts/prepare_lm_data.py
python scripts/train_lm.py               # -> outputs/lm/qwen2.5-0.5b-zhtw
```

The libchewing dictionary is dumped to `outputs/dict/` the first time the
server starts in the container.

### 3. Start the conversion server

In the container, pick one. The server runs in the foreground and logs to
the terminal; keep that shell open while you use the input method (append
`&` to get the prompt back):

```bash
# Decoder only (CPU, needs no reranker model)
python -m zhuyin_ime.server

# CPU reranker: 16M character LM on ONNX Runtime, int8, one thread
# (without --int8: exact fp32 scores, same speed, 0.1 GB more memory)
python -m zhuyin_ime.server --reranker outputs/charlm/small --reranker-type charlm --int8

# GPU reranker: Qwen2.5-0.5B zh-TW, replayed as a CUDA graph
python -m zhuyin_ime.server --reranker outputs/lm/qwen2.5-0.5b-zhtw --graph
```

With a reranker, input typed with tones also gets libchewing 0.14's n-best
in the candidate pool. The dev image builds libchewing 0.14 and the server
finds it through `CHEWING14_LIB` / `CHEWING14_PATH`; in a container built
before that, add `--chewing-lib outputs/libchewing-0.14/lib/libchewing.so.3
--chewing-syspath outputs/libchewing-0.14/share/libchewing` (or leave it
out: the pool is then the decoder's own). Add `--pool-k 10` to keep the CPU
reranker's p95 latency under 50 ms at a small cost in toneless accuracy.

Resident memory on the dev host after typing 200 sentences: about 0.8 GB
decoder only, 0.95 GB with the CPU reranker, 2.6 GB plus 1.9 GB of GPU memory
with the GPU reranker. About 0.7 GB of it is the memory mapped n-gram, which
the kernel can reclaim (at the cost of slower lookups afterwards). Details:
[docs/ime.md](docs/ime.md#memory).

To stop it: Ctrl+C, or `pkill -f zhuyin_ime.server` in the container.

Without Docker, the decoder-only server runs on the host with Python 3.10+
and numpy, once `outputs/dict/` and `outputs/ngram/zhtw-o4-q16` exist (the
CPU reranker additionally needs `pip install onnxruntime` and the ONNX files
exported in step 2, no PyTorch; the libchewing 0.14 n-best needs a libchewing
0.14 build, which the dev container provides):

```bash
PYTHONPATH=src python3 -m zhuyin_ime.server
```

The server lowers its own priority (`--nice 5`) and sleeps when idle.

### 4. Install the IBus front end (on the host)

```bash
scripts/ibus/install.sh            # writes ~/.local/bin/zhuyin-ime-ibus
```

Then either register it for this session only (no root):

```bash
~/.local/bin/zhuyin-ime-ibus &     # registers the engine with the running ibus-daemon
ibus engine zhuyin-lm              # switch to it; back to English: ibus engine xkb:us::eng
```

or install it permanently so it appears in the desktop settings:

```bash
scripts/ibus/install.sh --system   # installs /usr/share/ibus/component/zhuyin-lm.xml (sudo)
```

and add "Zhuyin LM" under Settings > Keyboard > Input Sources (Chinese
(Taiwan)); switch input sources with Super+Space.

### 5. Type

Dai Chien (standard) layout. Finish each syllable with a tone key
(`6` `3` `4` `7`) or with Space to leave the tone open; the sentence is
converted as you type. Down opens the candidate window, `1` to `9` pick (the
choice is kept), Enter commits, Shift+`,` and Shift+`.` type full width
punctuation. The full key table is in [docs/ime.md](docs/ime.md).

### Troubleshooting

- Keys come out as plain Latin letters: the server is not running or the
  socket path differs. Check that `outputs/run/zhuyin-ime.sock` exists and
  that `ZHUYIN_IME_SOCKET` (if set) points to it.
- The engine is not listed: run `ibus restart` after `install.sh --system`,
  or start `~/.local/bin/zhuyin-ime-ibus` again after an IBus restart.
- To see the server log, run it in the foreground (without `&`).

### Uninstall

```bash
rm -f ~/.local/bin/zhuyin-ime-ibus
sudo rm -f /usr/share/ibus/component/zhuyin-lm.xml && ibus restart
```

## Repository

- `src/zhuyin_rescore/`: libchewing wrapper, lexicon, character n-gram and
  lattice beam decoder, LM scorers (Qwen, CUDA graph, character LM on torch
  and ONNX Runtime), metrics
- `src/zhuyin_ime/`: input method engine, conversion server, IBus front end
- `scripts/`: data, training, evaluation and benchmark entry points
- `tests/`: unit tests and a headless IBus end to end test
- `docker/`: the dev container (also builds libchewing 0.13.1 and 0.14)
- Docs: [proposal](docs/proposal.md), [phase 0 results](docs/results.md)
  (against libchewing 0.13.1), [phase 1 results](docs/results-phase1.md),
  [CPU reranker](docs/results-cpu-reranker.md), [input method](docs/ime.md)

License: Apache-2.0. libchewing (LGPL-2.1-or-later) is used as a separate
library and through its dictionary dump; no libchewing source is included.

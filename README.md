# Phonetic-Symbols-Candidate

Zhuyin candidate selection with small LM rescoring: libchewing generates the
top-k candidate sentences on the CPU, and a small resident causal LM (Qwen
0.5B to 1.5B) rescores them on the GPU using the committed text as context.

- Design and plan: [docs/proposal.md](docs/proposal.md)
- Current results: [docs/results.md](docs/results.md)
- Code: `src/zhuyin_rescore/` (libchewing wrapper, candidate generation, LM
  scorer, metrics), experiment entry points in `scripts/`, tests in `tests/`

Everything runs inside the dev container, see `docker/` and
`docs/results.md` for the commands.

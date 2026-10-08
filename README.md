# Phonetic-Symbols-Candidate

Zhuyin candidate selection with small LM rescoring: libchewing generates the
top-k candidate sentences on the CPU, and a small resident causal LM (Qwen
0.5B to 1.5B) rescores them on the GPU using the committed text as context.

- Design and plan: [docs/proposal.md](docs/proposal.md)
- Results: [docs/results-phase1.md](docs/results-phase1.md) (against libchewing
  0.14, five domains), [docs/results.md](docs/results.md) (phase 0, against
  0.13.1)
- Linux input method (IBus): [docs/ime.md](docs/ime.md)
- Code: `src/zhuyin_rescore/` (libchewing wrapper, candidate generation,
  char n-gram and lattice decoder, LM scorer, metrics), `src/zhuyin_ime/`
  (input method engine, conversion server, IBus front end), experiment entry
  points in `scripts/`, tests in `tests/`

Everything runs inside the dev container, see `docker/` and
`docs/results.md` for the commands.

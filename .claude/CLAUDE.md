# Project: Zhuyin candidate LM rescoring

Two stage pipeline: libchewing generates top-10 candidate sentences on CPU, a
small resident LM (0.5B to 1.5B, Qwen class) rescores them on GPU. Full design,
budgets and experiment plan: `docs/proposal.md`. Read it before non-trivial work.

## Hard constraints for all code work

1. Language: git commit messages, code comments, docstrings, log messages,
   identifiers and CLI help text are written in English.
2. ASCII only: commit messages and code comments must not contain special
   symbols. No emoji, no Unicode arrows, bullets, box drawing, smart quotes,
   em dashes or full width punctuation. Use plain ASCII equivalents such as
   `->`, `-`, `"`.
3. Exception: Chinese characters and zhuyin symbols are allowed only as data,
   i.e. string literals, test fixtures, and data files, because they are the
   subject of this project. Never use them in comments or commit messages.
4. Pull request descriptions follow the same rules; do not add an emoji footer.
5. No attribution trailers: do not add `Co-Authored-By:` or similar lines to
   commits or pull requests.

## Git commit message format

```
<type>: <Capitalized imperative summary, max 72 chars>

- <optional body as a concise list, no length limit>
```

`type` is one of: feat, fix, refactor, perf, test, docs, build, chore, exp.
Use `exp` for experiment scripts and result tables. See the `commit-style` skill.

## Environment

- All work runs inside the `phonetic-candidate-dev` container. Docker files live only in
  `docker/`; do not add Docker related files elsewhere. See `docker-dev` skill.
- Do not build or start containers unless asked; give the commands instead.
- The dev host GPU is an RTX 5090, while the proposal targets RTX 3060. Report
  which GPU a measurement came from; do not reuse 5090 numbers as 3060 results.

## Code conventions

- Python 3, source under `src/`, tests under `tests/`, experiment entry points
  under `scripts/`, outputs under `outputs/` (gitignored).
- Format and lint with `ruff`. Type hints on public functions.
- Never commit model weights, datasets, or HF cache content.
- Keep CPU busy waiting out of the scoring server: see the
  `resource-benchmark` skill for the required thread and sync settings.

## Skills in this repo

- `docker-dev`: container build, run, and exec workflow
- `libchewing-candidates`: driving libchewing via ctypes and building top-k candidates
- `zhuyin-data-pipeline`: corpus to zhuyin test sets with g2pW, three input conditions
- `lm-rescoring`: batched candidate scoring with KV cached context and score fusion
- `resource-benchmark`: latency, GPU and CPU utilization measurement protocol
- `commit-style`: commit message and comment rules

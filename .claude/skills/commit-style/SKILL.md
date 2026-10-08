---
name: commit-style
description: Rules for git commit messages, PR descriptions, and code comments in this repo (English only, ASCII only, conventional type prefix). Use whenever writing a commit, PR text, docstring, or code comment.
---

# Commit and comment style

## Rules

- English only in commit messages, PR descriptions, code comments,
  docstrings, and log messages.
- ASCII only in those places. Forbidden: emoji, Unicode arrows, bullets,
  check marks, box drawing, smart quotes, em and en dashes, full width
  punctuation, Chinese characters, zhuyin symbols.
- Chinese text and zhuyin are allowed only as data: string literals, test
  fixtures, data files. If a comment needs to refer to such data, describe
  it in English or point to the fixture name.

## Commit message format

```
<type>: <Imperative summary starting with a capital letter, max 72 chars, no trailing period>

- <optional body as a short list of points>
- <one change or reason per line, keep it concise>
```

- Summary: lowercase `type`, then a colon and a space, then a capital letter.
- Body: optional, no line length or line count limit, but keep it short.
  Write it as a `- ` list, one point per line. Explain why when not obvious.
- No trailers: do not add `Co-Authored-By:` or any other attribution line.

Types: feat, fix, refactor, perf, test, docs, build, chore, exp.

Good:

```
feat: Add ctypes wrapper for libchewing candidate list

- Wrap cand_open, cand_TotalChoice and cand_string_by_index_static
- Set argtypes and restype explicitly to avoid int truncation of pointers
```

Good: `exp: Add Oracle@10 results for notone input`
Bad: `feat: add scorer` (lowercase after colon), `Update stuff` (no type),
a paragraph body instead of a list, an emoji, any Chinese text, a
`Co-Authored-By:` trailer.

## Check before committing

Run this on the staged diff and the message; both should print nothing for
comment lines and the message.

```bash
git diff --cached -U0 | grep -nP '^\+.*#.*[^\x00-\x7F]'
printf '%s' "$MSG" | grep -nP '[^\x00-\x7F]'
```

The first check can flag legitimate data lines that contain a `#`; inspect
hits manually rather than blindly rewriting string literals.

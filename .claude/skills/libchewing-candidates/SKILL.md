---
name: libchewing-candidates
description: Drive libchewing from Python via ctypes to get the 1-best sentence, word boundaries, and per-interval alternatives, then build up to N candidate sentences for rescoring. Use when writing or debugging candidate generation, the Oracle@k metric, or anything calling libchewing.
---

# libchewing candidate generation

## Library facts

- Installed at `/opt/libchewing` in the dev container, loaded as
  `ctypes.CDLL("libchewing.so.3")`. Header: `/opt/libchewing/include/chewing/chewing.h`.
  Read the header before guessing a signature; it is the source of truth.
- Dictionary dir: `$CHEWING_PATH`. Create contexts with
  `chewing_new2(syspath, userpath, logger, loggerdata)`; pass a temp dir as
  `userpath` so experiments never touch or learn into a real user dictionary.
- Disable learning for benchmarks: `chewing_set_autoLearn(ctx, 1)`
  (`AUTOLEARN_DISABLED` is 1 in the v0.13.1 header).
- Strings returned by `*_String` functions must be freed with `chewing_free`;
  prefer the `*_static` variants when available. All strings are UTF-8 bytes.
- Upstream `contrib/python/chewing.py` is old and Python 2 era. Use it only as
  a reference; write the project wrapper in `src/` with explicit
  `argtypes` and `restype` for every function used.

## Feeding input

The v0.13.1 C API has no call to set a phone sequence directly, so input is
fed by key simulation: map each zhuyin syllable to keys of the default layout
(KB_DEFAULT, Dai Chien) and call `chewing_handle_Default` per key, with the
tone key or space to finish a syllable. `chewing_phone_to_bopomofo` helps
when debugging the parsed phones.

`chewing_get_phoneSeq` and `chewing_get_phoneSeqLen` return the parsed phone
sequence; assert it matches the intended input before trusting results.

## Building top-k sentences (1-best plus local substitution)

libchewing has no public whole sentence n-best API. Procedure from the proposal:

1. Read the 1-best preedit string (`chewing_buffer_String`) and its word
   intervals (`chewing_interval_Enumerate` / `chewing_interval_hasNext` /
   `chewing_interval_Get`).
2. For each interval, move the cursor there, open the candidate list
   (`chewing_cand_open` or the down key), enumerate the first few choices with
   `chewing_cand_TotalChoice` and `chewing_cand_string_by_index_static`,
   then close it (`chewing_cand_close`).
3. Combine: keep the 1-best, then single interval substitutions ranked by the
   candidate order, then pairs, until N (default 10) unique sentences.
4. Every candidate must have exactly one character per syllable. Drop any that
   do not; the LM comparison relies on equal length.

Record for each candidate: text, which intervals were substituted, and a
libchewing side score (rank based if no numeric score is exposed).

## Metrics owned by this stage

- libchewing 1-best CER (baseline).
- Oracle@k: CER of the best candidate among top-k. Report it before any LM
  work; if Oracle@10 is low, widen k (20 to 30) per the proposal risk table.

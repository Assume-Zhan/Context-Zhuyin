"""libchewing dictionary as a lexicon of (phrase, reading, frequency).

The binary dictionaries are dumped with chewing-cli (CSV: phrase, freq,
space separated bopomofo) and indexed by the condition specific key of the
reading, so the decoder can look up every phrase that matches a span of the
typed input in any of the three input conditions.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from zhuyin_rescore.textproc import PURE_HAN, cp950_ok
from zhuyin_rescore.zhuyin import derive_condition, has_tone, normalize, strip_tone

DICT_FILES = ("tsi.dat", "word.dat", "alt.dat")


@dataclass(frozen=True)
class Entry:
    phrase: str
    full: tuple[str, ...]
    freq: int


def dump_dictionary(dat_path: str, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chewing-cli", "dump", "--csv", dat_path, str(csv_path)], check=True, capture_output=True)


def load_entries(cache_dir: str | Path = "outputs/dict", syspath: str | None = None) -> list[Entry]:
    syspath = syspath or os.environ.get("CHEWING_PATH", "/opt/libchewing/share/libchewing")
    best: dict[tuple[str, tuple[str, ...]], int] = {}
    for name in DICT_FILES:
        csv_path = Path(cache_dir) / (name.replace(".dat", ".csv"))
        if not csv_path.exists():
            dump_dictionary(os.path.join(syspath, name), csv_path)
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3 or row[0].startswith("#"):
                    continue
                phrase, freq, bopo = row[0], row[1], row[2]
                syls = bopo.split("　")
                if len(syls) != len(phrase):
                    continue
                try:
                    full = tuple(normalize(s) for s in syls)
                    freq_i = int(freq)
                except ValueError:
                    continue
                key = (phrase, full)
                best[key] = max(best.get(key, 0), freq_i)
    return [Entry(p, r, f) for (p, r), f in best.items()]


class Lexicon:
    """Phrase lookup by condition specific reading key.

    Conditions: "full", "notone" and "initial" as in the benchmark, plus
    "mixed" for live typing, where a syllable typed with a tone mark must
    match exactly and a syllable typed without one matches any tone.
    """

    def __init__(self, entries: list[Entry], condition: str, charset: str | None = "cp950", max_len: int = 6):
        self.condition = condition
        self.max_len = max_len
        key_condition = "notone" if condition == "mixed" else condition
        self._key_condition = key_condition
        index: dict[tuple[str, ...], dict[str, int]] = defaultdict(dict)
        # Full readings of every phrase, for exact matching and consistency checks.
        self.full_readings: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for e in entries:
            if len(e.phrase) > max_len or not PURE_HAN.match(e.phrase):
                continue
            if charset == "cp950" and not cp950_ok(e.phrase):
                continue
            key = tuple(derive_condition(list(e.full), key_condition))
            bucket = index[key]
            bucket[e.phrase] = max(bucket.get(e.phrase, 0), e.freq)
            self.full_readings[e.phrase].add(e.full)
        # Sort each bucket by frequency, most frequent first.
        self.index = {k: sorted(v.items(), key=lambda kv: -kv[1]) for k, v in index.items()}
        self._span_cache: dict[tuple[tuple[str, ...], int | None], list[tuple[str, int]]] = {}

    def key(self, syllables: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        if self.condition == "mixed":
            return tuple(strip_tone(s) for s in syllables)
        return tuple(syllables)

    def _reading_ok(self, full: tuple[str, ...], syllables: tuple[str, ...]) -> bool:
        if self.condition == "mixed":
            return all(r == s for r, s in zip(full, syllables, strict=True) if has_tone(s))
        return tuple(derive_condition(list(full), self.condition)) == syllables

    def lookup(self, key: tuple[str, ...], limit: int | None = None) -> list[tuple[str, int]]:
        hits = self.index.get(key, [])
        return hits[:limit] if limit else hits

    def lookup_span(self, syllables: tuple[str, ...], limit: int | None = None) -> list[tuple[str, int]]:
        """Phrases (most frequent first) whose reading matches the typed span."""
        cache_key = (syllables, limit)
        hit = self._span_cache.get(cache_key)
        if hit is not None:
            return hit
        hits = self.index.get(self.key(syllables), [])
        if self.condition == "mixed" and any(has_tone(s) for s in syllables):
            hits = [
                (p, f) for p, f in hits if any(self._reading_ok(r, syllables) for r in self.full_readings[p])
            ]
        hits = hits[:limit] if limit else hits
        if len(self._span_cache) > 200_000:
            self._span_cache.clear()
        self._span_cache[cache_key] = hits
        return hits

    def reading_matches(self, phrase: str, syllables: tuple[str, ...]) -> bool | None:
        """None if phrase is not a multi char dictionary word, else whether
        any of its readings matches the typed syllables."""
        if len(phrase) < 2:
            return None
        readings = self.full_readings.get(phrase)
        if not readings:
            return None
        return any(len(r) == len(syllables) and self._reading_ok(r, syllables) for r in readings)

    def edges(self, syllables: list[str], single_limit: int | None = None):
        """All (start, end, phrase, freq) spans matching the condition input."""
        out = []
        n = len(syllables)
        for i in range(n):
            for length in range(1, min(self.max_len, n - i) + 1):
                limit = single_limit if length == 1 else None
                hits = self.lookup_span(tuple(syllables[i : i + length]), limit)
                for phrase, freq in hits:
                    out.append((i, i + length, phrase, freq))
        return out

    @staticmethod
    def log_prior(freq: int) -> float:
        return math.log1p(freq)

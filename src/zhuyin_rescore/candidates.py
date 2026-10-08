"""Top-k candidate sentences from libchewing.

Two sources, both driven through the public C API:

- tab: the engine's own whole sentence n-best, reached by pressing Tab at the
  end of the buffer (1-best plus 9 alternatives, then it wraps around).
- sub: 1-best plus local substitution. At every buffer position the candidate
  window lists phrases of each length that start there (longest list first).
  Replacing the matching span of the 1-best with one of those phrases gives a
  candidate with the same number of characters.

The pool keeps the libchewing 1-best at rank 0 and never filters it.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from zhuyin_rescore.chewing import ENGINE_FOR_CONDITION, Chewing

HAN_RE = re.compile("^[㐀-䶿一-鿿\U00020000-\U0003134f]+$")


def is_han(text: str) -> bool:
    return bool(HAN_RE.match(text))


def in_charset(text: str, charset: str | None) -> bool:
    """True if text is encodable in charset (None disables the check).

    cp950 (Big5) keeps every common Taiwan form and drops Simplified only
    characters and very rare glyphs from the alternatives.
    """
    if charset is None:
        return True
    try:
        text.encode(charset)
    except UnicodeEncodeError:
        return False
    return True


@dataclass
class Substitution:
    pos: int
    length: int
    phrase: str
    index: int  # position of the phrase within its libchewing candidate list


@dataclass
class Candidate:
    text: str
    source: str  # "1best", "tab" or "sub"
    rank: int
    sub: Substitution | None = None


@dataclass
class CandidateSet:
    syllables: list[str]
    onebest: str
    valid_input: bool  # 1-best has one Han character per syllable
    candidates: list[Candidate] = field(default_factory=list)
    tab: list[str] = field(default_factory=list)
    sub: list[str] = field(default_factory=list)
    gen_ms: float = 0.0

    def texts(self, k: int | None = None) -> list[str]:
        return [c.text for c in self.candidates[:k]]

    def to_json(self) -> dict:
        return asdict(self)


def tab_nbest(chewing: Chewing, onebest: str, max_n: int = 30) -> list[str]:
    """Whole sentence n-best via Tab, stopping when the cycle wraps."""
    out = [onebest]
    seen = {onebest}
    for _ in range(max_n - 1):
        text = chewing.next_conversion()
        if text in seen:
            break
        seen.add(text)
        out.append(text)
    return out


def substitution_options(chewing: Chewing, n_chars: int, per_list: int) -> list[Substitution]:
    """Read the candidate lists at every position of the current buffer."""
    lib, ctx = chewing.lib, chewing.ctx
    options = []
    for pos in range(n_chars):
        lib.chewing_handle_Home(ctx)
        for _ in range(pos):
            lib.chewing_handle_Right(ctx)
        if lib.chewing_cand_open(ctx) != 0:
            continue
        try:
            lib.chewing_cand_list_first(ctx)
            while True:
                total = lib.chewing_cand_TotalChoice(ctx)
                for i in range(min(total, per_list)):
                    raw = lib.chewing_cand_string_by_index_static(ctx, i)
                    phrase = raw.decode("utf-8") if raw else ""
                    if phrase and pos + len(phrase) <= n_chars:
                        options.append(Substitution(pos, len(phrase), phrase, i))
                if not lib.chewing_cand_list_has_next(ctx):
                    break
                lib.chewing_cand_list_next(ctx)
        finally:
            lib.chewing_cand_close(ctx)
    lib.chewing_handle_End(ctx)
    return options


class CandidateGenerator:
    """Builds candidate pools for one input condition."""

    def __init__(
        self,
        condition: str,
        pool_size: int = 30,
        per_list: int = 5,
        charset: str | None = "cp950",
        use_tab: bool = True,
    ):
        self.condition = condition
        self.pool_size = pool_size
        self.per_list = per_list
        self.charset = charset
        self.use_tab = use_tab
        self.chewing = Chewing(ENGINE_FOR_CONDITION[condition])
        # Frequency order within each candidate list is a better prior for
        # which substitutions to try first.
        self.chewing.set_int("chewing.sort_candidates_by_frequency", 1)

    def close(self) -> None:
        self.chewing.close()

    def _ok(self, text: str, n: int) -> bool:
        return len(text) == n and is_han(text) and in_charset(text, self.charset)

    def generate(self, syllables: list[str]) -> CandidateSet:
        t0 = time.perf_counter()
        n = len(syllables)
        conv = self.chewing.convert(syllables)
        onebest = conv.text
        cset = CandidateSet(
            syllables=list(syllables),
            onebest=onebest,
            valid_input=len(onebest) == n and is_han(onebest),
        )
        cset.candidates.append(Candidate(onebest, "1best", 0))
        if not cset.valid_input:
            cset.gen_ms = (time.perf_counter() - t0) * 1000
            return cset

        options = substitution_options(self.chewing, n, self.per_list)
        if self.use_tab:
            cset.tab = tab_nbest(self.chewing, onebest)

        # Merge: 1-best, then the remaining Tab n-best and single substitutions
        # interleaved by their list index, so both sources fill the pool.
        seen = {onebest}
        subs = sorted(options, key=lambda s: (s.index, -s.length, s.pos))
        sub_texts = []
        sub_seen = {onebest}
        for s in subs:
            text = onebest[: s.pos] + s.phrase + onebest[s.pos + s.length :]
            if text in sub_seen or not self._ok(text, n):
                continue
            sub_seen.add(text)
            sub_texts.append((text, s))
        cset.sub = [onebest] + [t for t, _ in sub_texts]
        tab_alts = cset.tab[1:]
        queue: list[tuple[str, str, Substitution | None]] = []
        ti = si = 0
        while ti < len(tab_alts) or si < len(sub_texts):
            if ti < len(tab_alts):
                queue.append((tab_alts[ti], "tab", None))
                ti += 1
            for _ in range(2):
                if si < len(sub_texts):
                    queue.append((sub_texts[si][0], "sub", sub_texts[si][1]))
                    si += 1
        for text, source, sub in queue:
            if len(cset.candidates) >= self.pool_size:
                break
            if text in seen or not self._ok(text, n):
                continue
            seen.add(text)
            cset.candidates.append(Candidate(text, source, len(cset.candidates), sub))
        cset.gen_ms = (time.perf_counter() - t0) * 1000
        return cset

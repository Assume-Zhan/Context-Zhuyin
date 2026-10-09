"""IME session state machine, independent of any input method framework.

A session holds the syllables typed so far, the syllable being composed,
the cursor, the phrases the user picked (locked spans), the candidate window
and the committed history used as LM context. Punctuation stays in the
preedit like a syllable until Enter, so earlier choices remain editable; it
splits the input into segments that are decoded separately, each with the
text before it as context. After every change the beam decoder updates the
segment being typed and the 1-best is shown right away; an optional LM
reranker can later pick a better candidate for a segment (apply_rerank), as
long as that segment has not changed in the meantime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from zhuyin_ime.composer import TONE_KEYS, Composer
from zhuyin_rescore.beam import BeamDecoder, Decoded, Weights
from zhuyin_rescore.lexicon import Lexicon
from zhuyin_rescore.ngram import CharNgram

# Shifted or unused keys that type full width punctuation.
PUNCTUATION = {
    "<": "，", ">": "。", "?": "？", "!": "！", ":": "：", '"': "；", "'": "、",
    "[": "「", "]": "」", "{": "『", "}": "』", "\\": "、", "(": "（", ")": "）", "~": "～",
}  # fmt: skip
# With Ctrl held, keys that are bopomofo symbols on the Dai Chien layout type
# punctuation too (Ctrl+, for a comma without reaching for Shift).
CTRL_PUNCTUATION = {",": "，", ".": "。", ";": "；", **PUNCTUATION}
MARKS = frozenset(CTRL_PUNCTUATION.values())
PAGE_SIZE = 9
HISTORY_CHARS = 200
MAX_SYLLABLES = 40


@dataclass
class Key:
    char: str = ""  # the character the key produces, "" for special keys
    name: str = ""  # special keys: Return, BackSpace, Delete, Escape, Left, Right, Up, Down, ...
    shift: bool = False
    ctrl: bool = False
    alt: bool = False


@dataclass
class State:
    """What a front end needs to draw after a key."""

    handled: bool
    commit: str = ""
    preedit: str = ""
    cursor: int = 0
    candidates: list[str] = field(default_factory=list)
    highlight: int = 0
    page: int = 0
    pages: int = 0
    version: int = 0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    """Decoder k-best of one run of syllables between punctuation marks."""

    kbest: list[Decoded]
    choice: int = 0  # index of the candidate shown in the preedit
    seen: bool = False  # already scored by the reranker


class Engine:
    """Shared, read only resources; one Session per input context."""

    def __init__(
        self,
        lexicon: Lexicon,
        ngram: CharNgram,
        weights: Weights | None = None,
        beam: int = 32,
        k: int = 10,
        fusion: tuple[float, float, float] = (0.05, 2.0, 0.05),
    ):
        if lexicon.condition != "mixed":
            raise ValueError("the IME needs a lexicon built with condition='mixed'")
        self.lexicon = lexicon
        self.ngram = ngram
        self.weights = weights or Weights(prior=0.3, edge=-1.0, mismatch=-3.0)
        self.beam = beam
        self.k = k
        self.fusion = fusion

    def valid_syllable(self, syllable: str) -> bool:
        return bool(self.lexicon.lookup_span((syllable,), 1))

    def new_session(self) -> Session:
        return Session(self)


class Session:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.decoder = BeamDecoder(engine.lexicon, engine.ngram, beam=engine.beam, weights=engine.weights)
        self.composer = Composer(is_valid=engine.valid_syllable)
        # Syllables, plus the punctuation marks typed between them (one
        # preedit character each, so positions index both).
        self.syllables: list[str] = []
        self.locks: dict[tuple[int, int], str] = {}
        self.cursor = 0
        self.history = ""
        self.version = 0
        # Segments of the current input by (context, syllables, relative
        # locks); kept across keys so only the segment being typed is decoded.
        self.segments: dict[tuple, Segment] = {}
        self.segment_keys: list[tuple] = []
        self.text = ""
        self.cand_items: list[tuple[int, int, str]] = []
        self.cand_page = 0
        self.cand_cursor = 0

    # ------------------------------------------------------------------ helpers
    def _text(self) -> str:
        return self.text

    def _decode(self) -> None:
        self.version += 1
        self._rebuild()

    def _rebuild(self) -> None:
        """Decode each segment with the text before it as context; segments
        whose context, syllables and locks are unchanged are reused."""
        segments, keys, text = {}, [], ""
        n = len(self.syllables)
        start = 0
        for i in range(n + 1):
            if i < n and self.syllables[i] not in MARKS:
                continue
            if i > start:
                locks = tuple(
                    sorted(
                        (s - start, e - start, p) for (s, e), p in self.locks.items() if start <= s < e <= i
                    )
                )
                context, syls = (self.history + text)[-64:], self.syllables[start:i]
                key = (context, tuple(syls), locks)
                seg = self.segments.get(key)
                if seg is None:
                    seg = Segment(
                        self.decoder.decode(syls, history=context, k=self.engine.k, locks=list(locks))
                    )
                segments[key] = seg
                keys.append(key)
                text += seg.kbest[seg.choice].text if seg.kbest else "".join(key[1])
            if i < n:
                text += self.syllables[i]
            start = i + 1
        self.segments, self.segment_keys, self.text = segments, keys, text

    def _state(self, handled: bool = True, commit: str = "") -> State:
        text = self._text() if self.syllables else ""
        preedit = text + self.composer.text()
        cursor = len(text) + len(self.composer.text()) if not self.composer.empty() else self.cursor
        st = State(handled=handled, commit=commit, preedit=preedit, cursor=cursor, version=self.version)
        if self.cand_items:
            st.candidates = [p for _, _, p in self._page_items(self.cand_page)]
            st.highlight, st.page, st.pages = self.cand_cursor, self.cand_page, self._pages()
        return st

    def _commit_all(self, boundary: str = "") -> str:
        """Commit the preedit; boundary is recorded in the history only
        (Enter ends a line or message, so the next clause starts fresh)."""
        text = self._text() if self.syllables else ""
        self.history = (self.history + text + boundary)[-HISTORY_CHARS:]
        self.syllables, self.locks, self.cursor = [], {}, 0
        self.composer.clear()
        self.cand_items = []
        self._decode()
        return text

    def _busy(self) -> bool:
        return bool(self.syllables) or not self.composer.empty()

    # ---------------------------------------------------------------- candidates
    def _pages(self) -> int:
        return max((len(self.cand_items) + PAGE_SIZE - 1) // PAGE_SIZE, 1)

    def _page_items(self, page: int) -> list[tuple[int, int, str]]:
        return self.cand_items[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    def _open_candidates(self) -> State:
        """Phrases starting at the cursor, or ending at the end of the input
        (before any trailing punctuation), within one segment."""
        n = len(self.syllables)
        if self.cursor < n:
            if self.syllables[self.cursor] in MARKS:
                return self._state()
            end = self.cursor
            while end < n and self.syllables[end] not in MARKS:
                end += 1
            spans = [(self.cursor, self.cursor + n_syl) for n_syl in range(min(6, end - self.cursor), 0, -1)]
        else:
            end = n
            while end > 0 and self.syllables[end - 1] in MARKS:
                end -= 1
            start = end
            while start > 0 and self.syllables[start - 1] not in MARKS:
                start -= 1
            spans = [(end - length, end) for length in range(min(6, end - start), 0, -1)]
        items, seen = [], set()
        for s, e in spans:
            for phrase, _ in self.engine.lexicon.lookup_span(tuple(self.syllables[s:e])):
                if (s, e, phrase) not in seen:
                    seen.add((s, e, phrase))
                    items.append((s, e, phrase))
        self.cand_items, self.cand_page, self.cand_cursor = items, 0, 0
        return self._state()

    def _pick(self, index: int) -> State:
        pos = self.cand_page * PAGE_SIZE + index
        if pos >= len(self.cand_items):
            return self._state()
        s, e, phrase = self.cand_items[pos]
        self.locks = {span: p for span, p in self.locks.items() if span[1] <= s or span[0] >= e}
        self.locks[(s, e)] = phrase
        self.cursor = min(e, len(self.syllables))
        self.cand_items = []
        self._decode()
        return self._state()

    def _candidate_key(self, key: Key) -> State:
        """Digits pick from the page; Up and Down move the highlight (and
        turn the page at its ends), Enter picks the highlighted candidate."""
        pages = self._pages()
        if key.char and key.char in "123456789":
            return self._pick(int(key.char) - 1)
        if key.name == "Return":
            return self._pick(self.cand_cursor)
        if key.name == "Down":
            if self.cand_cursor + 1 < len(self._page_items(self.cand_page)):
                self.cand_cursor += 1
            else:
                self.cand_page, self.cand_cursor = (self.cand_page + 1) % pages, 0
        elif key.name == "Up":
            if self.cand_cursor > 0:
                self.cand_cursor -= 1
            else:
                self.cand_page = (self.cand_page - 1) % pages
                self.cand_cursor = len(self._page_items(self.cand_page)) - 1
        elif key.name == "Page_Down" or key.char == " ":
            self.cand_page, self.cand_cursor = (self.cand_page + 1) % pages, 0
        elif key.name == "Page_Up":
            self.cand_page, self.cand_cursor = (self.cand_page - 1) % pages, 0
        elif key.name == "Escape":
            self.cand_items = []
        return self._state()

    # ---------------------------------------------------------------- key input
    def process_key(self, key: Key) -> State:
        mark = "" if key.alt else (CTRL_PUNCTUATION if key.ctrl else PUNCTUATION).get(key.char, "")
        if (key.ctrl or key.alt) and not mark:
            return State(handled=False, preedit=self._state().preedit, version=self.version)
        if self.cand_items:
            return self._candidate_key(key)

        ch, name = key.char, key.name
        if mark:  # before the symbol keys: Ctrl+, is a comma, not the symbol on that key
            return self._add_mark(mark)
        if ch and Composer.handles(ch) and not key.shift:
            if ch in TONE_KEYS:
                return self._finish_syllable(TONE_KEYS[ch])
            if self.cursor < len(self.syllables) and self.composer.empty():
                self.cursor = len(self.syllables)
            self.composer.feed_symbol(ch)
            return self._state()
        if ch == " ":
            if not self.composer.empty():
                return self._finish_syllable("")
            if self.syllables:
                return self._open_candidates()
            return State(handled=False, version=self.version)
        if name == "Return":
            if not self._busy():
                return State(handled=False, version=self.version)
            return self._state(commit=self._commit_all(boundary="\n"))
        if name == "BackSpace":
            return self._backspace()
        if name == "Delete":
            if self.cursor < len(self.syllables) and self.composer.empty():
                self._remove_syllable(self.cursor)
                return self._state()
            return self._state(handled=self._busy())
        if name == "Escape":
            if not self._busy():
                return State(handled=False, version=self.version)
            if not self.composer.empty():
                self.composer.clear()
            else:
                self.syllables, self.locks, self.cursor = [], {}, 0
                self._decode()
            return self._state()
        if name in ("Left", "Right", "Home", "End"):
            if not self.syllables:
                return State(handled=False, version=self.version)
            n = len(self.syllables)
            self.cursor = {
                "Left": max(0, self.cursor - 1),
                "Right": min(n, self.cursor + 1),
                "Home": 0,
                "End": n,
            }[name]
            return self._state()
        if name == "Down" and self.syllables and self.composer.empty():
            return self._open_candidates()
        if self._busy():
            # Any other printable key: commit what we have and let it through.
            commit = self._commit_all()
            st = self._state(handled=False, commit=commit)
            return st
        return State(handled=False, version=self.version)

    def _finish_syllable(self, tone: str) -> State:
        if self.composer.empty():
            return self._state(handled=self._busy())
        syl = self.composer.finish(tone)
        if syl is None:
            return self._state()  # not a valid syllable: keep composing
        self.syllables.append(syl)
        self.cursor = len(self.syllables)
        commit = ""
        if len(self.syllables) > MAX_SYLLABLES:
            commit = self._commit_all()
        else:
            self._decode()
        return self._state(commit=commit)

    def _add_mark(self, mark: str) -> State:
        """Punctuation joins the preedit; a syllable still being composed is
        finished without a tone first (dropped if it is not a syllable)."""
        if not self.composer.empty():
            syl = self.composer.finish("")
            self.composer.clear()
            if syl is not None:
                self.syllables.append(syl)
        self.syllables.append(mark)
        self.cursor = len(self.syllables)
        if len(self.syllables) > MAX_SYLLABLES:
            return self._state(commit=self._commit_all())
        self._decode()
        return self._state()

    def _remove_syllable(self, index: int) -> None:
        del self.syllables[index]
        shifted = {}
        for (s, e), p in self.locks.items():
            if e <= index:
                shifted[(s, e)] = p
            elif s > index:
                shifted[(s - 1, e - 1)] = p
        self.locks = shifted
        self._decode()

    def _backspace(self) -> State:
        if not self.composer.empty():
            self.composer.backspace()
            return self._state()
        if self.cursor > 0:
            self._remove_syllable(self.cursor - 1)
            self.cursor -= 1
            return self._state()
        return self._state(handled=self._busy())

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> State:
        self.syllables, self.locks, self.cursor = [], {}, 0
        self.composer.clear()
        self.cand_items = []
        self._decode()
        return self._state()

    def focus_out(self) -> State:
        """Commit pending text when the input context loses focus."""
        commit = self._commit_all() if self.syllables else ""
        self.composer.clear()
        return self._state(commit=commit)

    # ---------------------------------------------------------------- reranking
    def rerank_request(self) -> tuple[tuple, str, list[Decoded], list[str]] | None:
        """(segment key, context, k-best, typed syllables) of the first
        segment the reranker has not scored yet, or None."""
        for key in self.segment_keys:
            seg = self.segments[key]
            if not seg.seen and len(seg.kbest) >= 2:
                return key, key[0], list(seg.kbest), list(key[1])
        return None

    def apply_rerank(self, key: tuple, lm_scores: list[float]) -> State | None:
        """Fuse LM scores with the decoder's n-gram scores for the segment
        with this key; None if it is stale or the choice did not change.
        Later segments are decoded again with the new text as context, but
        the version stays: the typed input is the same."""
        seg = self.segments.get(key)
        if seg is None or seg.seen or len(lm_scores) != len(seg.kbest):
            return None
        seg.seen = True
        a, beta, mu = self.engine.fusion
        best, best_s = 0, float("-inf")
        for i, (d, lm) in enumerate(zip(seg.kbest, lm_scores, strict=True)):
            s = lm + a * d.feats[0] - beta * (i > 0) - mu * i
            if s > best_s:
                best, best_s = i, s
        if best == seg.choice:
            return None
        seg.choice = best
        self._rebuild()
        return self._state()

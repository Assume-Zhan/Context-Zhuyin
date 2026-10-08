"""IME session state machine, independent of any input method framework.

A session holds the syllables typed so far, the syllable being composed,
the cursor, the phrases the user picked (locked spans), the candidate window
and the committed history used as LM context. After every change it decodes
incrementally with the beam decoder and shows the 1-best right away; an
optional LM reranker can later replace the preedit with a better candidate
(apply_rerank), as long as the input has not changed in the meantime.
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


class Engine:
    """Shared, read only resources; one Session per input context."""

    def __init__(
        self,
        lexicon: Lexicon,
        ngram: CharNgram,
        weights: Weights | None = None,
        beam: int = 32,
        k: int = 10,
        fusion: tuple[float, float] = (0.05, 2.0),
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
        self.syllables: list[str] = []
        self.locks: dict[tuple[int, int], str] = {}
        self.cursor = 0
        self.history = ""
        self.kbest: list[Decoded] = []
        self.reranked: str | None = None
        self.version = 0
        self.cand_items: list[tuple[int, int, str]] = []
        self.cand_page = 0

    # ------------------------------------------------------------------ helpers
    def _text(self) -> str:
        if self.reranked is not None:
            return self.reranked
        if self.kbest:
            return self.kbest[0].text
        return "".join(self.syllables)

    def _decode(self) -> None:
        self.version += 1
        self.reranked = None
        locks = [(s, e, p) for (s, e), p in self.locks.items()]
        self.kbest = self.engine_decode(locks) if self.syllables else []

    def engine_decode(self, locks) -> list[Decoded]:
        return self.decoder.decode(self.syllables, history=self.history[-64:], k=self.engine.k, locks=locks)

    def _state(self, handled: bool = True, commit: str = "") -> State:
        text = self._text() if self.syllables else ""
        preedit = text + self.composer.text()
        cursor = len(text) + len(self.composer.text()) if not self.composer.empty() else self.cursor
        st = State(handled=handled, commit=commit, preedit=preedit, cursor=cursor, version=self.version)
        if self.cand_items:
            pages = (len(self.cand_items) + PAGE_SIZE - 1) // PAGE_SIZE
            items = self.cand_items[self.cand_page * PAGE_SIZE : (self.cand_page + 1) * PAGE_SIZE]
            st.candidates = [p for _, _, p in items]
            st.page, st.pages = self.cand_page, pages
        return st

    def _commit_all(self, extra: str = "") -> str:
        text = (self._text() if self.syllables else "") + extra
        self.history = (self.history + text)[-HISTORY_CHARS:]
        self.syllables, self.locks, self.cursor = [], {}, 0
        self.composer.clear()
        self.cand_items = []
        self._decode()
        return text

    def _busy(self) -> bool:
        return bool(self.syllables) or not self.composer.empty()

    # ---------------------------------------------------------------- candidates
    def _open_candidates(self) -> State:
        n = len(self.syllables)
        if self.cursor < n:
            spans = [(self.cursor, self.cursor + length) for length in range(min(6, n - self.cursor), 0, -1)]
        else:
            spans = [(n - length, n) for length in range(min(6, n), 0, -1)]
        items, seen = [], set()
        for s, e in spans:
            for phrase, _ in self.engine.lexicon.lookup_span(tuple(self.syllables[s:e])):
                if (s, e, phrase) not in seen:
                    seen.add((s, e, phrase))
                    items.append((s, e, phrase))
        self.cand_items, self.cand_page = items, 0
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
        pages = (len(self.cand_items) + PAGE_SIZE - 1) // PAGE_SIZE
        if key.char and key.char in "123456789":
            return self._pick(int(key.char) - 1)
        if key.name in ("Down", "Page_Down") or key.char == " ":
            self.cand_page = (self.cand_page + 1) % max(pages, 1)
        elif key.name in ("Up", "Page_Up"):
            self.cand_page = (self.cand_page - 1) % max(pages, 1)
        elif key.name == "Escape":
            self.cand_items = []
        return self._state()

    # ---------------------------------------------------------------- key input
    def process_key(self, key: Key) -> State:
        if key.ctrl or key.alt:
            return State(handled=False, preedit=self._state().preedit, version=self.version)
        if self.cand_items:
            return self._candidate_key(key)

        ch, name = key.char, key.name
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
        if ch in PUNCTUATION:
            commit = self._commit_all(PUNCTUATION[ch])
            return self._state(commit=commit)
        if name == "Return":
            if not self._busy():
                return State(handled=False, version=self.version)
            return self._state(commit=self._commit_all())
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
    def rerank_request(self) -> tuple[int, str, list[Decoded]] | None:
        if len(self.kbest) < 2:
            return None
        return self.version, self.history[-HISTORY_CHARS:], list(self.kbest)

    def apply_rerank(self, version: int, lm_scores: list[float]) -> State | None:
        """Fuse LM scores with the decoder's n-gram scores; None if stale."""
        if version != self.version or len(lm_scores) != len(self.kbest):
            return None
        a, beta = self.engine.fusion
        best, best_s = None, float("-inf")
        for i, (d, lm) in enumerate(zip(self.kbest, lm_scores, strict=True)):
            s = lm + a * d.feats[0] - beta * (i > 0)
            if s > best_s:
                best, best_s = d.text, s
        if best == self._text():
            return None
        self.reranked = best
        return self._state()

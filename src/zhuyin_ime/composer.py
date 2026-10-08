"""Bopomofo syllable composition for the Dai Chien (standard) keyboard layout.

Each key maps to one bopomofo symbol. A syllable has one slot each for the
initial, the medial and the rime; typing a symbol of a filled slot replaces
it, as most zhuyin IMEs do. A tone key finishes the syllable with that tone;
Space finishes it without a tone mark, which the decoder treats as "any
tone" (so toneless typing works without a separate mode).
"""

from __future__ import annotations

from zhuyin_rescore.zhuyin import DACHEN_KEYS, INITIALS, MEDIALS, RIMES, TONE_MARKS

KEY_TO_SYMBOL = {key: sym for sym, key in DACHEN_KEYS.items()}
TONE_KEYS = {DACHEN_KEYS[t]: t for t in TONE_MARKS}


class Composer:
    def __init__(self, is_valid=None):
        """is_valid(syllable) -> bool rejects syllables (with their tone
        mark, if any) that the lexicon does not know; None accepts all."""
        self.is_valid = is_valid
        self.initial = self.medial = self.rime = ""

    def empty(self) -> bool:
        return not (self.initial or self.medial or self.rime)

    def text(self) -> str:
        return self.initial + self.medial + self.rime

    def clear(self) -> None:
        self.initial = self.medial = self.rime = ""

    @staticmethod
    def handles(ch: str) -> bool:
        return ch in KEY_TO_SYMBOL

    def feed_symbol(self, ch: str) -> bool:
        """Add the symbol for key ch. Returns False if ch is not a symbol key."""
        sym = KEY_TO_SYMBOL.get(ch)
        if sym is None or sym in TONE_MARKS:
            return False
        if sym in INITIALS:
            self.initial = sym
        elif sym in MEDIALS:
            self.medial = sym
        elif sym in RIMES:
            self.rime = sym
        return True

    def finish(self, tone: str = "") -> str | None:
        """Close the syllable; returns it (canonical form) or None if invalid."""
        if self.empty():
            return None
        syllable = self.text() + tone
        if self.is_valid is not None and not self.is_valid(syllable):
            return None
        self.clear()
        return syllable

    def backspace(self) -> bool:
        for slot in ("rime", "medial", "initial"):
            if getattr(self, slot):
                setattr(self, slot, "")
                return True
        return False

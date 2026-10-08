"""Zhuyin (bopomofo) syllable utilities.

Canonical syllable form used across the project: bopomofo symbols in
initial, medial, rime order, followed by a tone mark for tones 2 to 5. Tone 1
has no mark. The neutral tone mark is placed at the end (as typed), not in
front (as printed in dictionaries).
"""

from __future__ import annotations

from dataclasses import dataclass

INITIALS = "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ"
MEDIALS = "ㄧㄨㄩ"
RIMES = "ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ"
TONE_MARKS = "ˊˇˋ˙"
TONE1_MARK = "ˉ"
SYMBOLS = INITIALS + MEDIALS + RIMES

# Numeric tone (as produced by g2pW) to tone mark. Tone 1 is unmarked.
TONE_NUM_TO_MARK = {"1": "", "2": "ˊ", "3": "ˇ", "4": "ˋ", "5": "˙"}

# Default (Dai Chien) keyboard layout, KB_DEFAULT in libchewing.
DACHEN_KEYS = {
    "ㄅ": "1",
    "ㄆ": "q",
    "ㄇ": "a",
    "ㄈ": "z",
    "ㄉ": "2",
    "ㄊ": "w",
    "ㄋ": "s",
    "ㄌ": "x",
    "ㄍ": "e",
    "ㄎ": "d",
    "ㄏ": "c",
    "ㄐ": "r",
    "ㄑ": "f",
    "ㄒ": "v",
    "ㄓ": "5",
    "ㄔ": "t",
    "ㄕ": "g",
    "ㄖ": "b",
    "ㄗ": "y",
    "ㄘ": "h",
    "ㄙ": "n",
    "ㄧ": "u",
    "ㄨ": "j",
    "ㄩ": "m",
    "ㄚ": "8",
    "ㄛ": "i",
    "ㄜ": "k",
    "ㄝ": ",",
    "ㄞ": "9",
    "ㄟ": "o",
    "ㄠ": "l",
    "ㄡ": ".",
    "ㄢ": "0",
    "ㄣ": "p",
    "ㄤ": ";",
    "ㄥ": "/",
    "ㄦ": "-",
    "ˊ": "6",
    "ˇ": "3",
    "ˋ": "4",
    "˙": "7",
}

CONDITIONS = ("full", "notone", "initial")


@dataclass(frozen=True)
class Syllable:
    initial: str
    medial: str
    rime: str
    tone: str  # one of "", or a mark in TONE_MARKS

    @property
    def text(self) -> str:
        return self.initial + self.medial + self.rime + self.tone


def parse_syllable(s: str) -> Syllable:
    """Parse a bopomofo syllable in any common notation.

    Accepts a leading or trailing neutral tone mark, an explicit tone 1 mark,
    or a trailing numeric tone 1 to 5.
    """
    s = s.strip()
    tone = ""
    if s and s[-1] in TONE_NUM_TO_MARK:
        tone = TONE_NUM_TO_MARK[s[-1]]
        s = s[:-1]
    if s.startswith("˙"):
        tone = "˙"
        s = s[1:]
    if s and (s[-1] in TONE_MARKS or s[-1] == TONE1_MARK):
        tone = "" if s[-1] == TONE1_MARK else s[-1]
        s = s[:-1]
    initial = medial = rime = ""
    rest = s
    if rest and rest[0] in INITIALS:
        initial, rest = rest[0], rest[1:]
    if rest and rest[0] in MEDIALS:
        medial, rest = rest[0], rest[1:]
    if rest and rest[0] in RIMES:
        rime, rest = rest[0], rest[1:]
    if rest or not (initial or medial or rime):
        raise ValueError(f"not a bopomofo syllable: {s!r}")
    return Syllable(initial, medial, rime, tone)


def normalize(s: str) -> str:
    """Return the canonical form of a syllable."""
    return parse_syllable(s).text


def strip_tone(s: str) -> str:
    syl = parse_syllable(s)
    return syl.initial + syl.medial + syl.rime


def initial_only(s: str) -> str:
    """Initial of a syllable, or its first symbol when it has no initial."""
    syl = parse_syllable(s)
    return syl.initial or (syl.medial + syl.rime)[0]


def derive_condition(full: list[str], condition: str) -> list[str]:
    """Map canonical full syllables to the input of a given condition."""
    if condition == "full":
        return [normalize(s) for s in full]
    if condition == "notone":
        return [strip_tone(s) for s in full]
    if condition == "initial":
        return [initial_only(s) for s in full]
    raise ValueError(f"unknown condition: {condition}")


def syllable_keys(s: str) -> tuple[str, bool]:
    """Keys for one syllable on the Dai Chien layout.

    Returns (symbol keys, needs_space). A syllable without a tone mark is
    finished with the space key, which is tone 1 in strict mode and an open
    tone in the fuzzy engine.
    """
    keys = []
    tone_mark = ""
    for ch in s:
        if ch in TONE_MARKS:
            tone_mark = ch
            continue
        if ch not in DACHEN_KEYS:
            raise ValueError(f"no key for symbol {ch!r} in {s!r}")
        keys.append(DACHEN_KEYS[ch])
    if tone_mark:
        keys.append(DACHEN_KEYS[tone_mark])
        return "".join(keys), False
    return "".join(keys), True

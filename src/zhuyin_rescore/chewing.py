"""Minimal ctypes wrapper around the libchewing C API (v0.13).

Signatures follow /opt/libchewing/include/chewing/chewing.h. Every function
used here has explicit argtypes and restype so pointers are never truncated.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from dataclasses import dataclass

from zhuyin_rescore.zhuyin import syllable_keys

AUTOLEARN_DISABLED = 1
SIMPLE_CONVERSION_ENGINE = 0
CHEWING_CONVERSION_ENGINE = 1
FUZZY_CHEWING_CONVERSION_ENGINE = 2
KB_DEFAULT = 0
MAX_CHI_SYMBOL_LEN = 39

# Strict engine for toned input, fuzzy engine (open tone, partial syllable
# prefix match) for the no tone and initials only conditions.
ENGINE_FOR_CONDITION = {
    "full": CHEWING_CONVERSION_ENGINE,
    "notone": FUZZY_CHEWING_CONVERSION_ENGINE,
    "initial": FUZZY_CHEWING_CONVERSION_ENGINE,
}


class _Interval(ctypes.Structure):
    _fields_ = [("from_", ctypes.c_int), ("to", ctypes.c_int)]


_CTX = ctypes.c_void_p
_SIGNATURES: dict[str, tuple[list, object]] = {
    "chewing_new2": ([ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p], _CTX),
    "chewing_delete": ([_CTX], None),
    "chewing_free": ([ctypes.c_void_p], None),
    "chewing_Reset": ([_CTX], ctypes.c_int),
    "chewing_set_KBType": ([_CTX, ctypes.c_int], ctypes.c_int),
    "chewing_set_autoLearn": ([_CTX, ctypes.c_int], None),
    "chewing_set_maxChiSymbolLen": ([_CTX, ctypes.c_int], None),
    "chewing_set_candPerPage": ([_CTX, ctypes.c_int], None),
    "chewing_set_phraseChoiceRearward": ([_CTX, ctypes.c_int], None),
    "chewing_config_has_option": ([_CTX, ctypes.c_char_p], ctypes.c_int),
    "chewing_config_get_int": ([_CTX, ctypes.c_char_p], ctypes.c_int),
    "chewing_config_set_int": ([_CTX, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
    "chewing_handle_Default": ([_CTX, ctypes.c_int], ctypes.c_int),
    "chewing_handle_Space": ([_CTX], ctypes.c_int),
    "chewing_handle_Tab": ([_CTX], ctypes.c_int),
    "chewing_handle_Esc": ([_CTX], ctypes.c_int),
    "chewing_handle_Enter": ([_CTX], ctypes.c_int),
    "chewing_handle_Home": ([_CTX], ctypes.c_int),
    "chewing_handle_End": ([_CTX], ctypes.c_int),
    "chewing_handle_Left": ([_CTX], ctypes.c_int),
    "chewing_handle_Right": ([_CTX], ctypes.c_int),
    "chewing_buffer_String_static": ([_CTX], ctypes.c_char_p),
    "chewing_buffer_Len": ([_CTX], ctypes.c_int),
    "chewing_bopomofo_String_static": ([_CTX], ctypes.c_char_p),
    "chewing_commit_String_static": ([_CTX], ctypes.c_char_p),
    "chewing_cursor_Current": ([_CTX], ctypes.c_int),
    "chewing_get_phoneSeq": ([_CTX], ctypes.POINTER(ctypes.c_ushort)),
    "chewing_get_phoneSeqLen": ([_CTX], ctypes.c_int),
    "chewing_phone_to_bopomofo": ([ctypes.c_ushort, ctypes.c_char_p, ctypes.c_ushort], ctypes.c_int),
    "chewing_interval_Enumerate": ([_CTX], None),
    "chewing_interval_hasNext": ([_CTX], ctypes.c_int),
    "chewing_interval_Get": ([_CTX, ctypes.POINTER(_Interval)], None),
    "chewing_cand_open": ([_CTX], ctypes.c_int),
    "chewing_cand_list_first": ([_CTX], ctypes.c_int),
    "chewing_cand_list_has_next": ([_CTX], ctypes.c_int),
    "chewing_cand_list_next": ([_CTX], ctypes.c_int),
    "chewing_cand_close": ([_CTX], ctypes.c_int),
    "chewing_cand_TotalChoice": ([_CTX], ctypes.c_int),
    "chewing_cand_string_by_index_static": ([_CTX, ctypes.c_int], ctypes.c_char_p),
}

_LIB: ctypes.CDLL | None = None


def load_library(name: str = "libchewing.so.3") -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(name)
        for fn_name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(lib, fn_name)
            fn.argtypes = argtypes
            fn.restype = restype
        _LIB = lib
    return _LIB


def _decode(raw: bytes | None) -> str:
    return raw.decode("utf-8") if raw else ""


@dataclass
class Conversion:
    text: str
    intervals: list[tuple[int, int]]


class Chewing:
    """One libchewing context configured for offline benchmarking.

    Learning is disabled and the user dictionary lives in a private temp
    directory, so experiments never read or update a real user dictionary.
    """

    def __init__(self, engine: int = CHEWING_CONVERSION_ENGINE, syspath: str | None = None):
        self.lib = load_library()
        syspath = syspath or os.environ.get("CHEWING_PATH", "/opt/libchewing/share/libchewing")
        self._userdir = tempfile.TemporaryDirectory(prefix="chewing-user-")
        userpath = os.path.join(self._userdir.name, "chewing.dat")
        self.ctx = self.lib.chewing_new2(syspath.encode(), userpath.encode(), None, None)
        if not self.ctx:
            raise RuntimeError(f"chewing_new2 failed for syspath {syspath}")
        self.lib.chewing_set_KBType(self.ctx, KB_DEFAULT)
        self.lib.chewing_set_autoLearn(self.ctx, AUTOLEARN_DISABLED)
        self.lib.chewing_set_maxChiSymbolLen(self.ctx, MAX_CHI_SYMBOL_LEN)
        self.lib.chewing_set_candPerPage(self.ctx, 10)
        self.set_int("chewing.disable_auto_learn_phrase", 1)
        self.set_engine(engine)

    def close(self) -> None:
        if self.ctx:
            self.lib.chewing_delete(self.ctx)
            self.ctx = None
        self._userdir.cleanup()

    def __enter__(self) -> Chewing:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def set_int(self, name: str, value: int) -> None:
        if not self.lib.chewing_config_has_option(self.ctx, name.encode()):
            raise KeyError(f"unknown libchewing option {name}")
        if self.lib.chewing_config_set_int(self.ctx, name.encode(), value) != 0:
            raise ValueError(f"failed to set {name}={value}")

    def get_int(self, name: str) -> int:
        return self.lib.chewing_config_get_int(self.ctx, name.encode())

    def set_engine(self, engine: int) -> None:
        self.set_int("chewing.conversion_engine", engine)

    def reset(self) -> None:
        self.lib.chewing_Reset(self.ctx)

    def type_syllables(self, syllables: list[str]) -> None:
        """Type syllables (canonical form) via key simulation."""
        for syl in syllables:
            keys, needs_space = syllable_keys(syl)
            for key in keys:
                self.lib.chewing_handle_Default(self.ctx, ord(key))
            if needs_space:
                self.lib.chewing_handle_Space(self.ctx)

    def buffer(self) -> str:
        return _decode(self.lib.chewing_buffer_String_static(self.ctx))

    def bopomofo_pending(self) -> str:
        return _decode(self.lib.chewing_bopomofo_String_static(self.ctx))

    def commit_string(self) -> str:
        return _decode(self.lib.chewing_commit_String_static(self.ctx))

    def phone_seq(self) -> list[str]:
        """Parsed phone sequence as bopomofo strings, for input validation."""
        n = self.lib.chewing_get_phoneSeqLen(self.ctx)
        ptr = self.lib.chewing_get_phoneSeq(self.ctx)
        if not ptr:
            return []
        try:
            out = []
            buf = ctypes.create_string_buffer(64)
            for i in range(n):
                self.lib.chewing_phone_to_bopomofo(ptr[i], buf, len(buf))
                out.append(buf.value.decode("utf-8"))
            return out
        finally:
            self.lib.chewing_free(ptr)

    def intervals(self) -> list[tuple[int, int]]:
        out = []
        it = _Interval()
        self.lib.chewing_interval_Enumerate(self.ctx)
        while self.lib.chewing_interval_hasNext(self.ctx):
            self.lib.chewing_interval_Get(self.ctx, ctypes.byref(it))
            out.append((it.from_, it.to))
        return out

    def convert(self, syllables: list[str]) -> Conversion:
        """Reset, type the syllables and return the 1-best conversion."""
        self.reset()
        self.type_syllables(syllables)
        return Conversion(self.buffer(), self.intervals())

    def next_conversion(self) -> str:
        """Advance to the next whole sentence conversion (the Tab key).

        In libchewing v0.13 Tab at the end of the buffer cycles through the
        engine's n-best conversions of the whole preedit buffer.
        """
        self.lib.chewing_handle_Tab(self.ctx)
        return self.buffer()

    def candidates_at(self, pos: int) -> list[str]:
        """Phrase candidates for the interval starting at buffer position pos.

        Moves the cursor, opens the candidate window, reads every choice and
        closes it again. The candidate list is the longest phrase list that
        starts at the cursor.
        """
        lib = self.lib
        lib.chewing_handle_Home(self.ctx)
        for _ in range(pos):
            lib.chewing_handle_Right(self.ctx)
        if lib.chewing_cand_open(self.ctx) != 0:
            return []
        try:
            n = lib.chewing_cand_TotalChoice(self.ctx)
            return [_decode(lib.chewing_cand_string_by_index_static(self.ctx, i)) for i in range(n)]
        finally:
            lib.chewing_cand_close(self.ctx)
            lib.chewing_handle_End(self.ctx)

"""Text cleaning, clause splitting and g2pW conversion shared by data scripts."""

from __future__ import annotations

import re
from collections.abc import Iterator

HAN = "㐀-䶿一-鿿"
HAN_CHAR = re.compile(f"[{HAN}]")
PURE_HAN = re.compile(f"^[{HAN}]+$")
CLAUSE_DELIMS = re.compile(r"[，。！？；：、,.!?;:\n\r\t「」『』（）()《》〈〉【】\[\]\"'“”‘’…—～~|/／\s]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACES_RE = re.compile(r"[ \t　]+")


def han_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(HAN_CHAR.findall(text)) / len(text)


def cp950_ok(text: str) -> bool:
    try:
        text.encode("cp950")
    except UnicodeEncodeError:
        return False
    return True


def clean_paragraphs(text: str, min_chars: int = 10, min_han_ratio: float = 0.5) -> list[str]:
    """Split on newlines, drop URLs and short or mostly non-Han lines."""
    out = []
    for line in text.splitlines():
        line = SPACES_RE.sub(" ", URL_RE.sub(" ", line)).strip().lstrip("*-#>:= ")
        if len(line) >= min_chars and han_ratio(line) >= min_han_ratio:
            out.append(line)
    return out


def iter_clauses(text: str) -> Iterator[tuple[int, str]]:
    """Yield (start, clause) for every delimiter separated span."""
    pos = 0
    for m in CLAUSE_DELIMS.finditer(text):
        if m.start() > pos:
            yield pos, text[pos : m.start()]
        pos = m.end()
    if pos < len(text):
        yield pos, text[pos:]


def extract_examples(text: str, min_len: int, max_len: int, context_chars: int):
    """Yield (clause, context) for pure Han clauses within the length range."""
    for start, clause in iter_clauses(text):
        if min_len <= len(clause) <= max_len and PURE_HAN.match(clause):
            context = text[max(0, start - context_chars) : start].lstrip()
            yield clause, context


def han_clauses(text: str, min_len: int = 5, max_len: int = 20) -> Iterator[str]:
    for _, clause in iter_clauses(text):
        if min_len <= len(clause) <= max_len and PURE_HAN.match(clause):
            yield clause


_G2P = None


def _g2p_init(model_dir: str) -> None:
    global _G2P
    from g2pw import G2PWConverter

    _G2P = G2PWConverter(
        model_dir=model_dir,
        style="bopomofo",
        enable_non_tradional_chinese=False,
        batch_size=128,
        turnoff_tqdm=True,
    )
    # g2pW treats num_workers=0 as unset; force in-process data loading so it
    # can run inside a pool worker.
    _G2P.num_workers = 0


def _g2p_batch(sentences: list[str]) -> list[list[str | None]]:
    return _G2P(sentences)


def g2pw_readings(
    sentences: list[str], model_dir: str, workers: int = 8, chunk: int = 256
) -> list[list[str | None]]:
    """Run g2pW over sentences with a process pool; order is preserved."""
    import multiprocessing as mp

    chunks = [sentences[i : i + chunk] for i in range(0, len(sentences), chunk)]
    if workers <= 1:
        _g2p_init(model_dir)
        return [r for c in chunks for r in _g2p_batch(c)]
    with mp.Pool(workers, initializer=_g2p_init, initargs=(model_dir,)) as pool:
        results = pool.map(_g2p_batch, chunks)
    return [r for batch in results for r in batch]

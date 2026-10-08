import ctypes

import pytest

try:
    ctypes.CDLL("libchewing.so.3")
except OSError:
    pytest.skip("libchewing not available", allow_module_level=True)

from zhuyin_rescore.candidates import CandidateGenerator, in_charset
from zhuyin_rescore.chewing import CHEWING_CONVERSION_ENGINE, Chewing

SENT = ["ㄨㄛˇ", "ㄇㄧㄥˊ", "ㄊㄧㄢ", "ㄗㄞˋ", "ㄑㄩˋ", "ㄒㄩㄝˊ", "ㄒㄧㄠˋ"]
SENT_DE = ["ㄊㄚ", "ㄉㄜ˙", "ㄊㄡˊ", "ㄈㄚˇ", "ㄓㄤˇ", "ㄉㄜ˙", "ㄏㄣˇ", "ㄔㄤˊ"]


@pytest.fixture(scope="module")
def chewing():
    with Chewing(CHEWING_CONVERSION_ENGINE) as c:
        yield c


def test_phone_sequence_matches_input(chewing):
    conv = chewing.convert(SENT)
    assert chewing.phone_seq() == SENT
    assert len(conv.text) == len(SENT)
    assert conv.intervals[0][0] == 0 and conv.intervals[-1][1] == len(SENT)


def test_reset_clears_nbest_state(chewing):
    first = chewing.convert(SENT).text
    for _ in range(3):
        chewing.next_conversion()
    assert chewing.convert(SENT).text == first
    assert chewing.commit_string() == ""


def test_tab_nbest_wraps(chewing):
    first = chewing.convert(SENT).text
    seen = [first]
    for _ in range(40):
        text = chewing.next_conversion()
        if text == first:
            break
        seen.append(text)
    assert 1 < len(seen) < 40
    assert all(len(t) == len(SENT) for t in seen)


@pytest.mark.parametrize("condition", ["full", "notone", "initial"])
def test_generator_pool(condition):
    from zhuyin_rescore.zhuyin import derive_condition

    gen = CandidateGenerator(condition, pool_size=20)
    try:
        cset = gen.generate(derive_condition(SENT_DE, condition))
    finally:
        gen.close()
    texts = cset.texts()
    assert texts[0] == cset.onebest
    assert len(set(texts)) == len(texts)
    assert all(len(t) == len(SENT_DE) for t in texts)
    assert all(in_charset(t, "cp950") for t in texts[1:])
    if condition == "full":
        # The 1-best uses the wrong "de"; the correct sentence must be in the pool.
        assert "他的頭髮長得很長" in texts

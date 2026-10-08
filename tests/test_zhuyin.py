import pytest

from zhuyin_rescore.zhuyin import (
    derive_condition,
    initial_only,
    normalize,
    parse_syllable,
    strip_tone,
    syllable_keys,
)


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("ㄒㄧㄤ4", "ㄒㄧㄤˋ"),
        ("ㄊㄧㄢ1", "ㄊㄧㄢ"),
        ("ㄊㄧㄢˉ", "ㄊㄧㄢ"),
        ("ㄉㄜ5", "ㄉㄜ˙"),
        ("˙ㄉㄜ", "ㄉㄜ˙"),
        ("ㄉㄜ˙", "ㄉㄜ˙"),
        ("ㄓ1", "ㄓ"),
        ("ㄦˊ", "ㄦˊ"),
        ("ㄩㄝˋ", "ㄩㄝˋ"),
    ],
)
def test_normalize(raw, canonical):
    assert normalize(raw) == canonical


@pytest.mark.parametrize("bad", ["", "abc", "ㄒㄒ", "ㄚㄅ"])
def test_parse_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_syllable(bad)


def test_conditions():
    full = ["ㄨㄛˇ", "ㄧㄠˋ", "ㄓ", "ㄉㄜ˙"]
    assert derive_condition(full, "full") == full
    assert derive_condition(full, "notone") == ["ㄨㄛ", "ㄧㄠ", "ㄓ", "ㄉㄜ"]
    assert derive_condition(full, "initial") == ["ㄨ", "ㄧ", "ㄓ", "ㄉ"]
    assert strip_tone("ㄒㄧㄤˋ") == "ㄒㄧㄤ"
    assert initial_only("ㄒㄧㄤˋ") == "ㄒ"


def test_keys_put_tone_last():
    assert syllable_keys("ㄨㄛˇ") == ("ji3", False)
    assert syllable_keys("ㄉㄜ˙") == ("2k7", False)
    assert syllable_keys("ㄊㄧㄢ") == ("wu0", True)

import pytest

from zhuyin_rescore.metrics import Accumulator, edit_distance, oracle_errors, recovered_fraction


def test_edit_distance():
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("abc", "ab") == 1
    assert edit_distance("", "abc") == 3
    assert edit_distance("kitten", "sitting") == 3


def test_accumulator_and_oracle():
    acc = Accumulator()
    acc.add("abcd", "abcd")
    acc.add("abcd", "abxd")
    assert acc.cer == 1 / 8
    assert acc.sent_acc == 0.5
    assert oracle_errors("abcd", ["axxd", "abxd"]) == 1


def test_recovered_fraction():
    assert recovered_fraction(0.10, 0.06, 0.02) == pytest.approx(0.5)
    assert recovered_fraction(0.10, 0.10, 0.10) == 0.0

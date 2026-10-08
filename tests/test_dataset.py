from pathlib import Path

from zhuyin_rescore.data import read_jsonl
from zhuyin_rescore.zhuyin import CONDITIONS, derive_condition

FIXTURE = Path(__file__).parent / "fixtures" / "news_sample.jsonl"


def test_fixture_schema():
    rows = list(read_jsonl(FIXTURE))
    assert len(rows) == 20
    for row in rows:
        n = len(row["text"])
        assert 5 <= n <= 20
        for cond in CONDITIONS:
            assert len(row[f"zhuyin_{cond}"]) == n
            assert row[f"zhuyin_{cond}"] == derive_condition(row["zhuyin_full"], cond)


def test_clause_extraction():
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from build_dataset import extract_examples

    text = "記者林曉慧報導，中職一般棒球賽是在開賽前2小時才會開放觀眾入場。下雨都不會被影響"
    got = list(extract_examples(text, 5, 20, 10))
    # The clause with a digit is dropped; context is the raw preceding text.
    assert got == [("記者林曉慧報導", ""), ("下雨都不會被影響", "時才會開放觀眾入場。")]

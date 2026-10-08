from collections import Counter

import numpy as np
import pytest

from zhuyin_rescore.ngram import BOS, EOS, UNK, CharNgram, pack

TEXTS = ["我明天再去學校", "我明天在家", "他明天再來", "我在家", "明天再說"] * 3


@pytest.fixture(scope="module")
def model():
    return CharNgram.train(TEXTS, order=4, min_char_count=1, min_count={})


def brute_counts(model, k):
    counts = Counter()
    for t in TEXTS:
        ids = [BOS] + model.ids(t) + [EOS]
        for i in range(len(ids) - k + 1):
            counts[tuple(ids[i : i + k])] += 1
    return {int(pack(np.array([key]))[0]): v for key, v in counts.items()}


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_counts_match_brute_force(model, k):
    got = dict(zip(model.keys[k - 1].tolist(), model.counts[k - 1].tolist(), strict=True))
    assert got == brute_counts(model, k)


def test_chunking_does_not_change_counts(model):
    small = CharNgram.train(TEXTS, order=4, min_char_count=1, min_count={}, chunk=7)
    for k in range(4):
        assert np.array_equal(model.keys[k], small.keys[k])
        assert np.array_equal(model.counts[k], small.counts[k])


def test_scores_rank_fluent_text_higher(model):
    good, homophone, shuffled = model.score_batch(["我明天再去學校", "我明天在去學校", "校學去再天明我"])
    assert good > homophone > shuffled


def test_batch_matches_scalar(model):
    texts = ["我明天再去學校", "我明天在去學校"]
    assert model.score_batch(texts, history="他說") == pytest.approx(
        [model.score_text(t, history="他說") for t in texts]
    )


def test_unknown_char_gets_floor(model):
    ctx = np.array([model.start_context("")], dtype=np.int64)
    assert model.logp(ctx, np.array([UNK]))[0] == model.unk_logp
    assert model.unk_logp < model.logp(ctx, np.array(model.ids("我")))[0]


def test_save_load_truncates_order(model, tmp_path):
    model.save(tmp_path / "m")
    m2 = CharNgram.load(tmp_path / "m", order=2)
    assert m2.order == 2
    assert CharNgram.load(tmp_path / "m").score_text("我在家") == pytest.approx(model.score_text("我在家"))


def test_beam_decoder_reads_lattice(model):
    try:
        from zhuyin_rescore.lexicon import Lexicon, load_entries

        entries = load_entries()
    except Exception as exc:  # chewing-cli or the dictionary may be missing
        pytest.skip(f"lexicon unavailable: {exc}")
    from zhuyin_rescore.beam import BeamDecoder

    syl = ["ㄨㄛˇ", "ㄇㄧㄥˊ", "ㄊㄧㄢ", "ㄗㄞˋ", "ㄑㄩˋ", "ㄒㄩㄝˊ", "ㄒㄧㄠˋ"]
    out = BeamDecoder(Lexicon(entries, "full"), model, beam=16).decode(syl, k=5)
    assert out[0].text == "我明天再去學校"
    assert all(len(d.text) == len(syl) for d in out)
    assert len({d.text for d in out}) == len(out)


@pytest.fixture(scope="module")
def entries():
    try:
        from zhuyin_rescore.lexicon import load_entries

        return load_entries()
    except Exception as exc:  # chewing-cli or the dictionary may be missing
        pytest.skip(f"lexicon unavailable: {exc}")


SENT = ["ㄨㄛˇ", "ㄇㄧㄥˊ", "ㄊㄧㄢ", "ㄗㄞˋ", "ㄑㄩˋ", "ㄒㄩㄝˊ", "ㄒㄧㄠˋ"]


def test_incremental_decode_equals_full(model, entries):
    from zhuyin_rescore.beam import BeamDecoder
    from zhuyin_rescore.lexicon import Lexicon

    dec = BeamDecoder(Lexicon(entries, "full"), model, beam=16)
    full = [d.text for d in dec.decode(SENT, k=10, incremental=False)]
    for j in range(1, len(SENT) + 1):
        inc = dec.decode(SENT[:j], k=10)
    assert [d.text for d in inc] == full
    assert dec.last_new_positions == 1
    # Editing the middle recomputes from the change on.
    edited = SENT[:3] + ["ㄗㄞˇ", "ㄗㄞˋ"]
    dec.decode(edited, k=10)
    assert dec.last_new_positions == 2


def test_locked_span_is_respected(model, entries):
    from zhuyin_rescore.beam import BeamDecoder
    from zhuyin_rescore.lexicon import Lexicon

    dec = BeamDecoder(Lexicon(entries, "full"), model, beam=16)
    out = dec.decode(SENT, k=10, locks=[(3, 4, "在")])
    assert out and all(d.text[3] == "在" for d in out)


def test_mixed_tones_filter_toned_positions(entries):
    from zhuyin_rescore.lexicon import Lexicon

    lex = Lexicon(entries, "mixed")
    toneless = {p for p, _ in lex.lookup_span(("ㄗㄞ",))}
    toned = {p for p, _ in lex.lookup_span(("ㄗㄞˋ",))}
    assert {"在", "再"} <= toned < toneless
    assert "災" in toneless and "災" not in toned
    assert lex.reading_matches("明天", ("ㄇㄧㄥ", "ㄊㄧㄢ")) is True
    assert lex.reading_matches("明天", ("ㄇㄧㄥˇ", "ㄊㄧㄢ")) is False
    assert lex.reading_matches("明", ("ㄇㄧㄥ",)) is None

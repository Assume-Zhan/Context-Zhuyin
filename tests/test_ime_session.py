from pathlib import Path

import pytest

from zhuyin_ime.composer import Composer

NGRAM = Path(__file__).parents[1] / "outputs" / "ngram" / "zhtw-o4"


def test_composer_slots_and_tones():
    c = Composer()
    for ch in "ji":  # medial u, then rime o
        assert c.feed_symbol(ch)
    assert c.text() == "ㄨㄛ"
    c.feed_symbol("u")  # medial i replaces medial u
    assert c.text() == "ㄧㄛ"
    assert c.backspace() and c.text() == "ㄧ"
    c.clear()
    for ch in "ji":
        c.feed_symbol(ch)
    assert c.finish("ˇ") == "ㄨㄛˇ" and c.empty()
    assert not c.feed_symbol("3")  # tone keys are not symbols


def test_composer_rejects_unknown_syllables():
    c = Composer(is_valid=lambda s: s != "ㄅㄨㄛ")
    for ch in "1ji":
        c.feed_symbol(ch)
    assert c.finish() is None and c.text() == "ㄅㄨㄛ"


@pytest.fixture(scope="module")
def engine():
    if not NGRAM.exists():
        pytest.skip("n-gram model not built (scripts/train_ngram.py)")
    try:
        from zhuyin_rescore.lexicon import Lexicon, load_entries

        entries = load_entries()
    except Exception as exc:
        pytest.skip(f"lexicon unavailable: {exc}")
    from zhuyin_ime.session import Engine
    from zhuyin_rescore.ngram import CharNgram

    return Engine(Lexicon(entries, "mixed"), CharNgram.load(NGRAM))


def type_keys(session, keys: str):
    from zhuyin_ime.session import Key

    st = None
    for ch in keys:
        st = session.process_key(Key(char=ch))
    return st


def test_typing_and_commit(engine):
    from zhuyin_ime.session import Key

    s = engine.new_session()
    # wo3 ming2 tian1 zai4 qu4 xue2 xiao4 on the Dai Chien layout
    st = type_keys(s, "ji3au/6wu0 y94fm4vm,6vul4")
    assert st.preedit == "我明天再去學校"
    st = s.process_key(Key(name="Return"))
    assert st.commit == "我明天再去學校" and st.preedit == ""
    assert s.history.endswith("我明天再去學校\n")


def test_toneless_typing(engine):
    s = engine.new_session()
    st = type_keys(s, "ji au/ wu0 y9 fm vm, vul ")
    assert st.preedit == "我明天再去學校"


def test_composing_shows_bopomofo_and_backspace(engine):
    from zhuyin_ime.session import Key

    s = engine.new_session()
    st = type_keys(s, "ji3a")
    assert st.preedit == "我ㄇ"
    st = s.process_key(Key(name="BackSpace"))
    assert st.preedit == "我"
    st = s.process_key(Key(name="BackSpace"))
    assert st.preedit == "" and st.handled
    st = s.process_key(Key(name="BackSpace"))
    assert not st.handled  # nothing left: let the application handle it


def test_candidate_window_locks_choice(engine):
    from zhuyin_ime.session import Key

    s = engine.new_session()
    type_keys(s, "ji3au/6wu0 y94")  # wo ming tian zai
    st = s.process_key(Key(name="Down"))  # cursor at end: phrases ending there
    assert st.candidates
    target = "在" if "在" in st.candidates else st.candidates[-1]
    while target not in st.candidates:
        st = s.process_key(Key(name="Down"))
    st = s.process_key(Key(char=str(st.candidates.index(target) + 1)))
    assert st.preedit.endswith(target) and not st.candidates


def test_punctuation_commits(engine):
    from zhuyin_ime.session import Key

    s = engine.new_session()
    type_keys(s, "ji3")
    st = s.process_key(Key(char="<", shift=True))
    assert st.commit == "我，"


def test_rerank_applies_only_to_current_version(engine):
    s = engine.new_session()
    type_keys(s, "ji au/ wu0 ")
    req = s.rerank_request()
    assert req is not None
    version, _, kbest, syllables = req
    assert len(syllables) == 3
    scores = [0.0] * len(kbest)
    scores[1] = 1000.0  # force the second candidate
    st = s.apply_rerank(version, scores)
    assert st is not None and st.preedit == kbest[1].text
    type_keys(s, "y9 ")
    assert s.apply_rerank(version, scores) is None  # stale

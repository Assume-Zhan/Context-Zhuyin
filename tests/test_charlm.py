import math

import pytest
import torch

from zhuyin_rescore.charlm import (
    BOS,
    SEP,
    UNK,
    CharLM,
    CharLMConfig,
    CharLMScorer,
    CharVocab,
    save_charlm,
)

CHARS = list("我明天再在去學校他的頭髮長得很")


def test_vocab_normalization():
    v = CharVocab(CHARS)
    ids = v.encode("我  明天\n\nabc，再")
    toks = [v.tokens[i] for i in ids]
    assert toks == ["我", "<sep>", "明", "天", "<sep>", "<unk>", "，", "再"]
    assert v.encode("") == []
    assert UNK in v.encode("xyz") and len(v.encode("xyz")) == 1
    assert v.encode(" ") == [SEP]


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    torch.manual_seed(0)
    vocab = CharVocab(CHARS)
    cfg = CharLMConfig(vocab_size=len(vocab), d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=64)
    model = CharLM(cfg)
    path = tmp_path_factory.mktemp("charlm")
    save_charlm(model, vocab, path)
    table = {"ㄗㄞˋ": ["再", "在"], "ㄇㄧㄥˊ": ["明"]}
    return CharLMScorer(path, homophones=lambda s: table.get(s, []))


def reference(scorer, context, cand):
    ids = scorer.context_ids(context) + scorer.vocab.encode(cand)
    x = torch.tensor([ids])
    with torch.inference_mode():
        lp = torch.log_softmax(scorer.model(x), dim=-1)[0]
    p = len(scorer.context_ids(context))
    return sum(lp[t - 1, ids[t]].item() for t in range(p, len(ids)))


@pytest.mark.parametrize("context", ["他說，", ""])
def test_cached_scores_match_full_forward(scorer, context):
    cands = ["我明天再去", "我明天在去", "他明天再去"]
    got = scorer.score_cached(context, cands)
    assert got == pytest.approx([reference(scorer, context, c) for c in cands], abs=1e-4)


def test_homophone_normalization(scorer):
    syl = ["ㄨㄛˇ", "ㄇㄧㄥˊ", "ㄊㄧㄢ", "ㄗㄞˋ"]
    a, b = scorer.score_homophone("", ["我明天再", "我明天在"], syl)
    # Only position 4 differs; over its two homophones the probabilities sum to one.
    assert math.exp(a) + math.exp(b) == pytest.approx(
        math.exp(a) / math.exp(scorer.score_homophone("", ["我明天"], syl[:3])[0])
        + math.exp(b) / math.exp(scorer.score_homophone("", ["我明天"], syl[:3])[0]),
        rel=1e-4,
    )
    # Positions with a single homophone contribute log 1 = 0.
    single = scorer.score_homophone("", ["明"], ["ㄇㄧㄥˊ"])[0]
    assert single == pytest.approx(0.0, abs=1e-5)


def test_bos_is_first(scorer):
    assert scorer.context_ids("")[0] == BOS

import pytest
import torch

from zhuyin_rescore.charlm import CharLM, CharLMConfig, CharLMScorer, CharVocab, save_charlm
from zhuyin_rescore.listwise import Fusion, PoolExample, candidate_logprobs, collate, pool_losses

CHARS = list("我明天再在去學校他的頭髮長得很說")


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    torch.manual_seed(0)
    vocab = CharVocab(CHARS)
    cfg = CharLMConfig(vocab_size=len(vocab), d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=96)
    path = tmp_path_factory.mktemp("charlm")
    save_charlm(CharLM(cfg), vocab, path)
    return CharLMScorer(path)


EXAMPLES = [
    # Context with whitespace, unknown runs and more than context_chars ids.
    ("他說，\n\n我 xyz 很" + "學校的頭髮" * 16, "我明天再去", ["我明天在去", "我明天再去", "他明天再去"]),
    ("", "頭髮長得很", ["頭髮長的很", "頭髮長得很"]),
    ("他的", "去學校", ["去學效", "去學笑"]),  # correct sentence not in the pool
]


def errors(gold: str, texts: list[str]) -> list[int]:
    return [sum(a != b for a, b in zip(gold, t, strict=True)) for t in texts]


def pools() -> list[PoolExample]:
    return [PoolExample(ctx, gold, ts, [0.0] * len(ts), errors(gold, ts), 0) for ctx, gold, ts in EXAMPLES]


def test_training_rows_match_inference_scores(scorer):
    batch = collate(scorer.vocab, pools(), scorer.context_chars, "cpu")
    with torch.no_grad():
        lm = candidate_logprobs(scorer.model, batch)
    for (ctx, _, texts), (start, n) in zip(EXAMPLES, batch.pools, strict=True):
        assert lm[start : start + n].tolist() == pytest.approx(scorer.score_cached(ctx, texts), abs=1e-3)
    # The extra row for the correct sentence that the pool lacks.
    ctx, gold, _ = EXAMPLES[2]
    assert batch.gold_in_pool == [1, 1, -1]
    assert lm[batch.gold_rows[2]].item() == pytest.approx(scorer.score_cached(ctx, [gold])[0], abs=1e-3)


def test_losses_and_gradients(scorer):
    batch = collate(scorer.vocab, pools(), scorer.context_chars, "cpu")
    fusion = Fusion({"toned+chewing": (0.3, 1.0, 0.1), "open": (0.75, 0.5, 0.05)})
    lm = candidate_logprobs(scorer.model, batch)
    out = pool_losses(lm, batch, fusion)
    loss = out["mwer"] + out["ce"] + 0.1 * out["anchor"]
    loss.backward()
    assert fusion.weights.grad is not None and fusion.weights.grad[0].abs().sum() > 0
    assert fusion.weights.grad[1].abs().sum() == 0  # no "open" pool in the batch
    assert out["anchor"].item() > 0

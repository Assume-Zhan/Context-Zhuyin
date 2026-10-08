import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from huggingface_hub import try_to_load_from_cache  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
if not isinstance(try_to_load_from_cache(MODEL, "config.json"), str):
    pytest.skip(f"{MODEL} not in the HF cache", allow_module_level=True)

from zhuyin_rescore.scorer import LMScorer  # noqa: E402

CONTEXT = "記者林曉慧報導，中職一般棒球賽是在開賽前2小時才會開放觀眾入場，"
CANDS = ["他的頭髮長得很長", "他的頭髮長的很長", "她的頭髮長得很長", "他的投法長得很長"]


@pytest.fixture(scope="module")
def scorer():
    torch.manual_seed(0)
    return LMScorer(MODEL, dtype="float32")


def full_sequence_logprob(scorer, context, cand):
    """Slowest reference: plain forward with full vocab logits."""
    prefix = scorer.prefix_ids(context)
    ids = torch.tensor([prefix + scorer.encode(cand)], device=scorer.device)
    with torch.inference_mode():
        lp = torch.log_softmax(scorer.model(ids).logits.float(), -1)[0]
    return sum(lp[t - 1, ids[0, t]].item() for t in range(len(prefix), ids.shape[1]))


def test_batched_equals_unbatched(scorer):
    batched = scorer.score_reference(CONTEXT, CANDS)
    single = [full_sequence_logprob(scorer, CONTEXT, c) for c in CANDS]
    assert batched == pytest.approx(single, abs=1e-3)


@pytest.mark.parametrize("context", [CONTEXT, ""])
def test_cached_equals_reference(scorer, context):
    ref = scorer.score_reference(context, CANDS)
    assert scorer.score_cached(context, CANDS) == pytest.approx(ref, abs=1e-3)


def test_cache_reuse(scorer):
    cache = scorer.context_cache(CONTEXT)
    first = scorer.score_cached(CONTEXT, CANDS, cache=cache)
    second = scorer.score_cached(CONTEXT, CANDS[:2], cache=cache)
    assert second == pytest.approx(first[:2], abs=1e-3)


def test_context_truncation(scorer):
    long_ctx = "今天天氣很好，" * 50
    assert len(scorer.prefix_ids(long_ctx)) == scorer.max_context_tokens + 1


def test_prefers_correct_homophone(scorer):
    scores = scorer.score_reference(CONTEXT, CANDS)
    assert scores.index(max(scores)) == 0


def test_graph_scorer_matches_eager():
    from zhuyin_rescore.graph_scorer import GraphScorer

    s16 = LMScorer(MODEL, dtype="float16")
    g = GraphScorer(s16)
    for context in (CONTEXT, ""):
        ref = s16.score_cached(context, CANDS)
        assert g.score(context, CANDS) == pytest.approx(ref, abs=0.1)
    # A request larger than the static shapes falls back to the eager path.
    many = CANDS * 5
    assert g.score(CONTEXT, many) == pytest.approx(s16.score_cached(CONTEXT, many), abs=1e-6)
    assert g.fallbacks == 1

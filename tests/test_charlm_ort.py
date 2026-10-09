import sys

import pytest
import torch

from zhuyin_rescore.charlm import CharLM, CharLMConfig, CharLMScorer, CharVocab, save_charlm

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

CHARS = list("我明天再在去學校他的頭髮長得很說")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from zhuyin_rescore.charlm_ort import export_onnx

    torch.manual_seed(0)
    vocab = CharVocab(CHARS)
    cfg = CharLMConfig(vocab_size=len(vocab), d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=96)
    path = tmp_path_factory.mktemp("charlm")
    model = CharLM(cfg)
    with torch.no_grad():  # random init is near uniform; make scores differ
        for p in model.parameters():
            p.mul_(8.0)
    save_charlm(model, vocab, path)
    export_onnx(path, int8=True)
    return path


@pytest.mark.parametrize("context", ["", "他說，", "他說，\n\n我 xyz 很" + "學校的頭髮" * 16])
def test_onnx_scores_match_torch(model_dir, context):
    from zhuyin_rescore.charlm_ort import OrtCharLMScorer

    cands = ["我明天再去", "我明天在去", "他明天再去", "我很去學校"]
    ref = CharLMScorer(model_dir).score_cached(context, cands)
    assert OrtCharLMScorer(model_dir, int8=False).score_cached(context, cands) == pytest.approx(ref, abs=1e-3)
    assert max(ref) - min(ref) > 0.1
    # int8 runs; its agreement on real pools is measured (bench_cpu_charlm.py), not tested here.
    got8 = OrtCharLMScorer(model_dir, int8=True).score_cached(context, cands)
    assert len(got8) == len(cands) and all(x < 0 for x in got8)


def test_onnx_scorer_does_not_need_torch(model_dir):
    import subprocess

    code = (
        "import sys; from zhuyin_rescore.charlm_ort import OrtCharLMScorer;"
        f"s = OrtCharLMScorer({str(model_dir)!r}); s.score_cached('他', ['我明天', '我名天']);"
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

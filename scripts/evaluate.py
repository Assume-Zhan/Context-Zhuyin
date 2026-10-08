"""Compare libchewing 1-best with LM rescoring on the zhuyin benchmark.

Fusion: score_i = lm_i - mu * rank_i - beta * [i > 0], where rank_i is the
position in the libchewing pool (0 is the 1-best). mu and beta are tuned on
dev only and then applied to test. mu = beta = 0 is pure LM selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from zhuyin_rescore.data import read_jsonl
from zhuyin_rescore.metrics import Accumulator, edit_distance, recovered_fraction
from zhuyin_rescore.zhuyin import CONDITIONS

MU_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
BETA_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
FOCUS_GROUPS = {
    "zai": "在再",
    "de": "的得地",
    "ta": "他她它",
    "li": "裡裏",
    "zhe": "著着",
}


def load(cand_dir: Path, score_dir: Path, split: str, cond: str) -> list[dict]:
    scores = {r["id"]: r for r in read_jsonl(score_dir / f"{split}.{cond}.jsonl")}
    rows = []
    for row in read_jsonl(cand_dir / f"{split}.{cond}.jsonl"):
        s = scores[row["id"]]
        row["lm"], row["lm_noctx"] = s["lm"], s["lm_noctx"]
        row["texts"] = [c["text"] for c in row["candidates"]]
        rows.append(row)
    return rows


def pick(row: dict, k: int, mu: float, beta: float, field: str) -> str:
    lm = np.asarray(row[field][:k])
    rank = np.arange(len(lm))
    fused = lm - mu * rank - beta * (rank > 0)
    return row["texts"][int(np.argmax(fused))]


def errors(rows: list[dict], hyps: list[str]) -> np.ndarray:
    return np.array([edit_distance(r["text"], h) for r, h in zip(rows, hyps, strict=True)])


def summarize(rows: list[dict], errs: np.ndarray) -> dict:
    acc = Accumulator()
    for r, e in zip(rows, errs, strict=True):
        acc.add_errors(r["text"], int(e))
    return acc.summary()


def tune(rows: list[dict], k: int, field: str) -> tuple[float, float, float]:
    best = None
    for mu, beta in itertools.product(MU_GRID, BETA_GRID):
        e = errors(rows, [pick(r, k, mu, beta, field) for r in rows]).sum()
        if best is None or e < best[0]:
            best = (e, mu, beta)
    _, mu, beta = best
    return mu, beta, best[0]


def bootstrap_ci(rows: list[dict], base: np.ndarray, sys: np.ndarray, n: int = 2000, seed: int = 0):
    """95% CI of the relative CER reduction, paired bootstrap over sentences."""
    rng = np.random.default_rng(seed)
    rel = []
    for _ in range(n):
        idx = rng.integers(0, len(rows), len(rows))
        b, s = base[idx].sum(), sys[idx].sum()
        rel.append((b - s) / b if b else 0.0)
    return float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))


def focus_breakdown(rows: list[dict], hyps: list[str]) -> dict:
    out = {}
    for name, chars in FOCUS_GROUPS.items():
        total = err_base = err_sys = 0
        for r, h in zip(rows, hyps, strict=True):
            gold, base = r["text"], r["onebest"]
            if len(base) != len(gold) or len(h) != len(gold):
                continue
            for i, c in enumerate(gold):
                if c in chars:
                    total += 1
                    err_base += base[i] != c
                    err_sys += h[i] != c
        out[name] = {"gold_count": total, "base_errors": err_base, "sys_errors": err_sys}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="scores subdir under outputs/scores")
    ap.add_argument("--cand-dir", default="outputs/candidates")
    ap.add_argument("--score-dir", default="outputs/scores")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 30])
    ap.add_argument("--out", default="outputs/reports")
    args = ap.parse_args()

    cand_dir = Path(args.cand_dir)
    score_dir = Path(args.score_dir) / args.tag
    meta = json.loads((score_dir / "meta.json").read_text()) if (score_dir / "meta.json").exists() else {}
    report = {"tag": args.tag, "meta": meta, "conditions": {}}
    lines = [
        f"Model: {meta.get('model', args.tag)} ({meta.get('dtype', '?')}), GPU: {meta.get('gpu', '?')}. "
        "Fusion weights tuned on dev, numbers on test.",
        "",
        "| condition | system | k | mu | beta | CER | rel. CER reduction (95% CI) | SentAcc "
        "| recovered vs Oracle@k | fixed / broken sents |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cond in args.conditions:
        dev = load(cand_dir, score_dir, "dev", cond)
        test = load(cand_dir, score_dir, "test", cond)
        base_err = errors(test, [r["onebest"] for r in test])
        base = summarize(test, base_err)
        cond_rep = {"baseline": base, "systems": {}}
        lines.append(
            f"| {cond} | libchewing 1-best | - | - | - | {base['cer']:.4f} | - | "
            f"{base['sent_acc']:.3f} | - | - |"
        )
        for k in args.ks:
            orc_err = np.array([min(edit_distance(r["text"], t) for t in r["texts"][:k]) for r in test])
            orc = summarize(test, orc_err)
            cond_rep["systems"][f"oracle@{k}"] = orc
            lines.append(
                f"| {cond} | Oracle@{k} | {k} | - | - | {orc['cer']:.4f} | "
                f"{(base['cer'] - orc['cer']) / base['cer']:.1%} | {orc['sent_acc']:.3f} | 100% | - |"
            )
            variants = [
                ("LM", "lm", (0.0, 0.0)),
                ("LM + fusion", "lm", None),
                ("LM no ctx + fusion", "lm_noctx", None),
            ]
            for name, field, fixed in variants:
                mu, beta = fixed if fixed is not None else tune(dev, k, field)[:2]
                hyps = [pick(r, k, mu, beta, field) for r in test]
                err = errors(test, hyps)
                res = summarize(test, err)
                lo, hi = bootstrap_ci(test, base_err, err)
                rel = (base["cer"] - res["cer"]) / base["cer"]
                rec = recovered_fraction(base["cer"], res["cer"], orc["cer"])
                fixed_n = int(((base_err > 0) & (err == 0)).sum())
                broken_n = int(((base_err == 0) & (err > 0)).sum())
                res.update(
                    {
                        "mu": mu,
                        "beta": beta,
                        "rel_cer_reduction": rel,
                        "rel_ci95": [lo, hi],
                        "recovered": rec,
                        "fixed": fixed_n,
                        "broken": broken_n,
                    }
                )
                if name == "LM + fusion":
                    res["focus"] = focus_breakdown(test, hyps)
                cond_rep["systems"][f"{name}@{k}"] = res
                lines.append(
                    f"| {cond} | {name} | {k} | {mu:g} | {beta:g} | {res['cer']:.4f} | "
                    f"{rel:.1%} ({lo:.1%} to {hi:.1%}) | {res['sent_acc']:.3f} | {rec:.1%} | "
                    f"{fixed_n} / {broken_n} |"
                )
        report["conditions"][cond] = cond_rep

    lines += [
        "",
        "Focus characters (test, LM + fusion at the largest k): gold count, 1-best errors -> LM errors",
        "",
    ]
    lines.append("| condition | " + " | ".join(FOCUS_GROUPS) + " |")
    lines.append("| --- |" + " --- |" * len(FOCUS_GROUPS))
    kmax = max(args.ks)
    for cond, cond_rep in report["conditions"].items():
        focus = cond_rep["systems"][f"LM + fusion@{kmax}"]["focus"]
        cells = [f"{f['gold_count']}: {f['base_errors']} -> {f['sys_errors']}" for f in focus.values()]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval.{args.tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / f"eval.{args.tag}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()

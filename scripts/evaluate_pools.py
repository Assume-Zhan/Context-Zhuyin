"""Multi-domain evaluation of decoders and LM rescoring over candidate pools.

Systems per input condition:
- 1-best baselines: libchewing 0.13.1 (unigram), libchewing 0.14 (word
  bigram), and the lattice beam decoder with char n-gram order 2, 3 and 4.
- Oracle@k of each candidate pool (upper bound of any reranker).
- LM rescoring of each pool with fusion
  score = lm + a * ngram - beta * [not first in pool] - mu * rank,
  where (a, beta, mu) is tuned on the dev sets of all domains together and
  frozen for test.

Relative CER reduction is against libchewing 0.14, the current engine.
Scores come from scripts/score_union.py, candidates from gen_candidates.py
and gen_beam.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from zhuyin_rescore.data import read_jsonl
from zhuyin_rescore.metrics import edit_distance
from zhuyin_rescore.zhuyin import CONDITIONS

A_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
BETA_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
MU_GRID = [0.0, 0.05, 0.1]
BASELINES = {
    "libchewing 0.13.1 (unigram)": "chewing-0.13",
    "libchewing 0.14 (word bigram)": "chewing-0.14",
    "beam char 2-gram": "beam-o2",
    "beam char 3-gram": "beam-o3",
    "beam char 4-gram": "beam-o4",
    # Same decoder, n-gram trained on PTT and Wikipedia only: news and web are
    # then sources the n-gram never saw, like libchewing 0.14 (CC-100).
    "beam char 4-gram, PTT+wiki n-gram": "beam-o4-ptt-wiki",
}


def source_list(row: dict) -> list[str]:
    if "beam" in row:
        return [b["text"] for b in row["beam"]]
    return [c["text"] for c in row["candidates"]]


def compose(policy: str, src: dict[str, dict], k: int) -> list[str]:
    """Build a pool; the first entry is the pool's own 1-best."""
    if policy in BASELINES.values():
        seq = source_list(src[policy])
    elif policy == "c14+beam4":
        c14 = src["chewing-0.14"]
        seq = [c14["onebest"]] + c14["tab"][1:] + source_list(src["beam-o4"]) + source_list(c14)
    elif policy == "b4+c14tab":
        # Decoder 10-best first (its 1-best stays first), then libchewing's n-best.
        c14 = src["chewing-0.14"]
        seq = source_list(src["beam-o4"])[:10] + [c14["onebest"]] + c14["tab"][1:]
    elif policy == "b4+c14":
        # Decoder 10-best first, then libchewing's n-best and substitutions.
        seq = source_list(src["beam-o4"])[:10] + source_list(src["chewing-0.14"])
    elif policy == "beam4+c14":
        seq = source_list(src["beam-o4"])[:1] + [src["chewing-0.14"]["onebest"]]
        seq += source_list(src["beam-o4"])[1:] + source_list(src["chewing-0.14"])
    else:
        raise ValueError(policy)
    out, seen = [], set()
    for t in seq:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) == k:
            break
    return out


LM_FIELD = "lm"


def load(cand_root: Path, score_dir: Path, sources: list[str], domain: str, split: str, cond: str):
    rows: dict[str, dict] = {}
    for s in sources:
        path = cand_root / s / domain / f"{split}.{cond}.jsonl"
        for r in read_jsonl(path):
            rows.setdefault(r["id"], {"text": r["text"], "src": {}})["src"][s] = r
    feats = {}
    for r in read_jsonl(score_dir / domain / f"{split}.{cond}.jsonl"):
        zeros = [0.0] * len(r["texts"])
        lm, lm_noctx = r.get(LM_FIELD, zeros), r.get("lm_noctx", zeros)
        feats[r["id"]] = {t: (lm[i], lm_noctx[i], r["ngram"][i]) for i, t in enumerate(r["texts"])}
    for rid, row in rows.items():
        row["feats"] = feats.get(rid, {})
    return list(rows.values())


def pool_arrays(rows: list[dict], policy: str, k: int, use_ctx: bool):
    """Pad each pool to k: LM score, n-gram score, validity and errors per slot."""
    n = len(rows)
    lm = np.zeros((n, k))
    ng = np.zeros((n, k))
    valid = np.zeros((n, k), dtype=bool)
    err = np.zeros((n, k), dtype=np.int64)
    for i, r in enumerate(rows):
        pool = compose(policy, r["src"], k)
        if not pool:
            err[i, 0] = len(r["text"])
            valid[i, 0] = True
            continue
        for j, t in enumerate(pool):
            err[i, j] = edit_distance(r["text"], t)
            f = r["feats"].get(t)
            if f is not None:
                lm[i, j] = f[0] if use_ctx else f[1]
                ng[i, j] = f[2]
                valid[i, j] = True
        if not valid[i].any():
            valid[i, 0] = True
    return lm, ng, valid, err


def select_errors(arrays, a: float, beta: float, mu: float) -> np.ndarray:
    lm, ng, valid, err = arrays
    rank = np.arange(lm.shape[1])
    score = lm + a * ng - beta * (rank > 0) - mu * rank
    idx = np.argmax(np.where(valid, score, -np.inf), axis=1)
    return err[np.arange(len(err)), idx]


def first(pool: list[str]) -> str:
    return pool[0] if pool else ""


def oracle(row: dict, pool: list[str]) -> int:
    return min((edit_distance(row["text"], t) for t in pool), default=len(row["text"]))


def errors(rows: list[dict], hyps: list[str]) -> np.ndarray:
    return np.array([edit_distance(r["text"], h) for r, h in zip(rows, hyps, strict=True)])


def cer(rows: list[dict], errs: np.ndarray) -> float:
    return float(errs.sum() / sum(len(r["text"]) for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="scores subdir under outputs/scores")
    ap.add_argument("--cand-root", default="outputs/cand")
    ap.add_argument("--score-root", default="outputs/scores")
    ap.add_argument("--domains", nargs="+", default=["news", "web", "ptt", "wiki", "cv"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--sources", nargs="+", default=list(BASELINES.values()))
    ap.add_argument("--pools", nargs="+", default=["chewing-0.14", "beam-o4", "c14+beam4"])
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 30])
    ap.add_argument("--out", default="outputs/reports")
    ap.add_argument("--save-errors", action="store_true", help="also save per example errors (npz)")
    ap.add_argument("--lm-field", default="lm", help="score column for the LM feature (lm or lm_homo)")
    ap.add_argument(
        "--fusion-json",
        default=None,
        help="also report rows with these fixed fusion weights (fusion.json of scripts/train_rerank.py)",
    )
    args = ap.parse_args()
    global LM_FIELD
    LM_FIELD = args.lm_field
    report_tag = args.tag if args.lm_field == "lm" else f"{args.tag}.{args.lm_field}"
    all_errors: dict[str, np.ndarray] = {}
    learned: dict[str, tuple] = {}
    if args.fusion_json:
        # Pool kind per condition, as in scripts/build_rerank_pools.py.
        raw = json.loads(Path(args.fusion_json).read_text())
        learned = {"full": tuple(raw["toned+chewing"]), "notone": tuple(raw["open"])}

    score_dir = Path(args.score_root) / args.tag
    meta = json.loads((score_dir / "meta.json").read_text()) if (score_dir / "meta.json").exists() else {}
    report: dict = {"tag": args.tag, "meta": meta, "conditions": {}}
    lines = [
        f"Reranker: {meta.get('model', args.tag)} ({meta.get('dtype', '?')}), GPU: {meta.get('gpu', '?')}. "
        "Test CER per domain; fusion tuned on all dev sets; rel. = relative CER change vs libchewing 0.14.",
        "",
    ]
    for cond in args.conditions:
        dev = {d: load(Path(args.cand_root), score_dir, args.sources, d, "dev", cond) for d in args.domains}
        test = {d: load(Path(args.cand_root), score_dir, args.sources, d, "test", cond) for d in args.domains}
        systems: dict[str, dict[str, np.ndarray]] = {}
        params: dict[str, tuple] = {}
        for name, src in BASELINES.items():
            if src in args.sources:
                systems[name] = {
                    d: errors(rs, [first(compose(src, r["src"], 1)) for r in rs]) for d, rs in test.items()
                }
        for policy, k in itertools.product(args.pools, args.ks):
            systems[f"Oracle@{k} {policy}"] = {
                d: np.array([oracle(r, compose(policy, r["src"], k)) for r in rs]) for d, rs in test.items()
            }
            for label, use_ctx in (("LM rerank", True), ("LM rerank, no context", False)):
                dev_arrays = [pool_arrays(rs, policy, k, use_ctx) for rs in dev.values()]
                best = None
                for a, beta, mu in itertools.product(A_GRID, BETA_GRID, MU_GRID):
                    e = sum(int(select_errors(arr, a, beta, mu).sum()) for arr in dev_arrays)
                    if best is None or e < best[0]:
                        best = (e, a, beta, mu)
                _, a, beta, mu = best
                name = f"{label} {policy}@{k}"
                params[name] = (a, beta, mu)
                test_arrays = {d: pool_arrays(rs, policy, k, use_ctx) for d, rs in test.items()}
                systems[name] = {d: select_errors(arr, a, beta, mu) for d, arr in test_arrays.items()}
                if use_ctx and cond in learned:
                    name = f"LM rerank, learned fusion {policy}@{k}"
                    params[name] = learned[cond]
                    systems[name] = {d: select_errors(arr, *learned[cond]) for d, arr in test_arrays.items()}
        base = systems["libchewing 0.14 (word bigram)"]
        cond_rep = {}
        header = "| system | " + " | ".join(args.domains) + " | mean | rel. | fixed / broken |"
        lines += [f"### {cond}", "", header]
        lines.append("| --- |" + " --- |" * (len(args.domains) + 3))
        for name, errs in systems.items():
            cers = {d: cer(test[d], errs[d]) for d in args.domains}
            mean = float(np.mean(list(cers.values())))
            base_mean = float(np.mean([cer(test[d], base[d]) for d in args.domains]))
            fixed = sum(int(((base[d] > 0) & (errs[d] == 0)).sum()) for d in args.domains)
            broken = sum(int(((base[d] == 0) & (errs[d] > 0)).sum()) for d in args.domains)
            sent_acc = {d: float((errs[d] == 0).mean()) for d in args.domains}
            cond_rep[name] = {
                "cer": cers,
                "mean_cer": mean,
                "sent_acc": sent_acc,
                "rel_vs_c14": (mean - base_mean) / base_mean,
                "fixed": fixed,
                "broken": broken,
                "params": params.get(name),
            }
            p = ""
            if name in params:
                a, beta, mu = params[name]
                p = f" (a={a:g}, beta={beta:g}, mu={mu:g})"
            lines.append(
                f"| {name}{p} | "
                + " | ".join(f"{cers[d]:.2%}" for d in args.domains)
                + f" | {mean:.2%} | {(mean - base_mean) / base_mean:+.1%} | {fixed} / {broken} |"
            )
        lines.append("")
        report["conditions"][cond] = cond_rep
        for name, errs in systems.items():
            for d in args.domains:
                all_errors[f"{cond}|{name}|{d}"] = errs[d]
        print("\n".join(lines[-(len(systems) + 4) :]), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"pools.{report_tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / f"pools.{report_tag}.md").write_text("\n".join(lines) + "\n")
    if args.save_errors:
        np.savez_compressed(out / f"pools.{report_tag}.errors.npz", **all_errors)


if __name__ == "__main__":
    main()

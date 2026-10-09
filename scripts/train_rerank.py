"""Fine-tune the character LM to pick the correct sentence from its pool.

Starts from a pretrained char LM (scripts/train_charlm.py) and trains on
the pools of scripts/build_rerank_pools.py with the listwise loss of
src/zhuyin_rescore/listwise.py: expected error (MWER) plus cross entropy on
the correct sentence plus an LM anchor, with the fusion weights (a, beta,
mu) learned per pool kind. The dev pools are evaluated with the learned
fusion every --eval-every steps (fp32); the best checkpoint is kept, with
its weights in fusion.json next to the model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from zhuyin_ime.pool import FUSION_PRESETS
from zhuyin_rescore.charlm import load_charlm, save_charlm
from zhuyin_rescore.data import read_jsonl
from zhuyin_rescore.listwise import POOL_KINDS, Fusion, PoolExample, candidate_logprobs, collate, pool_losses


def load_pools(path: Path) -> list[PoolExample]:
    return [
        PoolExample(r["context"], r["text"], r["texts"], r["ngram"], r["errors"], POOL_KINDS.index(r["kind"]))
        for r in read_jsonl(path)
    ]


def batches(examples: list[PoolExample], rows: int, rng: random.Random | None):
    """Pack pools into batches of about `rows` candidate rows."""
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    cur, n = [], 0
    for i in order:
        cur.append(examples[i])
        n += len(examples[i].texts) + 1
        if n >= rows:
            yield cur
            cur, n = [], 0
    if cur:
        yield cur


A_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
BETA_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
MU_GRID = [0.0, 0.05, 0.1]


def grid_errors(lm: np.ndarray, ng: np.ndarray, valid: np.ndarray, err: np.ndarray) -> tuple[int, tuple]:
    """Fewest total errors over the fusion grid of scripts/evaluate_pools.py."""
    rank = np.arange(lm.shape[1])
    best = None
    for a in A_GRID:
        base = lm + a * ng
        for beta in BETA_GRID:
            for mu in MU_GRID:
                score = np.where(valid, base - beta * (rank > 0) - mu * rank, -np.inf)
                e = int(err[np.arange(len(err)), score.argmax(1)].sum())
                if best is None or e < best[0]:
                    best = (e, (a, beta, mu))
    return best


@torch.no_grad()
def evaluate(model, vocab, fusion, examples, context_chars, device, rows: int) -> dict:
    """Dev CER per pool kind with the learned fusion and with the best grid
    fusion (separates a better LM from drifting fusion weights), plus the
    anchor NLL."""
    model.eval()
    n_kinds = len(POOL_KINDS)
    err, chars, first = np.zeros(n_kinds), np.zeros(n_kinds), np.zeros(n_kinds)
    pools: list[list[tuple]] = [[] for _ in POOL_KINDS]
    nll, n_pools = 0.0, 0
    for group in batches(examples, rows, None):
        batch = collate(vocab, group, context_chars, device)
        lm = candidate_logprobs(model, batch)
        for i, (start, n) in enumerate(batch.pools):
            ex = group[i]
            score = fusion(lm[start : start + n], batch.ngram[i], ex.kind)
            err[ex.kind] += ex.errors[int(score.argmax())]
            first[ex.kind] += ex.errors[0]
            chars[ex.kind] += len(ex.gold)
            pools[ex.kind].append((lm[start : start + n].cpu().numpy(), ex.ngram, ex.errors))
            nll += float(-lm[batch.gold_rows[i]] / len(ex.gold))
            n_pools += 1
    model.train()
    out = {f"cer_{k}": float(err[i] / chars[i]) for i, k in enumerate(POOL_KINDS)}
    out.update({f"cer1best_{k}": float(first[i] / chars[i]) for i, k in enumerate(POOL_KINDS)})
    for i, k in enumerate(POOL_KINDS):
        width = max(len(p[2]) for p in pools[i])
        lm_a, ng_a = np.zeros((len(pools[i]), width)), np.zeros((len(pools[i]), width))
        valid = np.zeros((len(pools[i]), width), dtype=bool)
        err_a = np.zeros((len(pools[i]), width), dtype=np.int64)
        for j, (x, g, e) in enumerate(pools[i]):
            lm_a[j, : len(e)], ng_a[j, : len(e)], err_a[j, : len(e)] = x, g, e
            valid[j, : len(e)] = True
        e, params = grid_errors(lm_a, ng_a, valid, err_a)
        out[f"cer_grid_{k}"] = float(e / chars[i])
        out[f"grid_{k}"] = params
    out["cer_mean"] = float(np.mean([out[f"cer_{k}"] for k in POOL_KINDS]))
    out["cer_grid_mean"] = float(np.mean([out[f"cer_grid_{k}"] for k in POOL_KINDS]))
    out["gold_nll"] = nll / n_pools
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="outputs/charlm/small")
    ap.add_argument("--out", default="outputs/charlm/small-rerank")
    ap.add_argument("--data-dir", default="data/rerank_train")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--rows", type=int, default=1024, help="candidate rows per batch")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--fusion-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--w-mwer", type=float, default=1.0)
    ap.add_argument("--w-ce", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--freeze-fusion", action="store_true", help="keep the preset fusion weights")
    ap.add_argument("--context-chars", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="evaluate --init on the dev pools and exit")
    ap.add_argument(
        "--select",
        default="grid",
        choices=["grid", "learned"],
        help="keep the checkpoint with the best dev CER under the grid tuned or the learned fusion",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model, vocab = load_charlm(args.init, device="cuda")
    model.train()
    context_chars = min(args.context_chars, model.cfg.max_len - 32)
    preset = FUSION_PRESETS["charlm"]
    fusion = Fusion({k: (preset[k].a, preset[k].beta, preset[k].mu) for k in POOL_KINDS}).to(device)
    fusion.weights.requires_grad_(not args.freeze_fusion)

    train = load_pools(Path(args.data_dir) / "pools.train.jsonl")
    dev = load_pools(Path(args.data_dir) / "pools.dev.jsonl")
    rows_per_epoch = sum(len(e.texts) + 1 for e in train)
    total = int(math.ceil(args.epochs * rows_per_epoch / args.rows))
    print(f"train {len(train)} pools, dev {len(dev)}, about {total} steps", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr},
    ]
    groups.append({"params": [fusion.log_scale], "weight_decay": 0.0, "lr": 1e-2})
    if not args.freeze_fusion:
        groups.append({"params": [fusion.weights], "weight_decay": 0.0, "lr": args.fusion_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = min(1.0, (step - args.warmup) / max(1, total - args.warmup))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    out = Path(args.out)
    log: list[dict] = []
    base = evaluate(model, vocab, fusion, dev, context_chars, device, args.rows)
    base["step"] = 0
    log.append(base)
    print(f"step 0 dev {json.dumps(base)}", flush=True)
    if args.eval_only:
        return
    select = "cer_grid_mean" if args.select == "grid" else "cer_mean"
    best = base[select]
    save_charlm(model, vocab, out)
    (out / "fusion.json").write_text(json.dumps(fusion.as_dict(), indent=2) + "\n")

    rng = random.Random(args.seed)
    step, t0 = 0, time.time()
    run = {"mwer": 0.0, "ce": 0.0, "anchor": 0.0, "picked_err": 0.0}
    while step < total:
        for group in batches(train, args.rows, rng):
            if step >= total:
                break
            for g, lr in zip(opt.param_groups, base_lrs, strict=True):
                g["lr"] = lr * lr_scale(step)
            batch = collate(vocab, group, context_chars, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lm = candidate_logprobs(model, batch)
            parts = pool_losses(lm, batch, fusion, args.temperature)
            loss = args.w_mwer * parts["mwer"] + args.w_ce * parts["ce"] + args.w_anchor * parts["anchor"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            for k in run:
                run[k] += float(parts[k])
            if step % 100 == 0:
                avg = {k: round(v / 100, 4) for k, v in run.items()}
                run = dict.fromkeys(run, 0.0)
                scale = [round(float(x), 3) for x in fusion.log_scale.detach().exp()]
                elapsed = time.time() - t0
                msg = f"step {step}/{total} {avg} fusion {fusion.as_dict()} scale {scale}"
                print(f"{msg} {elapsed:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == total:
                res = evaluate(model, vocab, fusion, dev, context_chars, device, args.rows)
                res["step"] = step
                res["fusion"] = fusion.as_dict()
                log.append(res)
                print(f"step {step} dev {json.dumps(res)}", flush=True)
                save_charlm(model, vocab, out / "last")
                (out / "last" / "fusion.json").write_text(json.dumps(fusion.as_dict(), indent=2) + "\n")
                if res[select] < best:
                    best = res[select]
                    save_charlm(model, vocab, out)
                    (out / "fusion.json").write_text(json.dumps(fusion.as_dict(), indent=2) + "\n")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2) + "\n")
    print(f"best dev {select} {best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

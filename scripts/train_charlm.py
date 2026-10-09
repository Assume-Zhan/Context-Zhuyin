"""Train the small character LM used as the CPU reranker.

Pre-norm causal transformer (src/zhuyin_rescore/charlm.py), bf16 autocast on
the GPU, AdamW with warmup and cosine decay, one pass over non-overlapping
windows of the character stream (scripts/prepare_charlm_data.py).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from zhuyin_rescore.charlm import CharLM, CharLMConfig, CharVocab, save_charlm


def windows(tokens: np.ndarray, seq_len: int, batch: int, seed: int):
    n = (len(tokens) - 1) // seq_len
    order = np.random.default_rng(seed).permutation(n)
    for i in range(0, n - batch + 1, batch):
        starts = order[i : i + batch] * seq_len
        yield np.stack([tokens[s : s + seq_len + 1] for s in starts]).astype(np.int64)


@torch.no_grad()
def evaluate(model, val: np.ndarray, seq_len: int, device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for b in windows(val, seq_len, 64, seed=0):
        x = torch.from_numpy(b).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        losses.append(F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten()).item())
        if len(losses) >= max_batches:
            break
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/charlm/zhtw-v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    data = Path(args.data)
    vocab = CharVocab.load(data / "vocab.json")
    train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")
    cfg = CharLMConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_len=args.seq_len,
    )
    model = CharLM(cfg).to(device)
    tokens_per_step = args.seq_len * args.batch
    total_steps = (len(train) - 1) // tokens_per_step
    if args.max_tokens:
        total_steps = min(total_steps, int(args.max_tokens // tokens_per_step))
    n_params = model.num_params()
    print(f"params {n_params / 1e6:.2f}M, train tokens {len(train)}, steps {total_steps}", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=True,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    step, t0, run = 0, time.time(), []
    for b in windows(train, args.seq_len, args.batch, args.seed):
        x = torch.from_numpy(b).to(device, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        run.append(loss.item())
        step += 1
        if step % 200 == 0:
            rec = {"step": step, "loss": float(np.mean(run)), "lr": lr_at(step)}
            rec["tok_s"] = 200 * tokens_per_step / (time.time() - t0)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            run, t0 = [], time.time()
        if step % args.eval_every == 0 or step == total_steps:
            v = evaluate(model, val, args.seq_len, device)
            rec = {"step": step, "val_loss": v, "val_ppl": math.exp(v)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step >= total_steps:
            break
    save_charlm(model.cpu(), vocab, out)
    (out / "train_args.json").write_text(json.dumps(vars(args) | {"params": n_params}, indent=2) + "\n")
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

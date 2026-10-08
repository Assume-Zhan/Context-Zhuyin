"""Continued pretraining of a small causal LM on zh-TW text.

Full parameter training, fp32 master weights with bf16 autocast, AdamW,
linear warmup then cosine decay. The token stream (scripts/prepare_lm_data.py)
is cut into non-overlapping windows that are visited once in random order.
Inference only ever sees about 90 tokens, so short windows are enough.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def batches(tokens: np.ndarray, seq_len: int, batch: int, seed: int):
    n = (len(tokens) - 1) // seq_len
    order = np.random.default_rng(seed).permutation(n)
    for i in range(0, n - batch + 1, batch):
        starts = order[i : i + batch] * seq_len
        yield np.stack([tokens[s : s + seq_len + 1] for s in starts]).astype(np.int64)


@torch.no_grad()
def evaluate(model, val: np.ndarray, seq_len: int, n_windows: int, device) -> float:
    model.eval()
    losses = []
    for b in batches(val, seq_len, 16, seed=0):
        x = torch.from_numpy(b).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(input_ids=x[:, :-1], labels=x[:, :-1]).loss.item())
        if len(losses) * 16 >= n_windows:
            break
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", default="data/lm/zhtw-v1")
    ap.add_argument("--out", default="outputs/lm/qwen2.5-0.5b-zhtw")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=float, default=None, help="stop after this many training tokens")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    data = Path(args.data)
    train = np.memmap(data / "train.bin", dtype=np.uint32, mode="r")
    val = np.memmap(data / "val.bin", dtype=np.uint32, mode="r")
    tokens_per_step = args.seq_len * args.micro_batch * args.accum
    total_steps = (len(train) - 1) // (args.seq_len * args.micro_batch) // args.accum
    if args.max_tokens:
        total_steps = min(total_steps, int(args.max_tokens // tokens_per_step))
    print(f"train tokens {len(train)}, steps {total_steps}, tokens/step {tokens_per_step}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
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
    log = open(out / "train_log.jsonl", "a")
    val0 = evaluate(model, val, args.seq_len, 400, device)
    print(f"step 0 val loss {val0:.4f}", flush=True)
    log.write(json.dumps({"step": 0, "val_loss": val0}) + "\n")

    model.train()
    step, micro, t0, run_loss = 0, 0, time.time(), []
    for b in batches(train, args.seq_len, args.micro_batch, args.seed):
        x = torch.from_numpy(b).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x[:, :-1], labels=x[:, :-1]).loss
        (loss / args.accum).backward()
        run_loss.append(loss.item())
        micro += 1
        if micro % args.accum:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            el = time.time() - t0
            rec = {"step": step, "loss": float(np.mean(run_loss)), "lr": lr_at(step)}
            rec["tok_s"] = 50 * tokens_per_step / el
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run_loss, t0 = [], time.time()
        if step % args.eval_every == 0 or step == total_steps:
            v = evaluate(model, val, args.seq_len, 400, device)
            print(f"step {step} val loss {v:.4f}", flush=True)
            log.write(json.dumps({"step": step, "val_loss": v}) + "\n")
            log.flush()
        if step >= total_steps:
            break

    model.config.use_cache = True
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

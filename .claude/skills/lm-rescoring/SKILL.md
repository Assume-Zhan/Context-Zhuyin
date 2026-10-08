---
name: lm-rescoring
description: Score candidate sentences with a small causal LM (Qwen 0.5B to 1.7B) using a cached context prefix and one batched forward, then fuse with libchewing scores. Use when implementing or tuning the scorer, the score fusion lambda, or the resident scoring server.
---

# LM rescoring

## Scoring contract

- Input: committed context text (last 64 tokens), N candidate sentences of
  equal character length.
- Output: per candidate sum of token log-probs conditioned on the context.
- Score fusion: `score = lam * lm_logprob + (1 - lam) * chewing_score`. Tune
  `lam` on dev only. User defined phrases and manual selections always win.

## Implementation rules

1. Load once, keep resident: `AutoModelForCausalLM.from_pretrained(name,
   dtype=torch.float16).cuda().eval()` under `torch.inference_mode()`.
2. Context KV cache: run the context once, keep `past_key_values`, and expand
   it to batch size N for the candidate pass. Append committed text
   incrementally; recompute from scratch only when it exceeds the limit,
   preferably when idle.
3. One forward per rescoring: right pad candidates into a single batch with an
   attention mask covering context plus candidate tokens. Position ids must
   continue from the context length.
4. Only gather needed logits: take the final hidden states and multiply by
   the rows of `lm_head.weight` for the target token ids instead of computing
   the full vocab projection. Then log-softmax needs the full normalizer, so
   either compute `logsumexp` over the full projection once per position, or
   validate that the gather shortcut matches the reference within tolerance.
   Always keep a slow reference implementation in tests.
5. Shared prefix: candidates often differ in one or two words; a prefix tree
   pass that only computes tokens after the branch point is an optimization,
   added only after the plain batched version is correct.
6. Tokenization differs per candidate, but joint log-prob of the full string
   is comparable because every candidate covers the same number of
   characters. Never length normalize per token.

## Experiment matrix

| Config | Notes |
| --- | --- |
| libchewing 1-best | baseline |
| Oracle@10 | upper bound |
| Qwen2.5-0.5B or Qwen3-0.6B FP16 | |
| Qwen2.5-1.5B or Qwen3-1.7B FP16 and INT8 weight only | |
| Best LM without context | ablation |

Report CER, sentence accuracy, and the fraction of recoverable errors fixed:
`(CER_base - CER_lm) / (CER_base - CER_oracle)`. Break down by the three input
conditions and by focus error type. Watch for Taiwan form regressions caused
by Simplified Chinese heavy pretraining.

## Testing

- Unit test: batched scores equal per candidate unbatched scores within 1e-3.
- Unit test: context cached scores equal full sequence scores.
- Fix seeds and use `torch.use_deterministic_algorithms(True)` in tests where
  supported.

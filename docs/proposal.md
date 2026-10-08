# Zhuyin Candidate Selection with LLM Rescoring: libchewing + Small LM

Oct 8, 2026 - @Assume-Zhan

## Motivation and Goals

Proposal: keep libchewing for first stage candidate generation, then use a
small LM (0.5B to 1.5B) to rescore the top-10 candidate sentences, running on
an i5-14500 + RTX 3060 with average GPU and CPU utilization below 10%. A
preliminary estimate suggests this is feasible: GPU duty cycle around 2-6%,
CPU around 1-2%.

The n-gram language model in libchewing only sees short range context, so it
often picks the wrong homophone when the whole sentence is needed to decide,
as in "我明天再去" vs "我明天在家". A small LM can use the whole sentence and
the previously committed text, but using it for the entire decode is too
expensive, so it only reranks a small number of candidates.

Research questions:

1. Within libchewing's top-10 candidates, how much can LM rescoring improve
   top-1 character accuracy?
2. What are the trade-offs of model size (0.5B / 1.5B) and quantization in
   accuracy, latency, and resource usage?
3. Is the improvement larger under the "no tone" and "initials only" input
   conditions?

Success criteria:

- Character error rate (CER) reduced by >= 20% relative to libchewing
- Per rescoring latency p95 < 50 ms
- Under a sustained typing load of 100 characters per minute, average GPU and
  CPU utilization both < 10%

## System Architecture

The architecture is a two stage pipeline: libchewing generates candidates on
the CPU, a resident LM server scores them on the GPU, and the two communicate
over a local socket.

1. **Candidate generation (CPU)**: libchewing produces the 1-best sentence and
   word boundaries. Its public API mainly exposes candidate words at the
   cursor position and does not directly provide a whole sentence n-best, so
   we use "1-best + local substitution": for each word interval take the top
   few alternative words, and combine them into at most 10 candidate
   sentences (or build a lattice from its dictionary and run k-best Viterbi).
2. **LM scoring (GPU)**: the KV cache of the context (committed text,
   truncated to the most recent 64 tokens) stays resident; for the 10
   candidates only the candidate part is computed, as one batch in a single
   forward pass, taking the sum of log-probs of each sentence. All candidates
   have the same number of characters (one character per syllable), so their
   joint probabilities are directly comparable regardless of how they are
   tokenized.
3. **Score fusion**: score = lambda * LM log-prob + (1 - lambda) * libchewing
   score, with lambda tuned on the dev set. User defined phrases and words the
   user manually selected keep the highest priority and are never overridden
   by the LM.
4. **Trigger policy**: do not compute on every keystroke. Trigger only when a
   syllable is completed and there is a pause of >= 100 ms (debounce), or when
   the user presses the commit key. Show libchewing's 1-best until the LM
   result arrives, then silently update the preedit, so typing is never
   blocked.
5. **Context update**: committed text is appended incrementally to the KV
   cache; when it exceeds the limit, recompute the whole segment once
   (infrequent, and can be deferred to idle time).

## Feasibility Analysis (i5-14500 + RTX 3060)

Conclusion: feasible. A 0.5B class model has a GPU duty cycle of about 1-2%,
a 1.5B class model about 3-6%, and CPU stays below 1% in both cases, all
within the 10% budget. The following are unmeasured estimates that need to be
confirmed by measurement in Phase 1.

**Compute.** One rescoring = 10 candidates x about 10 tokens = about 100
tokens of prefill (the context is already in the KV cache). FLOPs ~= 2 x
parameter count x token count: about 0.1 TFLOP for 0.5B and about 0.3 TFLOP
for 1.5B. Estimating the usable FP16 tensor throughput of the 3060 at 10-20
TFLOPS, and weight reads (360 GB/s) at about 3 ms and 8 ms respectively, the
workload is overall compute bound.

**Utilization.** GPU utilization in nvidia-smi is the fraction of time a
kernel is executing, so utilization ~= trigger rate x GPU time per call. In
the worst case of 100 characters per minute with a trigger on every
character, that is about 1.7 calls per second; with debounce it is usually
lower.

| Config | VRAM | GPU time per call | GPU utilization (1.7 calls/s) | CPU utilization (20 threads) | p95 < 50 ms |
| --- | --- | --- | --- | --- | --- |
| 0.5B class (Qwen2.5-0.5B / Qwen3-0.6B) FP16 | ~1.2 GB | 5-12 ms | 1-2% | < 1% | Comfortable margin |
| 1.5B class (Qwen2.5-1.5B / Qwen3-1.7B) FP16 | ~3.5 GB | 15-35 ms | 3-6% | < 1% | Barely achievable |
| 1.5B class INT8 weight only | ~2 GB | 15-35 ms | 3-6% | < 1% | Barely achievable |
| CPU only, 0.5B Q4 (llama.cpp, 4 threads) | ~0.5 GB RAM | 60-150 ms (CPU) | 0% | ~3-5% | Not met |

Under a compute bound workload INT8 saves VRAM, not time. CPU only is suitable
only as a fallback; the latency would make candidate updates visibly "jump".

**Key optimizations to keep utilization low:**

- Compute only the needed logits: a candidate is only a few tokens, so gather
  the corresponding rows of lm\_head instead of computing the projection over
  the full 150K vocabulary (saves close to 30% of compute for a 0.5B model).
- Shared candidate prefixes: the 10 candidates usually differ by only one or
  two words, so use a prefix tree and only compute tokens after the branch
  point.
- Use CUDA Graphs, TensorRT-LLM, or the llama.cpp CUDA backend to cut the
  overhead of hundreds of kernel launches per call.

**CPU side pitfalls (more likely to blow the budget than the compute itself):**

- CUDA synchronization may busy wait by default, pinning one core at 100%
  while waiting for the GPU. Use `torch.cuda.Event(blocking=True)` and then
  `synchronize()`, or set cudaDeviceScheduleBlockingSync before
  initialization.
- PyTorch / OpenMP thread pools spin wait after every op and may consume
  several cores at once. Set `OMP_NUM_THREADS=1`, `OMP_WAIT_POLICY=PASSIVE`,
  and `torch.set_num_threads(1)`.
- When idle, the server must block on its socket and sleep, not poll.

**Coexisting with other GPU work:** if the same 3060 is also running training
or simulation, rescoring latency will increase. Use a high priority CUDA
stream, or automatically fall back to the libchewing 1-best when latency
exceeds a threshold.

## Experiment Design

Start with an offline benchmark to confirm that the accuracy gain is worth it,
before investing in online integration.

**Data.** Taiwan Traditional Chinese text (news, zh-TW Wikipedia, PTT),
converted to zhuyin sequences with g2pW as the input, with the original text
as the answer. Split into short sentences (5-20 characters) and keep the
previous sentence as context. About 5,000 sentences each for dev and test,
plus a small manually checked sample to estimate the g2pW conversion error
rate.

**Input conditions.** Three: full zhuyin with tone, no tone, and initials only
(abbreviated input).

**Baselines and systems.**

| Method | Description |
| --- | --- |
| libchewing 1-best | baseline |
| Oracle@10 | the candidate in the top-10 closest to the answer; the upper bound of rescoring |
| + 0.5B class LM | Qwen2.5-0.5B or Qwen3-0.6B, FP16 |
| + 1.5B class LM | Qwen2.5-1.5B or Qwen3-1.7B, FP16 and INT8 |
| + LM without context | ablation: only the current sentence, to measure the contribution of context |

**Metrics.**

- Accuracy: CER, sentence accuracy, and the gap to Oracle@10 (how much of the
  recoverable error has been recovered)
- Performance: per rescoring latency p50 / p95, VRAM
- Resources: replay the dataset as a typing event stream at 100 characters
  per minute, sample with NVML (`nvidia-smi dmon -s u`) and psutil every
  100 ms, and report average and peak GPU / CPU utilization

**Error types to watch.** High frequency homophones (在/再, 的/得/地,
他/她/它), Taiwan specific character forms (裡/裏, 著/着), and proper nouns. If
the LM was trained mainly on Simplified Chinese data, it may actually do worse
than libchewing on Taiwan character forms.

## Risks and Timeline

The biggest risk is not compute but whether libchewing's top-10 often
contains the correct answer; if Oracle@10 is too low, no rescoring can
recover the errors.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| libchewing has no ready made whole sentence n-best API | Candidate generation has to be built ourselves | Start with "1-best + local substitution"; if insufficient, build a lattice from the dictionary and run k-best |
| Oracle@10 too low | Limits the achievable improvement | Widen to top-20 to 30 and re-measure utilization |
| Small model biased against Taiwan character forms | Wrong Traditional Chinese forms selected | LoRA or continued pretraining on Taiwan text; set lambda conservatively |
| Candidate updates make the screen "jump" | Worse user experience | Only update while the preedit is not yet committed, and only switch when the score gap exceeds a threshold |
| Sharing the GPU with lab workloads | Latency spikes | Automatically fall back to the libchewing 1-best on timeout |

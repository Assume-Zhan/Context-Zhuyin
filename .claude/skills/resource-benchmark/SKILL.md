---
name: resource-benchmark
description: Measure rescoring latency (p50, p95), VRAM, and average and peak GPU and CPU utilization under a replayed 100 chars per minute typing load. Use when benchmarking, profiling, or checking the under 10 percent resource budget and the p95 under 50 ms latency target.
---

# Resource and latency benchmark protocol

Targets from the proposal: p95 rescoring latency < 50 ms; average GPU and CPU
utilization both < 10 percent at 100 chars per minute.

## Required process settings (scorer side)

Set before importing torch or creating CUDA contexts:

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

- Avoid CUDA busy wait: wait on `torch.cuda.Event(blocking=True)` and call
  `event.synchronize()`, or set blocking sync on the device before first use.
  A plain `torch.cuda.synchronize()` may spin one core at 100 percent.
- The idle server must block on its socket (`recv`, `select`), never poll.

## Latency

- Warm up at least 20 calls first; exclude them.
- Time with CUDA events around the scoring call plus wall clock around the
  whole request; report both, p50 and p95, over at least 1,000 requests.
- Report VRAM via `torch.cuda.max_memory_allocated()` and NVML used memory.

## Utilization under load

1. Replay dataset sentences as a typing event stream: 100 chars per minute,
   one rescoring trigger per completed syllable after a 100 ms debounce.
2. Sample every 100 ms in a separate lightweight process:
   - GPU: `pynvml.nvmlDeviceGetUtilizationRates(h).gpu` (module from the
     `nvidia-ml-py` package), or `nvidia-smi dmon -s u` as a cross check.
   - CPU: `psutil.Process(pid).cpu_percent(None)` for the scorer process and
     `psutil.cpu_percent(None)` for the system. Normalize per process numbers
     by logical core count when comparing with the budget.
3. Run at least 5 minutes. Report mean and peak for GPU and CPU.
4. Record the GPU model, driver, torch version, and whether other jobs were
   running. The proposal targets RTX 3060; the dev host has an RTX 5090, so
   label results by GPU and never present 5090 numbers as 3060 results.

## Output

Write a CSV per run to `outputs/bench/<date>-<config>.csv` with raw samples
and a one row summary JSON next to it. Summary keys: `config`, `gpu_name`,
`latency_p50_ms`, `latency_p95_ms`, `vram_mb`, `gpu_util_mean`,
`gpu_util_peak`, `cpu_util_mean`, `cpu_util_peak`.

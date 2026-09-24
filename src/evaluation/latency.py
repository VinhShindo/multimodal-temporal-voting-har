"""Đo latency, FPS, GPU/CPU cost."""
import time
import torch


def measure_latency(fn, n_runs: int = 50) -> dict:
    # warmup
    for _ in range(5):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "avg_latency_ms": elapsed / n_runs * 1000,
        "fps": n_runs / elapsed,
        "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 1e6
                        if torch.cuda.is_available() else 0),
    }

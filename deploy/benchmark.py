import time
import torch


def benchmark_latency(infer_fn, image, runs: int = 20, warmup: int = 3) -> dict:
    """测单图端到端推理延时(GPU 自动 synchronize)。
    infer_fn: 接受 image 的可调用对象。返回 {'mean_ms','p95_ms'}。"""
    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        infer_fn(image)
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        infer_fn(image)
        if use_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean_ms = sum(times) / len(times)
    p95_ms = times[min(len(times) - 1, int(0.95 * len(times)))]
    return {"mean_ms": mean_ms, "p95_ms": p95_ms}

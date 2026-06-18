from deploy.benchmark import benchmark_latency

def test_runs_warmup_plus_measured_and_reports():
    calls = []
    def fn(x):
        calls.append(1)
    m = benchmark_latency(fn, image=None, runs=5, warmup=2)
    assert len(calls) == 7                 # 2 warmup + 5 measured
    assert m["mean_ms"] >= 0.0
    assert m["p95_ms"] >= 0.0
    assert set(m.keys()) == {"mean_ms", "p95_ms"}

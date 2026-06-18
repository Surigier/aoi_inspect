# Plan 5a — 延时基准 + 记忆库 GPU 加速 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供带 GPU 同步的端到端延时基准,并修掉纹理分支的延时瓶颈(记忆库强制 CPU 导致 cdist 跑在 CPU),坐实赛题"检测时间 30%"分。

**Architecture:** `deploy/benchmark.py` 提供 `benchmark_latency(infer_fn, image, runs, warmup)`,含 warmup 与 `torch.cuda.synchronize()`,返回 mean/p95。`MemoryBank` 改为**保留输入张量所在设备**(不再强制 `.cpu()`),使 GPU 上的特征查询在 GPU 完成;`TextureADBranch.infer` 相应改为 `.cpu().numpy()`。CLI 脚本对各分支在真实图上测延时。

**Tech Stack:** Python 3.8 · PyTorch 2.4 · pytest(复用现有 Backbone/各分支/load_category)

**工作目录:** 路径相对 `/home/srj/yolo/aoi_inspect/`。建议在分支 `plan5a-latency` 上实现。每个 task 末尾提交。

**背景实测:** 当前 GPU 上 texture≈295ms(超 200ms)、structural≈13ms、clip≈10ms。瓶颈在 `MemoryBank.query`/`add` 的 `.cpu()`(`aoi/memory_bank.py`)。

---

### Task 1: 延时基准工具 benchmark_latency

**Files:**
- Create: `deploy/__init__.py`
- Create: `deploy/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_benchmark.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.benchmark'`

- [ ] **Step 3: 建 `deploy/__init__.py`(空文件)并实现 `deploy/benchmark.py`**

```bash
mkdir -p deploy && touch deploy/__init__.py
```

```python
# deploy/benchmark.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: Commit**

```bash
git add deploy/__init__.py deploy/benchmark.py tests/test_benchmark.py && git commit -m "feat: 延时基准 benchmark_latency(含 cuda 同步)"
```

---

### Task 2: 记忆库保留设备 + 纹理分支适配

**Files:**
- Modify: `aoi/memory_bank.py`
- Modify: `aoi/branches/texture_ad.py`
- Test: `tests/test_memory_bank.py`(追加一条)

- [ ] **Step 1: 追加失败测试到 `tests/test_memory_bank.py`**

在文件末尾追加:
```python
def test_query_preserves_input_device():
    # query/add 不再强制 CPU:在 CPU 输入下结果仍在 CPU(GPU 上则会留在 GPU 加速 cdist)
    mb = MemoryBank()
    mb.add(torch.zeros(3, 2))
    d = mb.query(torch.zeros(1, 2))
    assert d.device.type == "cpu"
    assert mb.bank.device.type == "cpu"
```

- [ ] **Step 2: 跑测试确认通过或失败**

Run: `python -m pytest tests/test_memory_bank.py -v`
Expected: 现状下 PASS(当前强制 CPU,CPU 输入恰好满足)。本测试用于**锁定行为**,Step 3 改完仍须 PASS。

- [ ] **Step 3: 改 `aoi/memory_bank.py`——不再强制 `.cpu()`,保留输入设备**

把 `add` 与 `query` 改为:
```python
    def add(self, features: torch.Tensor) -> None:
        features = features.detach().float()            # 保留所在设备(GPU 上则留 GPU)
        self.bank = features if self.bank is None else torch.cat([self.bank, features], dim=0)

    def query(self, features: torch.Tensor) -> torch.Tensor:
        """返回每个 query 特征到 bank 的最小 L2 距离,形状 (M,)。在 bank 所在设备上计算。"""
        d = torch.cdist(features.detach().float().to(self.bank.device), self.bank)
        return d.min(dim=1).values
```
(`coreset_subsample` 不变。)

- [ ] **Step 4: 改 `aoi/branches/texture_ad.py`——查询结果可能在 GPU,转 CPU 再 numpy**

把 `infer` 中:
```python
        dist = self.bank.query(feats)                 # (h*w,)
        amap = dist.reshape(h, w).numpy()
```
改为:
```python
        dist = self.bank.query(feats)                 # (h*w,) 可能在 GPU
        amap = dist.reshape(h, w).cpu().numpy()
```

- [ ] **Step 5: 跑全量测试确认无回归**

Run: `python -m pytest`
Expected: 全部 PASS(CPU 路径行为不变;新测试 PASS)

- [ ] **Step 6: Commit**

```bash
git add aoi/memory_bank.py aoi/branches/texture_ad.py tests/test_memory_bank.py && git commit -m "perf: 记忆库保留设备,GPU 上 cdist 留在 GPU(加速纹理分支延时)"
```

---

### Task 3: 延时基准 CLI

**Files:**
- Create: `scripts/run_benchmark.py`

**说明:** 对真实图测各分支端到端延时,验证优化后 GPU 上达标(<200ms)。手动跑。

- [ ] **Step 1: 实现 `scripts/run_benchmark.py`**

```python
"""延时基准:python scripts/run_benchmark.py data/mvtec/bottle
对各分支测单图端到端延时(优化前后对比看 texture)。"""
import sys
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from deploy.benchmark import benchmark_latency
from eval.mvtec import load_category


def main(root):
    data = load_category(root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(pretrained=True, device=device)
    normals = torch.stack(data["train_normal"][:50])
    probe = data["test_normal"][0].unsqueeze(0)
    branches = {
        "texture": TextureADBranch(backbone=bb, coreset_ratio=0.25),
        "structural": StructuralADBranch(backbone=bb, grid_size=16),
    }
    print(f"device={device}")
    for name, b in branches.items():
        b.fit(normals)
        m = benchmark_latency(b.infer, probe, runs=20, warmup=3)
        print(f"{name:11s} mean={m['mean_ms']:.1f}ms  p95={m['p95_ms']:.1f}ms")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('scripts/run_benchmark.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: 全量测试无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/run_benchmark.py && git commit -m "feat: 延时基准 CLI run_benchmark"
```

---

## Self-Review

**Spec 覆盖:** 设计 spec"检测时间 30% / OpenVINO 提速 / 延时基准"中的**延时基准 + GPU 加速**部分 → Task 1/2/3 ✅。(OpenVINO CPU 导出留作 Plan 5a 后续/独立项,先用 GPU 加速达标 <200ms。)

**占位符扫描:** 无 TBD/TODO;每步含完整代码。

**类型一致性:** `benchmark_latency(infer_fn, image, runs, warmup)` 被 Task 3 CLI 一致调用;`MemoryBank.add/query` 签名不变(仅去掉强制 CPU);`TextureADBranch.infer` 仍返回 `BranchResult`,仅内部 `.cpu().numpy()`。✅

**范围(YAGNI):**
- **纳入**:延时基准(含 cuda 同步,修 Plan 1 登记项)+ 记忆库 GPU 加速 + CLI。
- **延后**:OpenVINO CPU 导出(达到 CPU<2s 的挑战目标)→ Plan 5a-2;2500×2500 切块策略 → 若实测大图超时再做;LLM 冷路径/尺寸分支/文档 → Plan 5b/5c/5d。

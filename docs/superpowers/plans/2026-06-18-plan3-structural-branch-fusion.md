# Plan 3 — 位置感知结构分支 + 多分支融合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增**无监督的位置感知结构分支**(捕捉缺件/错位/逻辑异常),并用**多分支融合适配器**把纹理/零样本/结构三分支统一成一个评分系统;在 MVTec LOCO 逻辑异常上验证。

**Architecture:** 标准 PatchCore 把所有 patch 混进一个库(平移不变),查不出"某位置该有的部件没了"。`StructuralADBranch` 改为**按空间格子分别建库**(自适应池化到 grid×grid 网格,每格一个正常特征库):某格测试特征远离该格正常库 = 该位置缺件/错位。`MultiBranchAdapter` 对各分支用"正常分均值方差"做 z 归一化后融合(取最大),再标定单一阈值;它复用 Plan 1 的 `FewShotAdapter._calibrate`,且接口与 `FewShotAdapter` 一致,可直接喂给现有 `run_protocol`。

**Tech Stack:** Python 3.8 · PyTorch 2.4 · numpy · pytest(复用 Plan 1 的 Backbone/MemoryBank/BranchResult/run_protocol/load_category)

**工作目录:** 所有路径相对 `/home/srj/yolo/aoi_inspect/`。建议在新分支 `plan3-structural-fusion` 上实现。每个 task 末尾提交。

---

### Task 1: 位置感知结构分支 StructuralADBranch

**Files:**
- Create: `aoi/branches/structural_ad.py`
- Test: `tests/test_structural_ad.py`

- [ ] **Step 1: 写失败测试**(关键:位置感知能查出"缺件",普通混合库查不出)

```python
# tests/test_structural_ad.py
import pytest
import torch
from aoi.backbone import Backbone
from aoi.branches.structural_ad import StructuralADBranch


def _branch():
    return StructuralADBranch(backbone=Backbone(pretrained=False), grid_size=8)

def _with_square():
    """左上角亮块 = 部件存在。"""
    img = torch.full((1, 3, 64, 64), 0.5)
    img[:, :, :16, :16] = 1.0
    return img

def _missing():
    """无亮块 = 部件缺失。"""
    return torch.full((1, 3, 64, 64), 0.5)

def test_flags_missing_component():
    b = _branch()
    b.fit(torch.cat([_with_square() for _ in range(4)], dim=0))   # 正常都带左上块
    s_present = b.infer(_with_square()).score
    s_missing = b.infer(_missing()).score
    assert s_missing > s_present                                   # 缺件 → 高分

def test_infer_map_shape_and_batch_guard():
    b = _branch()
    b.fit(_with_square())
    r = b.infer(_with_square())
    assert r.anomaly_map.shape == (8, 8)
    assert r.defect_type == "structural"
    with pytest.raises(AssertionError):
        b.infer(torch.zeros(2, 3, 64, 64))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_structural_ad.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.branches.structural_ad'`

- [ ] **Step 3: 实现 `aoi/branches/structural_ad.py`**

```python
import time
import torch
import torch.nn.functional as F
from ..types import BranchResult
from ..backbone import Backbone


class StructuralADBranch:
    """位置感知记忆库:按空间格子分别建正常特征库,捕捉缺件/错位/逻辑异常
    (标准平移不变记忆库捕捉不到"某位置该有的部件没了")。"""
    defect_type = "structural"

    def __init__(self, backbone: Backbone = None, grid_size: int = 8):
        self.backbone = backbone or Backbone()
        self.grid_size = grid_size
        self.bank = None  # (num_cells, N, C)

    def _cell_features(self, image: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (num_cells, B, C);特征图自适应池化到 grid×grid 网格。"""
        fmap = self.backbone.extract(image)                      # (B,C,h,w)
        g = self.grid_size
        pooled = F.adaptive_avg_pool2d(fmap, output_size=g)      # (B,C,g,g)
        b, c, _, _ = pooled.shape
        return pooled.reshape(b, c, g * g).permute(2, 0, 1)      # (num_cells, B, C)

    def fit(self, images: torch.Tensor) -> None:
        feats = self._cell_features(images)                      # (num_cells, B, C)
        self.bank = feats if self.bank is None else torch.cat([self.bank, feats], dim=1)

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        q = self._cell_features(image)                           # (num_cells, 1, C)
        d = torch.cdist(q, self.bank)                            # (num_cells, 1, N)
        cell_dist = d.min(dim=2).values.squeeze(1)              # (num_cells,)
        amap = cell_dist.reshape(self.grid_size, self.grid_size).cpu().numpy()
        score = float(amap.max())
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=amap,
                            defect_type=self.defect_type, latency_ms=lat)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_structural_ad.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 全量测试无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add aoi/branches/structural_ad.py tests/test_structural_ad.py && git commit -m "feat: 位置感知结构分支 StructuralADBranch(无监督查缺件/错位)"
```

---

### Task 2: 融合工具 fusion

**Files:**
- Create: `aoi/fusion.py`
- Test: `tests/test_fusion.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fusion.py
from aoi.fusion import znorm, fuse

def test_znorm_basic():
    assert znorm(2.0, mean=0.0, std=2.0) == 1.0

def test_znorm_zero_std_safe():
    # std 为 0 时退化为减均值(不除零)
    assert znorm(3.0, mean=1.0, std=0.0) == 2.0

def test_fuse_takes_max():
    assert fuse([0.2, 1.5, -0.3]) == 1.5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.fusion'`

- [ ] **Step 3: 实现 `aoi/fusion.py`**

```python
def znorm(score: float, mean: float, std: float) -> float:
    """按正常分的均值/方差做 z 归一化;std≈0 时退化为减均值,避免除零。"""
    return (score - mean) / (std if std > 1e-12 else 1.0)


def fuse(norm_scores) -> float:
    """多分支归一化分数融合:取最大(任一分支报异常即视为异常)。"""
    return max(norm_scores)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: Commit**

```bash
git add aoi/fusion.py tests/test_fusion.py && git commit -m "feat: 融合工具 znorm + fuse(max)"
```

---

### Task 3: 多分支适配器 MultiBranchAdapter

**Files:**
- Create: `aoi/multibranch.py`
- Test: `tests/test_multibranch.py`

**接口:** 与 `FewShotAdapter` 一致(`fit_fewshot(normal_list, defect_list)` + `predict((1,3,H,W))->(BranchResult, bool)`),可直接喂给 `run_protocol`。复用 `FewShotAdapter._calibrate` 标定融合阈值。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_multibranch.py
import torch
from aoi.multibranch import MultiBranchAdapter
from aoi.types import BranchResult


class _FakeBranch:
    def __init__(self, defect_type, score_fn):
        self.defect_type = defect_type
        self.score_fn = score_fn
    def fit(self, images):
        return None
    def infer(self, image):
        return BranchResult(score=float(self.score_fn(image)), defect_type=self.defect_type)


def test_fit_predict_fuses_and_separates():
    # b1 按图像均值打分(正常0/缺陷1);b2 恒为0(静默分支)
    b1 = _FakeBranch("appearance", lambda im: im.mean())
    b2 = _FakeBranch("structural", lambda im: 0.0)
    a = MultiBranchAdapter([b1, b2])
    normals = [torch.zeros(3, 8, 8) for _ in range(4)]
    defects = [torch.ones(3, 8, 8) for _ in range(4)]
    a.fit_fewshot(normals, defects)
    r_n, is_n = a.predict(torch.zeros(1, 3, 8, 8))
    r_d, is_d = a.predict(torch.ones(1, 3, 8, 8))
    assert is_n is False
    assert is_d is True
    assert r_d.defect_type == "appearance"   # 最异常的分支决定缺陷类型
    assert r_n.defect_type == "normal"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_multibranch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.multibranch'`

- [ ] **Step 3: 实现 `aoi/multibranch.py`**

```python
import torch
from .fewshot import FewShotAdapter
from .fusion import znorm, fuse


class MultiBranchAdapter:
    """多分支:各分支 fit + 按正常分均值方差 z 归一化,融合(max)后标定单一阈值。
    接口与 FewShotAdapter 一致,可直接喂给 run_protocol。"""

    def __init__(self, branches):
        self.branches = branches
        self.stats = []        # [(mean, std), ...] 每分支正常分统计
        self.threshold = None

    def fit_fewshot(self, normal_images, defect_images):
        stacked = torch.stack(normal_images)
        for b in self.branches:
            b.fit(stacked)
        self.stats = []
        for b in self.branches:
            ns = [b.infer(img.unsqueeze(0)).score for img in normal_images]
            m = sum(ns) / len(ns)
            var = sum((x - m) ** 2 for x in ns) / len(ns)
            self.stats.append((m, var ** 0.5))
        norm_fused = [self._fused(img.unsqueeze(0))[0] for img in normal_images]
        def_fused = [self._fused(img.unsqueeze(0))[0] for img in defect_images]
        self.threshold = FewShotAdapter._calibrate(norm_fused, def_fused)
        return self.threshold

    def _fused(self, image):
        """image (1,3,H,W) -> (融合分, 最异常分支的 BranchResult)"""
        zs, best = [], None
        for b, (m, s) in zip(self.branches, self.stats):
            r = b.infer(image)
            z = znorm(r.score, m, s)
            zs.append(z)
            if best is None or z > best[0]:
                best = (z, r)
        return fuse(zs), best[1]

    def predict(self, image):
        fused, res = self._fused(image)
        is_defect = bool(fused >= self.threshold)
        res.defect_type = res.defect_type if is_defect else "normal"
        return res, is_defect
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_multibranch.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 全量测试无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add aoi/multibranch.py tests/test_multibranch.py && git commit -m "feat: 多分支融合适配器 MultiBranchAdapter"
```

---

### Task 4: MVTec LOCO 逻辑异常冒烟

**Files:**
- Create: `scripts/run_loco_smoke.py`

**说明:** LOCO 类别布局同 MVTec(`train/good`、`test/good`、`test/logical_anomalies`、`test/structural_anomalies`),现有 `eval/mvtec.py:load_category` 直接可用(把 logical+structural 都归为缺陷)。对比 纹理 / 结构 / 融合 三种配置,验证结构分支抓住纹理分支漏掉的逻辑异常。真实数据存在时手动跑。

- [ ] **Step 1: 实现 `scripts/run_loco_smoke.py`**

```python
"""LOCO 逻辑异常冒烟:python scripts/run_loco_smoke.py data/_dl/mvtec_loco/breakfast_box
对比 纹理 / 结构 / 融合 在逻辑异常上的 AUROC。"""
import sys
import random
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.fewshot import FewShotAdapter
from aoi.multibranch import MultiBranchAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category


def main(root):
    data = load_category(root)
    rng = random.Random(0)
    normals = data["train_normal"][:]
    defects = data["test_defect"][:]
    rng.shuffle(normals)
    rng.shuffle(defects)
    fn = normals[:100]
    fd = defects[:30]
    ti = data["test_normal"] + defects[30:]
    tl = [0] * len(data["test_normal"]) + [1] * len(defects[30:])
    bb = Backbone(pretrained=True, device="cuda")
    configs = {
        "texture": FewShotAdapter(TextureADBranch(backbone=bb, coreset_ratio=0.25)),
        "structural": FewShotAdapter(StructuralADBranch(backbone=bb, grid_size=16)),
        "fused": MultiBranchAdapter([
            TextureADBranch(backbone=bb, coreset_ratio=0.25),
            StructuralADBranch(backbone=bb, grid_size=16),
        ]),
    }
    for name, ad in configs.items():
        m = run_protocol(ad, fn, fd, ti, tl)
        print(f"{name:11s} AUROC={m['auroc']:.3f} acc={m['accuracy']:.3f} lat={m['latency_ms_mean']:.0f}ms")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('scripts/run_loco_smoke.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: 全量测试无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 4:(有数据时)手动冒烟** — 无 LOCO 数据则 SKIP。

Run: `python scripts/run_loco_smoke.py data/_dl/mvtec_loco/breakfast_box`
Expected: 打印三行 AUROC;预期结构/融合在逻辑异常上优于纯纹理。

- [ ] **Step 5: Commit**

```bash
git add scripts/run_loco_smoke.py && git commit -m "feat: MVTec LOCO 逻辑异常冒烟(纹理/结构/融合对比)"
```

---

## Self-Review

**Spec 覆盖:** 设计 spec 的"缺件检测 + 顺序/逻辑校验"→ Task 1 位置感知结构分支(无监督,契合未知域)✅;"编排器 + 融合"→ Task 2/3(MultiBranchAdapter 承担路由+融合)✅;复用 run_protocol/load_category → Task 3/4 ✅;MVTec LOCO 验证 → Task 4 ✅。

**占位符扫描:** 无 TBD/TODO;每步含完整代码。

**类型一致性:** `StructuralADBranch` 实现 `fit(images)` / `infer((1,3,H,W))->BranchResult` / `defect_type`,与 Plan 1 Branch 接口一致;`MultiBranchAdapter` 暴露 `fit_fewshot`/`predict`,与 `FewShotAdapter` 一致并复用其 `_calibrate`;`znorm`/`fuse`(Task 2)被 Task 3 一致调用。✅

**范围(YAGNI):**
- **纳入**:结构分支 + 融合 + 多分支适配器 + LOCO 验证。
- **延后**:**尺寸测量分支**(需相机标定/尺度,自成一体)→ 后续小 plan;**编排器的延时守卫/跳过慢分支** → Plan 5(部署/延时专项);视频时序 → Plan 5。
- **已知**:`MultiBranchAdapter.fit_fewshot` 对正常样本做了两遍 infer(先统计后融合),100 张可接受,后续可缓存优化。

---

## 代码审查后续项(最终审查记录)

最终审查发现 **1 个 Critical**,已即时修复并补回归测试 + 重跑验证:

**已修复(C1,Critical):** `MultiBranchAdapter.predict` 原返回最异常分支的**原始分**,导致 `run_protocol` 的 AUROC 基于错误量纲(非融合分)。修复:`predict` 中 `res.score = fused`,使 AUROC/阈值同基于融合 z 分;保留最异常分支的 anomaly_map。补 `test_predict_returns_fused_score_not_raw` 回归测试。重跑 LOCO:fused 仍 0.837(结构分支 z 值主导,max 融合=结构排序;此为可信结果,融合增益需在外观+逻辑混合缺陷上体现)。

**登记延后:**
- [ ] **优化(I2)**:`fit_fewshot` 对正常样本两遍 infer;缓存首遍原始分即可去掉 2× GPU 开销。
- [ ] **Plan 5/部署(I3)**:结构库内存 = `grid² × N × C`,grid 增大呈平方增长且无 coreset;RK3588 受限,需文档化内存公式 + 可选 per-cell 上限/coreset。
- [ ] **健壮性(M2/M4)**:空分支列表无保护;`znorm` 的 std≈0 边界(1e-15)未测。
- [ ] **可用性(M5)**:LOCO 冒烟脚本硬编码 `device="cuda"`,可加 CPU 回退。
- [ ] **observation**:texture 配置因 coreset 随机下采样未固定种子,run-to-run AUROC 有小幅抖动;复现实验前可固定种子。

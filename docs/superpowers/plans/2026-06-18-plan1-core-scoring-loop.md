# Plan 1 — 地基 + 核心评分闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭起项目骨架并跑通"100 正常 + 30 缺陷 → `fit_fewshot` 迁移 → 测试 → 输出 AUROC/准确率/延时"的最小可运行评分闭环。

**Architecture:** 自包含的 PatchCore 风格记忆库异常检测分支(timm 骨干 + 记忆库,无梯度训练,契合现场 `fit_fewshot`),配少样本适配器与官方协议评测器。后续 Plan 2~5 在此地基上挂载其余分支/编排/主动学习/部署/Demo。

**Tech Stack:** Python 3 · PyTorch 2.4(cu121,已装)· timm · numpy · pytest · Pillow

**工作目录:** 所有路径相对 `/home/srj/yolo/aoi_inspect/`。每个 task 末尾提交。

---

### Task 0: 项目骨架

**Files:**
- Create: `aoi/__init__.py`, `aoi/branches/__init__.py`, `eval/__init__.py`, `tests/__init__.py`
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `README.md`

- [ ] **Step 1: 建目录与空包文件**

```bash
cd /home/srj/yolo/aoi_inspect
mkdir -p aoi/branches eval tests scripts data
touch aoi/__init__.py aoi/branches/__init__.py eval/__init__.py tests/__init__.py
```

- [ ] **Step 2: 写 `requirements.txt`**

```
timm>=1.0
numpy>=1.24
Pillow>=10.0
pytest>=8.0
```
(torch 已随系统环境提供,不写入以免覆盖 cu121 版本。)

- [ ] **Step 3: 写 `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
data/
runs/
*.pt
*.pth
*.onnx
```

- [ ] **Step 4: 写 `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 5: 写 `README.md`(占位一句话)**

```markdown
# AOI 实时在线 AI 质检系统(华为赛题一)
见 `docs/superpowers/specs/` 设计文档与 `docs/superpowers/plans/` 实现计划。
```

- [ ] **Step 6: 装依赖并验证 pytest 可运行**

Run: `pip install -r requirements.txt && python -m pytest`
Expected: `no tests ran`(收集到 0 个测试,无错误)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: 项目骨架与依赖"
```

---

### Task 1: BranchResult 数据结构

**Files:**
- Create: `aoi/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_types.py
import numpy as np
from aoi.types import BranchResult

def test_branch_result_defaults():
    r = BranchResult(score=0.7)
    assert r.score == 0.7
    assert r.anomaly_map is None
    assert r.boxes == []
    assert r.defect_type == "unknown"
    assert r.latency_ms == 0.0

def test_branch_result_with_map():
    m = np.zeros((4, 4))
    r = BranchResult(score=1.0, anomaly_map=m, defect_type="appearance", latency_ms=12.3)
    assert r.anomaly_map.shape == (4, 4)
    assert r.defect_type == "appearance"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.types'`

- [ ] **Step 3: 实现 `aoi/types.py`**

```python
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class BranchResult:
    """单个检测分支的输出。所有分支统一返回此结构。"""
    score: float                                  # 图像级异常分,越大越异常
    anomaly_map: Optional[np.ndarray] = None      # HxW 像素级异常图
    boxes: List = field(default_factory=list)     # [(x1,y1,x2,y2,label), ...]
    defect_type: str = "unknown"
    latency_ms: float = 0.0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_types.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add aoi/types.py tests/test_types.py && git commit -m "feat: BranchResult 统一分支输出结构"
```

---

### Task 2: 记忆库 MemoryBank

**Files:**
- Create: `aoi/memory_bank.py`
- Test: `tests/test_memory_bank.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_memory_bank.py
import torch
from aoi.memory_bank import MemoryBank

def test_add_and_query_nearest():
    mb = MemoryBank()
    mb.add(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
    d = mb.query(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert d.shape == (2,)
    assert d[0].item() < 1e-5                       # 与 [0,0] 重合
    assert abs(d[1].item() - (2.0 ** 0.5)) < 1e-4   # 到 [0,0] 的距离 sqrt(2)

def test_coreset_reduces_size():
    mb = MemoryBank()
    mb.add(torch.randn(100, 8))
    mb.coreset_subsample(0.25)
    assert mb.bank.shape[0] == 25
    assert mb.bank.shape[1] == 8
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_memory_bank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.memory_bank'`

- [ ] **Step 3: 实现 `aoi/memory_bank.py`**

```python
import torch


class MemoryBank:
    """存储正常样本的 patch 特征,查询时返回到最近邻的 L2 距离。"""

    def __init__(self):
        self.bank = None  # (N, C) tensor

    def add(self, features: torch.Tensor) -> None:
        features = features.detach().float().cpu()
        self.bank = features if self.bank is None else torch.cat([self.bank, features], dim=0)

    def coreset_subsample(self, ratio: float) -> None:
        """v1 用随机下采样近似 coreset(后续 Plan 可替换为 k-center-greedy)。"""
        n = self.bank.shape[0]
        k = max(1, int(n * ratio))
        idx = torch.randperm(n)[:k]
        self.bank = self.bank[idx]

    def query(self, features: torch.Tensor) -> torch.Tensor:
        """返回每个 query 特征到 bank 的最小 L2 距离,形状 (M,)。"""
        d = torch.cdist(features.detach().float().cpu(), self.bank)  # (M, N)
        return d.min(dim=1).values
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_memory_bank.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add aoi/memory_bank.py tests/test_memory_bank.py && git commit -m "feat: 记忆库 MemoryBank(最近邻距离+随机coreset)"
```

---

### Task 3: 特征骨干 Backbone

**Files:**
- Create: `aoi/backbone.py`
- Test: `tests/test_backbone.py`

- [ ] **Step 1: 写失败测试**(用 `pretrained=False` 避免下载权重)

```python
# tests/test_backbone.py
import torch
from aoi.backbone import Backbone

def test_extract_shape():
    bb = Backbone(pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    f = bb.extract(x)
    assert f.ndim == 4
    assert f.shape[0] == 2          # batch 维保持
    assert f.shape[1] > 0           # 通道维 = 多层拼接
    assert f.shape[2] == f.shape[3] # 方形特征图
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_backbone.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.backbone'`

- [ ] **Step 3: 实现 `aoi/backbone.py`**

```python
import torch
import torch.nn.functional as F
import timm


class Backbone:
    """timm 多层特征提取器,把若干中间层上采样到同尺寸后按通道拼接。"""

    def __init__(self, name: str = "wide_resnet50_2", layers=(2, 3),
                 pretrained: bool = True, device: str = "cpu"):
        self.device = device
        self.model = (
            timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=layers)
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [0,1] -> (B, C_concat, h, w)"""
        x = x.to(self.device)
        feats = self.model(x)
        size = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats]
        return torch.cat(feats, dim=1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_backbone.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: Commit**

```bash
git add aoi/backbone.py tests/test_backbone.py && git commit -m "feat: timm 多层特征骨干 Backbone"
```

---

### Task 4: 纹理异常分支 TextureADBranch

**Files:**
- Create: `aoi/branches/texture_ad.py`
- Test: `tests/test_texture_ad.py`

- [ ] **Step 1: 写失败测试**(正常图打分应低于纯噪声图)

```python
# tests/test_texture_ad.py
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.types import BranchResult

def _branch():
    return TextureADBranch(backbone=Backbone(pretrained=False), coreset_ratio=1.0)

def test_infer_returns_result_with_map():
    b = _branch()
    normal = torch.full((4, 3, 64, 64), 0.5)
    b.fit(normal)
    r = b.infer(torch.full((1, 3, 64, 64), 0.5))
    assert isinstance(r, BranchResult)
    assert r.anomaly_map is not None and r.anomaly_map.ndim == 2
    assert r.latency_ms >= 0.0

def test_anomaly_scores_higher_than_normal():
    b = _branch()
    normal = torch.full((4, 3, 64, 64), 0.5)
    b.fit(normal)
    s_normal = b.infer(torch.full((1, 3, 64, 64), 0.5)).score
    s_noise = b.infer(torch.rand(1, 3, 64, 64)).score
    assert s_noise > s_normal
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_texture_ad.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.branches.texture_ad'`

- [ ] **Step 3: 实现 `aoi/branches/texture_ad.py`**

```python
import time
import torch
from ..types import BranchResult
from ..backbone import Backbone
from ..memory_bank import MemoryBank


def _to_patch_features(fmap: torch.Tensor):
    """(B,C,h,w) -> ((B*h*w, C), (h,w))"""
    b, c, h, w = fmap.shape
    feats = fmap.permute(0, 2, 3, 1).reshape(-1, c)
    return feats, (h, w)


class TextureADBranch:
    """PatchCore 风格:正常 patch 特征入记忆库,推理时取最近邻距离作异常分。"""
    defect_type = "appearance"

    def __init__(self, backbone: Backbone = None, coreset_ratio: float = 0.25):
        self.backbone = backbone or Backbone()
        self.bank = MemoryBank()
        self.coreset_ratio = coreset_ratio

    def fit(self, images: torch.Tensor) -> None:
        fmap = self.backbone.extract(images)
        feats, _ = _to_patch_features(fmap)
        self.bank.add(feats)
        if self.coreset_ratio < 1.0:
            self.bank.coreset_subsample(self.coreset_ratio)

    def infer(self, image: torch.Tensor) -> BranchResult:
        t0 = time.perf_counter()
        fmap = self.backbone.extract(image)
        feats, (h, w) = _to_patch_features(fmap)
        dist = self.bank.query(feats)                 # (h*w,)
        amap = dist.reshape(h, w).numpy()
        score = float(amap.max())
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=amap,
                            defect_type=self.defect_type, latency_ms=lat)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_texture_ad.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add aoi/branches/texture_ad.py tests/test_texture_ad.py && git commit -m "feat: 纹理异常分支 TextureADBranch"
```

---

### Task 5: 少样本适配器 FewShotAdapter

**Files:**
- Create: `aoi/fewshot.py`
- Test: `tests/test_fewshot.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fewshot.py
from aoi.fewshot import FewShotAdapter

def test_calibrate_separates_scores():
    t = FewShotAdapter._calibrate(normal_scores=[0.0, 1.0], defect_scores=[5.0, 6.0])
    assert 1.0 < t <= 5.0

class _FakeBranch:
    defect_type = "appearance"
    def __init__(self): self.fitted = False
    def fit(self, imgs): self.fitted = True
    def infer(self, img):
        from aoi.types import BranchResult
        # 用图像均值当分数:>0.5 视为缺陷
        return BranchResult(score=float(img.mean()))

def test_fit_fewshot_sets_threshold_and_predicts():
    import torch
    b = _FakeBranch()
    a = FewShotAdapter(b)
    normals = [torch.zeros(3, 8, 8) for _ in range(4)]      # 均值 0
    defects = [torch.ones(3, 8, 8) for _ in range(4)]       # 均值 1
    a.fit_fewshot(normals, defects)
    assert b.fitted is True
    _, is_def_normal = a.predict(torch.zeros(1, 3, 8, 8))
    _, is_def_defect = a.predict(torch.ones(1, 3, 8, 8))
    assert is_def_normal is False
    assert is_def_defect is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_fewshot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.fewshot'`

- [ ] **Step 3: 实现 `aoi/fewshot.py`**

```python
from typing import List, Tuple
import torch
from .types import BranchResult


class FewShotAdapter:
    """实现官方协议入口:用 100 正常 + 30 缺陷做迁移(建库 + 标定阈值)。"""

    def __init__(self, branch):
        self.branch = branch
        self.threshold = None

    def fit_fewshot(self, normal_images: List[torch.Tensor],
                    defect_images: List[torch.Tensor]) -> float:
        self.branch.fit(torch.stack(normal_images))
        normal_scores = [self.branch.infer(img.unsqueeze(0)).score for img in normal_images]
        defect_scores = [self.branch.infer(img.unsqueeze(0)).score for img in defect_images]
        self.threshold = self._calibrate(normal_scores, defect_scores)
        return self.threshold

    @staticmethod
    def _calibrate(normal_scores: List[float], defect_scores: List[float]) -> float:
        """在候选分数上选准确率最高的阈值;并列时取更大值(更保守)。"""
        candidates = sorted(set(normal_scores + defect_scores))
        best_t, best_acc = candidates[0], -1.0
        total = len(normal_scores) + len(defect_scores)
        for t in candidates:
            tp = sum(s >= t for s in defect_scores)
            tn = sum(s < t for s in normal_scores)
            acc = (tp + tn) / total
            if acc >= best_acc:
                best_acc, best_t = acc, t
        return best_t

    def predict(self, image: torch.Tensor) -> Tuple[BranchResult, bool]:
        r = self.branch.infer(image)
        is_defect = r.score >= self.threshold
        r.defect_type = self.branch.defect_type if is_defect else "normal"
        return r, is_defect
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_fewshot.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add aoi/fewshot.py tests/test_fewshot.py && git commit -m "feat: 少样本适配器 FewShotAdapter(fit_fewshot + 阈值标定)"
```

---

### Task 6: 评测指标 image_auroc

**Files:**
- Create: `eval/protocol.py`
- Test: `tests/test_protocol_metrics.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_protocol_metrics.py
import numpy as np
from eval.protocol import image_auroc

def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert abs(image_auroc(scores, labels) - 1.0) < 1e-9

def test_auroc_random_is_half():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 1, 0])
    assert abs(image_auroc(scores, labels) - 0.5) < 1e-9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_protocol_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.protocol'`

- [ ] **Step 3: 实现 `eval/protocol.py`(先只写 image_auroc)**

```python
import numpy as np


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """秩和法计算 image-level AUROC。labels: 1=缺陷, 0=正常。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_protocol_metrics.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: Commit**

```bash
git add eval/protocol.py tests/test_protocol_metrics.py && git commit -m "feat: image_auroc 评测指标"
```

---

### Task 7: 协议运行器 + 合成数据集集成测试

**Files:**
- Modify: `eval/protocol.py`(追加 `run_protocol`)
- Create: `tests/conftest.py`
- Create: `tests/test_protocol_integration.py`

- [ ] **Step 1: 写合成数据 fixture**

```python
# tests/conftest.py
import torch
import pytest


def _normal(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.full((3, 64, 64), 0.5) + 0.01 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1)
            for _ in range(n)]


def _defect(n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        img = torch.full((3, 64, 64), 0.5) + 0.01 * torch.randn(3, 64, 64, generator=g)
        x = int(torch.randint(0, 48, (1,), generator=g))
        y = int(torch.randint(0, 48, (1,), generator=g))
        img[:, y:y + 16, x:x + 16] = 1.0          # 亮斑当缺陷
        out.append(img.clamp(0, 1))
    return out


@pytest.fixture
def synth_dataset():
    return {"normal": _normal(20, seed=0), "defect": _defect(20, seed=1)}
```

- [ ] **Step 2: 写失败的集成测试**

```python
# tests/test_protocol_integration.py
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.fewshot import FewShotAdapter
from eval.protocol import run_protocol

def test_run_protocol_separates_synthetic(synth_dataset):
    normals, defects = synth_dataset["normal"], synth_dataset["defect"]
    branch = TextureADBranch(backbone=Backbone(pretrained=False), coreset_ratio=1.0)
    adapter = FewShotAdapter(branch)
    test_imgs = normals[10:] + defects[10:]
    test_labels = [0] * 10 + [1] * 10
    metrics = run_protocol(adapter, normals[:10], defects[:10], test_imgs, test_labels)
    assert metrics["auroc"] > 0.8
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["latency_ms_mean"] >= 0.0
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python -m pytest tests/test_protocol_integration.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_protocol'`

- [ ] **Step 4: 在 `eval/protocol.py` 追加 `run_protocol`**

```python
def run_protocol(adapter, normals_fit, defects_fit, test_images, test_labels):
    """复刻官方协议:fit_fewshot 迁移后在测试集上算 AUROC/准确率/平均延时。"""
    adapter.fit_fewshot(normals_fit, defects_fit)
    scores, lats, preds = [], [], []
    for img in test_images:
        r, is_def = adapter.predict(img.unsqueeze(0))
        scores.append(r.score)
        lats.append(r.latency_ms)
        preds.append(int(is_def))
    scores = np.array(scores)
    labels = np.array(test_labels)
    preds = np.array(preds)
    return {
        "auroc": image_auroc(scores, labels),
        "accuracy": float((preds == labels).mean()),
        "latency_ms_mean": float(np.mean(lats)),
    }
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_protocol_integration.py -v`
Expected: PASS(1 passed)

- [ ] **Step 6: 跑全量测试确保无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add eval/protocol.py tests/conftest.py tests/test_protocol_integration.py && git commit -m "feat: run_protocol 协议运行器 + 合成数据集成测试"
```

---

### Task 8: MVTec 加载器 + 真实数据冒烟脚本

**Files:**
- Create: `eval/mvtec.py`
- Create: `scripts/run_mvtec_smoke.py`
- Test: `tests/test_mvtec_loader.py`

**说明:** MVTec AD 需手动下载(官网/HuggingFace 镜像)解压到 `data/mvtec/<category>/`,结构为 `train/good/*.png`、`test/<defect>/*.png`、`test/good/*.png`。本 task 的单测用临时假目录验证加载逻辑,不依赖真实下载;真实冒烟由脚本手动跑。

- [ ] **Step 1: 写失败测试(用 tmp_path 造假数据集)**

```python
# tests/test_mvtec_loader.py
from PIL import Image
import numpy as np
from eval.mvtec import load_category

def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(path)

def test_load_category_splits(tmp_path):
    root = tmp_path / "mvtec" / "bottle"
    for i in range(5): _img(root / "train" / "good" / f"{i}.png")
    for i in range(3): _img(root / "test" / "good" / f"{i}.png")
    for i in range(4): _img(root / "test" / "broken" / f"{i}.png")
    data = load_category(str(root))
    assert len(data["train_normal"]) == 5
    assert len(data["test_normal"]) == 3
    assert len(data["test_defect"]) == 4
    assert data["train_normal"][0].shape == (3, 64, 64)   # CHW float[0,1] tensor
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_mvtec_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.mvtec'`

- [ ] **Step 3: 实现 `eval/mvtec.py`**

```python
from pathlib import Path
import torch
import numpy as np
from PIL import Image


def _load_img(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0      # H,W,3
    return torch.from_numpy(arr).permute(2, 0, 1)        # 3,H,W


def load_category(root: str, size: int = 256) -> dict:
    """读取 MVTec 单类别,返回 train_normal / test_normal / test_defect 张量列表。"""
    root = Path(root)
    train_normal = [_load_img(p, size) for p in sorted((root / "train" / "good").glob("*.png"))]
    test_normal = [_load_img(p, size) for p in sorted((root / "test" / "good").glob("*.png"))]
    test_defect = []
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            test_defect += [_load_img(p, size) for p in sorted(sub.glob("*.png"))]
    return {"train_normal": train_normal, "test_normal": test_normal, "test_defect": test_defect}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_mvtec_loader.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: 写真实数据冒烟脚本 `scripts/run_mvtec_smoke.py`**

```python
"""真实 MVTec 冒烟:python scripts/run_mvtec_smoke.py data/mvtec/bottle
按官方协议抽 100 正常 + 30 缺陷迁移,在剩余测试样本上输出 AUROC/准确率/延时。"""
import sys
import random
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.fewshot import FewShotAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category

def main(root):
    data = load_category(root)
    random.seed(0)
    normals = data["train_normal"]
    fit_normal = normals[:100] if len(normals) >= 100 else normals
    fit_defect = data["test_defect"][:30]
    test_imgs = data["test_normal"] + data["test_defect"][30:]
    test_labels = [0] * len(data["test_normal"]) + [1] * len(data["test_defect"][30:])
    branch = TextureADBranch(backbone=Backbone(pretrained=True, device="cuda"), coreset_ratio=0.25)
    adapter = FewShotAdapter(branch)
    print(run_protocol(adapter, fit_normal, fit_defect, test_imgs, test_labels))

if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 6: (有数据时)手动冒烟验证**

Run: `python scripts/run_mvtec_smoke.py data/mvtec/bottle`
Expected: 打印形如 `{'auroc': 0.9x, 'accuracy': 0.8x, 'latency_ms_mean': ...}`(pretrained 骨干 + 真实数据,AUROC 应显著 >0.8)

- [ ] **Step 7: Commit**

```bash
git add eval/mvtec.py scripts/run_mvtec_smoke.py tests/test_mvtec_loader.py && git commit -m "feat: MVTec 加载器 + 真实数据冒烟脚本"
```

---

## Self-Review

**Spec 覆盖检查:**
- 交付物=预训练权重+`fit_fewshot`代码 → Task 5 `FewShotAdapter.fit_fewshot` ✅
- PatchCore 记忆库(无梯度迁移)→ Task 2/4 ✅
- 官方协议复现 + 准确率/延时 → Task 6/7 ✅
- 数据集策略(MVTec 加载 + 模拟协议)→ Task 7(合成)/Task 8(MVTec)✅
- 跨域泛化测试 → 由 `run_protocol` 支持,真实跨类评测在 Plan 2+ 扩展(本 plan 提供运行器基础)。
- 未覆盖(刻意留给后续 plan):零样本分支、缺件/逻辑/尺寸分支、编排器、融合、主动学习、LLM 冷路径、OpenVINO、Demo、文档 → Plan 2~5。

**占位符扫描:** 无 TBD/TODO;每个代码步骤含完整可运行代码。

**类型一致性:** `BranchResult` 字段(score/anomaly_map/boxes/defect_type/latency_ms)在 Task 1 定义,Task 4/5/7 一致使用;`Branch` 隐式接口 `fit(images)` + `infer(image)->BranchResult` + 类属性 `defect_type`,Task 4 实现、Task 5 `_FakeBranch` 与 `FewShotAdapter` 一致调用;`run_protocol` 签名在 Task 7 定义并被 Task 8 脚本一致调用。✅

**范围:** 本 plan 聚焦"可运行评分闭环"单一子系统,独立可测、可交付。

# Plan 5b/5c/5d — LLM报告 + 尺寸分支 + 交付文档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** 收尾全项目:① 冷路径缺陷报告生成(可注入 LLM,默认模板);② 尺寸测量分支;③ 三份交付文档(可解释性/使用说明/结果报告)。

**Architecture:** `aoi/report.py` 从 `BranchResult` 抽结构化事实并格式化为中文报告;`DefectReporter` 可注入 `llm_fn(prompt)->str` 生成更自然报告(默认走模板,无 LLM 依赖)。`aoi/branches/dimension_ad.py` 用前景面积(偏离背景的像素数)与正常分布比较,捕捉尺寸偏差,接口同其它分支。文档直接由控制者撰写,基于真实实测结果。

**Tech Stack:** Python 3.8 · PyTorch · numpy · pytest

**工作目录:** 相对 `/home/srj/yolo/aoi_inspect/`。分支 `plan5-finish`。每 task 一提交。

---

### Task 1(5b): 缺陷报告 report

**Files:** Create `aoi/report.py`; Test `tests/test_report.py`

- [ ] **Step 1: 失败测试**

```python
# tests/test_report.py
import numpy as np
from aoi.report import summarize_detection, format_report, DefectReporter
from aoi.types import BranchResult


def test_summarize_extracts_peak_cell():
    amap = np.zeros((4, 4)); amap[2, 3] = 9.0
    r = BranchResult(score=1.5, anomaly_map=amap, defect_type="structural")
    facts = summarize_detection(r, is_defect=True)
    assert facts["is_defect"] is True
    assert facts["defect_type"] == "structural"
    assert facts["peak_cell"] == (2, 3)
    assert facts["grid"] == (4, 4)


def test_format_report_normal_vs_defect():
    normal = format_report({"is_defect": False, "defect_type": "normal", "score": 0.1})
    assert "正常" in normal
    defect = format_report({"is_defect": True, "defect_type": "appearance", "score": 2.0,
                            "peak_cell": (1, 1), "grid": (8, 8)})
    assert "缺陷" in defect and "appearance" in defect


def test_reporter_template_and_llm():
    r = BranchResult(score=2.0, defect_type="appearance")
    # 默认模板
    txt = DefectReporter().report(r, is_defect=True)
    assert "缺陷" in txt
    # 注入 LLM
    captured = {}
    def fake_llm(prompt):
        captured["prompt"] = prompt
        return "LLM报告"
    out = DefectReporter(llm_fn=fake_llm).report(r, is_defect=True)
    assert out == "LLM报告"
    assert "appearance" in captured["prompt"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_report.py -v` → FAIL (No module named 'aoi.report')

- [ ] **Step 3: 实现 `aoi/report.py`**

```python
import numpy as np


def summarize_detection(result, is_defect) -> dict:
    """从 BranchResult 抽结构化事实。"""
    facts = {
        "is_defect": bool(is_defect),
        "defect_type": result.defect_type,
        "score": float(result.score),
    }
    if result.anomaly_map is not None:
        a = np.asarray(result.anomaly_map)
        idx = np.unravel_index(int(a.argmax()), a.shape)
        facts["peak_cell"] = (int(idx[0]), int(idx[1]))
        facts["grid"] = (int(a.shape[0]), int(a.shape[1]))
    return facts


def format_report(facts: dict) -> str:
    """结构化事实 -> 中文缺陷报告文本。"""
    if not facts["is_defect"]:
        return f"检测结果:正常(异常分 {facts['score']:.3f})。"
    loc = ""
    if "peak_cell" in facts:
        r, c = facts["peak_cell"]
        gh, gw = facts["grid"]
        loc = f",最可疑区域位于 {gh}×{gw} 网格第 ({r},{c}) 格"
    return (f"检测结果:缺陷(类型:{facts['defect_type']},异常分 {facts['score']:.3f}){loc}。"
            "建议人工复核该区域。")


class DefectReporter:
    """冷路径缺陷报告:默认模板;可注入 llm_fn(prompt)->str 生成更自然报告。"""

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def report(self, result, is_defect) -> str:
        facts = summarize_detection(result, is_defect)
        if self.llm_fn is None:
            return format_report(facts)
        prompt = f"根据以下工业质检结果写一段简洁的中文缺陷报告:{facts}"
        return self.llm_fn(prompt)
```

- [ ] **Step 4: 跑测试通过(3 passed)+ 全量无回归;Commit**

```bash
python -m pytest tests/test_report.py -v && python -m pytest
git add aoi/report.py tests/test_report.py && git commit -m "feat: 冷路径缺陷报告 report(模板+可注入LLM)"
```

---

### Task 2(5c): 尺寸测量分支 DimensionADBranch

**Files:** Create `aoi/branches/dimension_ad.py`; Test `tests/test_dimension_ad.py`

- [ ] **Step 1: 失败测试**

```python
# tests/test_dimension_ad.py
import pytest
import torch
from aoi.branches.dimension_ad import DimensionADBranch


def _square(size):
    img = torch.full((1, 3, 64, 64), 0.5)
    img[:, :, :size, :size] = 1.0
    return img


def test_flags_size_deviation():
    b = DimensionADBranch()
    b.fit(torch.cat([_square(16) for _ in range(4)], dim=0))
    s_normal = b.infer(_square(16)).score
    s_big = b.infer(_square(24)).score
    assert s_big > s_normal

def test_result_fields_and_batch_guard():
    b = DimensionADBranch()
    b.fit(torch.cat([_square(16) for _ in range(4)], dim=0))
    r = b.infer(_square(16))
    assert r.defect_type == "dimension"
    assert r.score >= 0.0
    with pytest.raises(AssertionError):
        b.infer(torch.zeros(2, 3, 64, 64))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_dimension_ad.py -v` → FAIL (No module)

- [ ] **Step 3: 实现 `aoi/branches/dimension_ad.py`**

```python
import time
import torch
from ..types import BranchResult


class DimensionADBranch:
    """尺寸偏差:量前景面积(偏离背景中位数的像素数),与正常面积分布比较。"""
    defect_type = "dimension"

    def __init__(self, dev_thresh: float = 0.2):
        self.dev_thresh = dev_thresh
        self.mean = None
        self.std = None

    def _area(self, image: torch.Tensor) -> float:
        """image: (1,3,H,W) -> 前景像素数。"""
        gray = image[0].mean(dim=0)                       # H,W
        bg = gray.median()
        return float((gray - bg).abs().gt(self.dev_thresh).sum().item())

    def fit(self, images: torch.Tensor) -> None:
        areas = torch.tensor([self._area(images[i:i + 1]) for i in range(images.shape[0])])
        self.mean = float(areas.mean())
        self.std = float(areas.std()) if images.shape[0] > 1 else 0.0

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        area = self._area(image)
        std = self.std if self.std > 1e-9 else 1.0
        score = abs(area - self.mean) / std
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, defect_type=self.defect_type, latency_ms=lat)
```

- [ ] **Step 4: 跑测试通过(2 passed)+ 全量无回归;Commit**

```bash
python -m pytest tests/test_dimension_ad.py -v && python -m pytest
git add aoi/branches/dimension_ad.py tests/test_dimension_ad.py && git commit -m "feat: 尺寸测量分支 DimensionADBranch"
```

---

### Task 3(5d): 三份交付文档

由控制者撰写(非 TDD),基于真实实测结果:
- `docs/delivery/可解释性文档.md` — 方案选型依据 + 论据(各分支原理、为何记忆库/位置感知/零样本、实测证据表)。
- `docs/delivery/使用说明.md` — 安装、数据准备、训练/评测/Demo/基准命令、API。
- `docs/delivery/结果报告.md` — 公开集结果(AUROC/准确率/延时表)、消融、跨域泛化、复现实验方法。

- [ ] 控制者撰写三份文档并提交。

---

## Self-Review

**Spec 覆盖:** "LLM 冷路径(报告/可解释)" → Task 1 ✅;"尺寸测量分支" → Task 2 ✅;"三份交付文档" → Task 3 ✅。

**类型一致性:** `DimensionADBranch` 实现 fit/infer/defect_type 同 Branch 接口;`DefectReporter.report(result, is_defect)` 接受 BranchResult。✅

**范围(YAGNI):** 报告默认模板(无强制 LLM 依赖,LLM 可注入);尺寸用简单前景面积(不引 opencv);真实 LLM API 接线为可选脚本,不入测试核心。OpenVINO CPU 导出仍延后(GPU 已达标 <200ms)。

---

## 代码审查后续项(最终审查记录)

最终审查 **Approve,无 Critical**。已即时修复 1 个 Important:

**已修复:** `DimensionADBranch._area` 原用全局中位数估背景,前景>50% 时中位数翻转 → 大缺陷反测出小面积被漏检。改为**边缘像素中位数估背景**(对大前景鲁棒);补 `test_large_foreground_not_inverted` 回归。另:`score` 加注释说明"仅幅度"。

**登记延后:**
- [ ] 尺寸分支可改为**有符号**偏差以区分偏大/偏小(当前仅幅度,赛题异常检测够用)。
- [ ] `summarize_detection` 假设 anomaly_map 为 2D(当前所有分支均 2D),可加 ndim 防御。
- [ ] `DimensionADBranch` std 下限在 fit/infer 两处处理,可统一到 fit。
- [ ] `DefectReporter` LLM prompt 用 dict repr 拼接,可改 JSON/复用 format_report 作事实种子。

# Plan 4 — 主动学习反馈闭环 + Gradio Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现赛题第三支柱"用户反馈驱动的优化"——操作员标记漏检/误检后系统在线更新;并提供一个 Gradio Web Demo 把"检测→热力图→反馈→在线更新"全流程可视化。

**Architecture:** `ActiveLearningLoop` 维护当前正常/缺陷样本集,包一个适配器(`FewShotAdapter` 或 `MultiBranchAdapter`)。操作员反馈一张图的真实标签 → 把它加入对应集合 → **重跑 `adapter.fit_fewshot`**(记忆库方法无需梯度训练,重建库+重标定阈值即完成"在线更新")。`overlay_heatmap` 把异常图叠到原图上做可视化。Gradio 脚本把三者串成交互界面。全部复用 Plan 1–3 的部件,不改其接口。

**Tech Stack:** Python 3.8 · PyTorch 2.4 · numpy · Pillow · gradio · pytest

**工作目录:** 路径相对 `/home/srj/yolo/aoi_inspect/`。建议在分支 `plan4-active-learning-demo` 上实现。每个 task 末尾提交。

---

### Task 1: 主动学习闭环 ActiveLearningLoop

**Files:**
- Create: `aoi/active_learning.py`
- Test: `tests/test_active_learning.py`

- [ ] **Step 1: 写失败测试**(反馈一张漏检图后应被召回)

```python
# tests/test_active_learning.py
import torch
from aoi.active_learning import ActiveLearningLoop
from aoi.fewshot import FewShotAdapter
from aoi.types import BranchResult


class _MeanBranch:
    """分数 = 图像均值(正常≈0,缺陷≈1)。"""
    defect_type = "appearance"
    def fit(self, images):
        return None
    def infer(self, image):
        return BranchResult(score=float(image.mean()), defect_type=self.defect_type)


def test_feedback_recovers_missed_defect():
    loop = ActiveLearningLoop(
        FewShotAdapter(_MeanBranch()),
        normal_images=[torch.zeros(3, 8, 8) for _ in range(4)],
        defect_images=[torch.ones(3, 8, 8) for _ in range(4)],
    )
    weak = torch.full((3, 8, 8), 0.4)               # 弱缺陷,初始阈值=1 → 漏检
    _, before = loop.predict(weak.unsqueeze(0))
    assert before is False
    n_norm, n_def = loop.feedback(weak, is_defect=True)   # 操作员标记漏检
    assert (n_norm, n_def) == (4, 5)
    _, after = loop.predict(weak.unsqueeze(0))
    assert after is True                             # 反馈后被召回


def test_feedback_grows_normal_set():
    loop = ActiveLearningLoop(
        FewShotAdapter(_MeanBranch()),
        normal_images=[torch.zeros(3, 8, 8) for _ in range(4)],
        defect_images=[torch.ones(3, 8, 8) for _ in range(4)],
    )
    n_norm, n_def = loop.feedback(torch.zeros(3, 8, 8), is_defect=False)
    assert (n_norm, n_def) == (5, 4)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_active_learning.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.active_learning'`

- [ ] **Step 3: 实现 `aoi/active_learning.py`**

```python
class ActiveLearningLoop:
    """主动学习闭环:维护正常/缺陷样本集,操作员反馈后重跑少样本适配
    (记忆库方法无需梯度训练,重建库 + 重标定阈值即完成在线更新)。"""

    def __init__(self, adapter, normal_images, defect_images):
        self.adapter = adapter
        self.normals = list(normal_images)
        self.defects = list(defect_images)
        self.adapter.fit_fewshot(self.normals, self.defects)

    def predict(self, image):
        """image: (1,3,H,W) -> (BranchResult, is_defect)"""
        return self.adapter.predict(image)

    def feedback(self, image, is_defect):
        """image: (3,H,W) 单图;is_defect=操作员判定的真实标签。
        返回更新后的 (正常集大小, 缺陷集大小)。"""
        (self.defects if is_defect else self.normals).append(image)
        self.adapter.fit_fewshot(self.normals, self.defects)
        return len(self.normals), len(self.defects)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_active_learning.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 全量测试无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add aoi/active_learning.py tests/test_active_learning.py && git commit -m "feat: 主动学习反馈闭环 ActiveLearningLoop"
```

---

### Task 2: 热力图叠加 overlay_heatmap

**Files:**
- Create: `aoi/viz.py`
- Test: `tests/test_viz.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_viz.py
import torch
import numpy as np
from PIL import Image
from aoi.viz import overlay_heatmap

def test_overlay_returns_image_matching_input_size():
    img = torch.full((3, 32, 32), 0.5)
    amap = np.zeros((4, 4), dtype=float)
    amap[0, 0] = 1.0
    out = overlay_heatmap(img, amap)
    assert isinstance(out, Image.Image)
    assert out.size == (32, 32)          # (W, H)
    assert out.mode == "RGB"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_viz.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.viz'`

- [ ] **Step 3: 实现 `aoi/viz.py`**

```python
import numpy as np
from PIL import Image


def overlay_heatmap(image_chw, amap, alpha: float = 0.5) -> Image.Image:
    """把异常图以红色半透明叠到原图上。
    image_chw: (3,H,W) [0,1] 的 tensor 或 array;amap: (h,w) 异常图。返回 RGB PIL.Image。"""
    img = np.asarray(image_chw, dtype=np.float32).transpose(1, 2, 0)   # H,W,3
    h, w = img.shape[:2]
    a = np.asarray(amap, dtype=np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)                     # 归一化到 [0,1]
    a_resized = np.asarray(Image.fromarray((a * 255).astype("uint8")).resize((w, h))) / 255.0
    heat = np.zeros_like(img)
    heat[..., 0] = a_resized                                           # 红色通道
    out = (1.0 - alpha) * img + alpha * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype("uint8"))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_viz.py -v`
Expected: PASS(1 passed)

- [ ] **Step 5: Commit**

```bash
git add aoi/viz.py tests/test_viz.py && git commit -m "feat: 异常热力图叠加 overlay_heatmap"
```

---

### Task 3: Gradio Web Demo

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/demo_app.py`

**说明:** Demo 是交互脚本,不做 pytest 单测(仅语法检查 + 手动启动)。它预加载一个 MVTec 类别建闭环,支持:上传图 → 检测(热力图+判决)→ 操作员标注实际情况 → 提交反馈在线更新。

- [ ] **Step 1: 追加 gradio 依赖到 `requirements.txt`**

末尾追加:
```
gradio>=4.0
```

- [ ] **Step 2: 安装并验证**

Run: `pip install "gradio>=4.0" && python -c "import gradio; print('gradio', gradio.__version__)"`
Expected: 打印版本号。(失败则报告 NEEDS_CONTEXT)

- [ ] **Step 3: 实现 `scripts/demo_app.py`**

```python
"""AOI 质检 Demo:python scripts/demo_app.py data/mvtec/bottle
上传图 → 检测(热力图+判决)→ 操作员标注 → 提交反馈在线更新。"""
import sys
import numpy as np
import torch
import gradio as gr
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.multibranch import MultiBranchAdapter
from aoi.active_learning import ActiveLearningLoop
from aoi.viz import overlay_heatmap
from eval.mvtec import load_category

LOOP = None


def _to_tensor(pil):
    arr = np.asarray(pil.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def predict_fn(pil):
    t = _to_tensor(pil)
    r, is_def = LOOP.predict(t.unsqueeze(0))
    overlay = overlay_heatmap(t, r.anomaly_map) if r.anomaly_map is not None else pil
    return overlay, f"{'缺陷' if is_def else '正常'}  (score={r.score:.3f}, type={r.defect_type})"


def feedback_fn(pil, actual):
    LOOP.feedback(_to_tensor(pil), is_defect=(actual == "缺陷"))
    return f"已反馈「{actual}」;正常库={len(LOOP.normals)} 缺陷库={len(LOOP.defects)}。请重新点击「检测」查看更新效果。"


def main(root):
    global LOOP
    data = load_category(root)
    bb = Backbone(pretrained=True, device="cuda")
    adapter = MultiBranchAdapter([
        TextureADBranch(backbone=bb, coreset_ratio=0.25),
        StructuralADBranch(backbone=bb, grid_size=16),
    ])
    LOOP = ActiveLearningLoop(adapter, data["train_normal"][:100], data["test_defect"][:30])
    with gr.Blocks() as demo:
        gr.Markdown("# AOI 实时质检 Demo:检测 + 用户反馈闭环")
        with gr.Row():
            inp = gr.Image(type="pil", label="上传待检图")
            out_img = gr.Image(label="异常热力图")
        out_lbl = gr.Textbox(label="判决")
        gr.Button("检测").click(predict_fn, inp, [out_img, out_lbl])
        actual = gr.Radio(["正常", "缺陷"], label="操作员标注实际情况")
        fb_out = gr.Textbox(label="反馈结果")
        gr.Button("提交反馈(在线更新)").click(feedback_fn, [inp, actual], fb_out)
    demo.launch()


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: 语法检查**

Run: `python -c "import ast; ast.parse(open('scripts/demo_app.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 5: 全量测试无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/demo_app.py && git commit -m "feat: Gradio 质检 Demo(检测+反馈闭环)"
```

---

## Self-Review

**Spec 覆盖:** 设计 spec 的"用户反馈驱动优化(主动学习)"→ Task 1 ✅;"Gradio Web Demo:上传→热力图→标漏检→看在线更新"→ Task 3 ✅;可解释性可视化(热力图)→ Task 2 ✅。

**占位符扫描:** 无 TBD/TODO;每步含完整代码。

**类型一致性:** `ActiveLearningLoop` 复用适配器的 `fit_fewshot(normal_list, defect_list)` 与 `predict((1,3,H,W))->(BranchResult,bool)`(Plan 1/3 已实现);`feedback(image_chw, is_defect)` 接受 (3,H,W) 与 fit_fewshot 列表元素一致;`overlay_heatmap(image_chw,(3,H,W); amap (h,w))` 被 demo 一致调用。✅

**范围(YAGNI):**
- **纳入**:反馈闭环 + 热力图叠加 + Gradio Demo。
- **延后**:复杂的增量式记忆库更新(当前用"重跑 fit_fewshot"全量重建,Demo 规模够用)→ 若需大规模在线可在 Plan 5 优化;视频流输入 → Plan 5;RL 式策略 → 明确不做(spec 已定主动学习)。

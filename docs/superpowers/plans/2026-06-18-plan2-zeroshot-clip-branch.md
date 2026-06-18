# Plan 2 — 零样本 CLIP 异常检测分支 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个**零样本**异常检测分支(基于 CLIP 文本提示),无需任何正常样本即可对图像打异常分,与 Plan 1 的记忆库分支互补(冷启动能力)。

**Architecture:** 把"CLIP 编码器"与"分支判别逻辑"解耦:`CLIPEncoder` 用 open_clip 产出 L2 归一化的图像/文本嵌入(薄适配层,手动 smoke 验证);`ZeroShotCLIPBranch` 拿注入的 encoder,用"正常 vs 异常"文本提示组与图像嵌入算相似度,softmax 得异常概率作分数。分支逻辑用假 encoder 完整 TDD,不依赖联网下载权重。复用 Plan 1 的 `Branch` 接口(`fit/infer/defect_type`)和 `BranchResult`、`FewShotAdapter`、`run_protocol`。

**Tech Stack:** Python 3.8 · PyTorch 2.4 · open_clip_torch · numpy · pytest

**工作目录:** 所有路径相对 `/home/srj/yolo/aoi_inspect/`。建议在新分支 `plan2-zeroshot-clip` 上实现。每个 task 末尾提交。

---

### Task 1: 安装 open_clip 依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 追加依赖到 `requirements.txt`**

在现有内容后追加一行:
```
open_clip_torch>=2.20
```

- [ ] **Step 2: 安装并验证可导入**

Run: `pip install "open_clip_torch>=2.20" && python -c "import open_clip; print('open_clip', open_clip.__version__)"`
Expected: 打印版本号,无报错。
(若安装失败/被限,报告 NEEDS_CONTEXT,不要静默改动。)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt && git commit -m "chore: 增加 open_clip_torch 依赖"
```

---

### Task 2: ZeroShotCLIPBranch 分支逻辑(TDD,假 encoder)

**Files:**
- Create: `aoi/branches/zeroshot_clip.py`
- Test: `tests/test_zeroshot_clip.py`

**接口约定:** encoder 需提供 `encode_text(list[str]) -> (T, D) tensor`(已 L2 归一化)与 `encode_image((1,3,H,W)) -> (1, D) tensor`(已归一化)。分支用点积算相似度,故假设嵌入已归一化。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_zeroshot_clip.py
import torch
from aoi.branches.zeroshot_clip import ZeroShotCLIPBranch
from aoi.types import BranchResult


class FakeEncoder:
    """2 维嵌入:正常轴[1,0],异常轴[0,1]。文本含 defect/damaged/anomaly 归到异常轴;
    图像按均值映射:亮(均值大)→异常轴。"""
    def encode_text(self, prompts):
        embs = []
        for p in prompts:
            if any(k in p for k in ("defect", "damaged", "anomaly")):
                embs.append([0.0, 1.0])
            else:
                embs.append([1.0, 0.0])
        return torch.tensor(embs)

    def encode_image(self, image):
        m = float(image.mean())
        v = torch.tensor([[1.0 - m, m]])
        return v / v.norm()


def test_infer_returns_result_in_unit_range():
    b = ZeroShotCLIPBranch(FakeEncoder(), class_name="bottle")
    r = b.infer(torch.zeros(1, 3, 8, 8))
    assert isinstance(r, BranchResult)
    assert 0.0 <= r.score <= 1.0
    assert r.latency_ms >= 0.0

def test_abnormal_image_scores_higher():
    b = ZeroShotCLIPBranch(FakeEncoder(), class_name="bottle")
    s_normal = b.infer(torch.zeros(1, 3, 8, 8)).score   # 均值0 → 正常
    s_defect = b.infer(torch.ones(1, 3, 8, 8)).score    # 均值1 → 异常
    assert s_defect > s_normal

def test_fit_is_noop():
    b = ZeroShotCLIPBranch(FakeEncoder())
    assert b.fit(torch.zeros(2, 3, 8, 8)) is None

def test_infer_rejects_batch():
    import pytest
    b = ZeroShotCLIPBranch(FakeEncoder())
    with pytest.raises(AssertionError):
        b.infer(torch.zeros(2, 3, 8, 8))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_zeroshot_clip.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi.branches.zeroshot_clip'`

- [ ] **Step 3: 实现 `aoi/branches/zeroshot_clip.py`**

```python
import time
import torch
import torch.nn.functional as F
from ..types import BranchResult

DEFAULT_NORMAL = [
    "a photo of a normal {}",
    "a photo of a flawless {}",
    "a photo of a {} without defect",
]
DEFAULT_ABNORMAL = [
    "a photo of a {} with defect",
    "a photo of a damaged {}",
    "a photo of a {} with anomaly",
]


class ZeroShotCLIPBranch:
    """零样本:用 CLIP 正常/异常文本提示与图像嵌入的相似度,softmax 得异常概率。"""
    defect_type = "appearance"

    def __init__(self, encoder, class_name: str = "object",
                 normal_prompts=None, abnormal_prompts=None, temperature: float = 0.01):
        self.encoder = encoder
        normal = [p.format(class_name) for p in (normal_prompts or DEFAULT_NORMAL)]
        abnormal = [p.format(class_name) for p in (abnormal_prompts or DEFAULT_ABNORMAL)]
        self._normal_emb = encoder.encode_text(normal)        # (Tn, D)
        self._abnormal_emb = encoder.encode_text(abnormal)    # (Ta, D)
        self.temperature = temperature

    def fit(self, images: torch.Tensor):
        """零样本分支无需训练。"""
        return None

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        img_emb = self.encoder.encode_image(image)            # (1, D)
        sim_n = (img_emb @ self._normal_emb.T).mean()
        sim_a = (img_emb @ self._abnormal_emb.T).mean()
        logits = torch.stack([sim_n, sim_a]) / self.temperature
        score = float(F.softmax(logits, dim=0)[1])            # P(异常)
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=None,
                            defect_type=self.defect_type, latency_ms=lat)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_zeroshot_clip.py -v`
Expected: PASS(4 passed)

- [ ] **Step 5: 跑全量测试无回归**

Run: `python -m pytest`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add aoi/branches/zeroshot_clip.py tests/test_zeroshot_clip.py && git commit -m "feat: 零样本 CLIP 异常检测分支 ZeroShotCLIPBranch"
```

---

### Task 3: CLIPEncoder(open_clip 薄适配层)

**Files:**
- Create: `aoi/clip_encoder.py`

**说明:** CLIP 需要固定输入归一化与 224 分辨率。分支收到的是 [0,1] 张量,故 encoder 内部负责 resize + CLIP 均值方差归一化。这层依赖真实权重,不写 pytest 单测(联网+大权重),由 Task 4 的 smoke 脚本验证。

- [ ] **Step 1: 实现 `aoi/clip_encoder.py`**

```python
import torch
import torch.nn.functional as F
import open_clip

# OpenAI CLIP 预处理均值/方差
_MEAN = [0.48145466, 0.4578275, 0.40821073]
_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPEncoder:
    """open_clip 薄封装:输出 L2 归一化的图像/文本嵌入(CPU)。"""

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai", device: str = "cpu"):
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.eval().to(device)
        self.device = device
        self._mean = torch.tensor(_MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(_STD).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def encode_text(self, prompts):
        tok = self.tokenizer(prompts).to(self.device)
        emb = self.model.encode_text(tok)
        return F.normalize(emb, dim=-1).cpu()

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        x = F.interpolate(image.to(self.device), size=224, mode="bicubic", align_corners=False)
        x = (x - self._mean) / self._std
        emb = self.model.encode_image(x)
        return F.normalize(emb, dim=-1).cpu()
```

- [ ] **Step 2: 语法检查(不下载权重)**

Run: `python -c "import ast; ast.parse(open('aoi/clip_encoder.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: 跑全量测试确认无回归(不触发 CLIPEncoder 实例化)**

Run: `python -m pytest`
Expected: 全部 PASS(CLIPEncoder 未被任何单测导入/实例化)

- [ ] **Step 4: Commit**

```bash
git add aoi/clip_encoder.py && git commit -m "feat: CLIPEncoder(open_clip 图像/文本嵌入薄封装)"
```

---

### Task 4: 零样本 MVTec 冒烟脚本

**Files:**
- Create: `scripts/run_clip_smoke.py`

**说明:** 演示"零正常样本"下的零样本能力:用 CLIP 分支跑官方协议。`fit_fewshot` 中 `fit` 为空操作,少量正常+缺陷仅用于阈值标定;AUROC 本身与阈值无关,直接体现零样本判别力。真实数据存在时手动跑。

- [ ] **Step 1: 实现 `scripts/run_clip_smoke.py`**

```python
"""零样本 CLIP 冒烟:python scripts/run_clip_smoke.py data/mvtec/bottle [class_name]
用 CLIP 文本提示零样本检测,按官方协议输出 AUROC/准确率/延时。"""
import sys
from aoi.clip_encoder import CLIPEncoder
from aoi.branches.zeroshot_clip import ZeroShotCLIPBranch
from aoi.fewshot import FewShotAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category


def main(root, class_name="object"):
    data = load_category(root)
    encoder = CLIPEncoder(device="cuda")
    branch = ZeroShotCLIPBranch(encoder, class_name=class_name)
    adapter = FewShotAdapter(branch)
    # 零样本:fit 为空操作;少量样本仅用于阈值标定
    fit_normal = data["test_normal"][:10]
    fit_defect = data["test_defect"][:10]
    test_imgs = data["test_normal"][10:] + data["test_defect"][10:]
    test_labels = [0] * len(data["test_normal"][10:]) + [1] * len(data["test_defect"][10:])
    print(run_protocol(adapter, fit_normal, fit_defect, test_imgs, test_labels))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "object")
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('scripts/run_clip_smoke.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3:(有数据时)手动冒烟** — 无 MVTec 数据则 SKIP,不因缺数据判失败。

Run: `python scripts/run_clip_smoke.py data/mvtec/bottle bottle`
Expected: 打印 `{'auroc': ..., 'accuracy': ..., 'latency_ms_mean': ...}`(零样本 AUROC 通常 0.8+,视类别而定)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_clip_smoke.py && git commit -m "feat: 零样本 CLIP MVTec 冒烟脚本"
```

---

## Self-Review

**Spec 覆盖:** 设计 spec 的"零样本冷启动分支 AnomalyCLIP/WinCLIP" → Task 2/3 实现(image-level 零样本)✅。复用 Plan 1 的 Branch 接口/FewShotAdapter/run_protocol → Task 2/4 ✅。

**占位符扫描:** 无 TBD/TODO;每个代码步骤含完整代码。

**类型一致性:** `ZeroShotCLIPBranch` 实现 `fit(images)->None` / `infer((1,3,H,W))->BranchResult` / 类属性 `defect_type`,与 Plan 1 的 Branch 接口及 `FewShotAdapter`/`run_protocol` 调用一致;encoder 接口(`encode_text`/`encode_image`)在 `FakeEncoder`(Task 2)与 `CLIPEncoder`(Task 3)间一致。✅

**范围(YAGNI):** Plan 2 只做 image-level 零样本;**不做**像素级窗口聚合(WinCLIP 的 pixel-map)、不做 prompt 自动学习——留待后续增强。anomaly_map 暂为 None。

---

## 代码审查后续项(最终审查记录)

最终审查结论:**Approve,无 Critical**。已即时修复 1 项,其余登记延后:

**已修复:** 冒烟脚本默认类名从目录名推断(提示词更贴切)。

**登记延后:**
- [ ] **Plan 5(预处理)**:`CLIPEncoder.encode_image` 现对非方形图直接拉伸到 224;应改为 open_clip 标准的"短边缩放 + 中心裁剪"。当前管线 loader 已统一 resize 成方形,不触发;接入真实非方形 AOI 图前需修。
- [ ] **增强(准确率)**:零样本 prompt 为朴素模板(当前 AUROC 0.62–0.87);换 AnomalyCLIP 式可学习提示 / 更大 prompt 集成可显著提升。
- [ ] **校准**:temperature=0.01 使 score 趋近二值(对 AUROC 无碍,但与记忆库分支融合时需重新校准为可比概率)。
- [ ] **测试健壮性**:`FakeEncoder` 按英文子串(defect/damaged/anomaly)判正常/异常轴,改 prompt 时易误判;可改为按索引/显式标记。

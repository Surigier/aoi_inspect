"""赛场大图统一检测器(CompetitionLargeDetector):2500² 输入的生产路径。
核心 = EfficientAD 整图卷积(强检测 + 低延时);辅以色彩/尺寸/结构分支补齐 5 类缺陷覆盖。
可靠性加权 z-norm 软融合 → 二值判决 + 缺陷类型输出(最强分支=类型)。
延时 ≈ EfficientAD(~106ms@2060)+ 几个轻分支(几 ms),仍 <200ms。"""
import math
import torch
import torch.nn.functional as F
from .fusion import auroc, znorm
from .fewshot import FewShotAdapter
from .backbone import Backbone
from .tiled_efficientad import TiledEfficientAD
from .branches.color_ad import ColorADBranch
from .branches.dimension_ad import DimensionADBranch
from .branches.structural_ad import StructuralADBranch

WEIGHT_FLOOR = 0.05


def _down(img, size):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)


class _EADBranch:
    """EfficientAD 核心(整图卷积),映射到'外观缺陷'(一般异常的默认归类)。"""
    defect_type = "外观缺陷"

    def __init__(self, **kw):
        self.det = TiledEfficientAD(**kw)

    def fit(self, normals, defects):
        self.det.fit_fewshot(normals, defects)

    def score(self, img):
        return self.det._image_score(img)


class _AuxBranch:
    """轻量辅助分支(色彩/尺寸/结构),在下采样大图上跑,负责对应缺陷类型。"""
    def __init__(self, branch, defect_type, size):
        self.branch = branch
        self.defect_type = defect_type
        self.size = size

    def fit(self, normals, defects):
        self.branch.fit(torch.cat([_down(n, self.size) for n in normals], 0))

    def score(self, img):
        return self.branch.infer(_down(img, self.size)).score


class CompetitionLargeDetector:
    def __init__(self, device="cuda", aux_size=320, train_steps=10000):
        dev = device if torch.cuda.is_available() else "cpu"
        bb = Backbone(device=dev)
        self.branches = [
            _EADBranch(device=dev, train_steps=train_steps),
            _AuxBranch(ColorADBranch(grid_size=16), "色彩变化", aux_size),
            _AuxBranch(DimensionADBranch(), "尺寸偏差", aux_size),
            _AuxBranch(StructuralADBranch(backbone=bb, grid_size=16), "缺件/逻辑", aux_size),
        ]
        self.stats = []
        self.weights = []
        self.threshold = None

    def fit_fewshot(self, normals, defects):
        norm_scores, def_scores = [], []
        self.stats, self.weights = [], []
        for b in self.branches:
            b.fit(normals, defects)                       # 各分支单次拟合(EAD训练一次)
            ns = [b.score(x) for x in normals]
            ds = [b.score(d) for d in defects]
            m = sum(ns) / len(ns)
            s = (sum((x - m) ** 2 for x in ns) / len(ns)) ** 0.5
            self.stats.append((m, s))
            self.weights.append(max(WEIGHT_FLOOR,
                                    auroc(ns + ds, [0] * len(ns) + [1] * len(ds)) - 0.5))
            norm_scores.append(ns)
            def_scores.append(ds)
        nf = [self._fuse([norm_scores[j][i] for j in range(len(self.branches))])
              for i in range(len(normals))]
        df = [self._fuse([def_scores[j][i] for j in range(len(self.branches))])
              for i in range(len(defects))]
        self.threshold = FewShotAdapter._calibrate(nf, df)
        return self.threshold

    def _contribs(self, raws):
        return [w * (1.0 / (1.0 + math.exp(-znorm(r, m, s))))
                for r, (m, s), w in zip(raws, self.stats, self.weights)]

    def _fuse(self, raws):
        tot = sum(self.weights) or 1.0
        return sum(self._contribs(raws)) / tot

    def predict(self, img):
        raws = [b.score(img) for b in self.branches]
        fused = self._fuse(raws)
        is_def = bool(self.threshold is not None and fused >= self.threshold)
        contrib = self._contribs(raws)
        top = max(range(len(contrib)), key=lambda i: contrib[i])
        return {"score": fused, "is_defect": is_def,
                "defect_type": self.branches[top].defect_type if is_def else "normal"}

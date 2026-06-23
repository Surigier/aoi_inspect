"""全局+局部混合大图检测器。
全局分支:整图 resize→512 PatchCore(WRN50,抓纹理/色彩/尺寸/全局)。
局部分支:原生分辨率分块(ResNet18,抓缺件/外观小缺陷)。
融合:各分支 z-标准化(按 fit 正常分 μ/σ)→ 可靠性加权 sigmoid 软融合(复用 multibranch 思路)。
依据 MVTec AD2 实测:纯分块平均不如整图,但二者各擅一类缺陷 → 融合逼近 oracle。"""
import math
import torch
import torch.nn.functional as F
from .fusion import auroc, znorm
from .fewshot import FewShotAdapter
from .ensemble import default_adapter
from .tiled import TiledFewShotDetector

WEIGHT_FLOOR = 0.05         # 可靠性权重下限:防"好分支被噪声误判归零"(如 wallplugs)


def _resize(img, size=512):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)[0]


class _GlobalBranch:
    """整图 resize→320 的完整 ensemble(纹理+结构+判别头,现有proven配置),判别头吃30缺陷监督。
    全局只管粗/全局上下文(纹理/色彩/尺寸/缺件),小缺陷交局部分块 → 320 足够且显存安全。"""
    def __init__(self, backbone, size=320):
        self.size = size
        self.adapter = default_adapter(backbone)

    def fit(self, normals, defects):
        self.adapter.fit_fewshot([_resize(i, self.size) for i in normals],
                                 [_resize(d, self.size) for d in defects])

    def score(self, img):
        return self.adapter._fused(_resize(img, self.size).unsqueeze(0))[0]


class _LocalBranch:
    """原生分辨率分块(无监督,管局部小缺陷)。"""
    def __init__(self, backbone, **kw):
        self.det = TiledFewShotDetector(backbone, **kw)

    def fit(self, normals, defects):
        self.det.fit_fewshot(normals, defects)         # 建库;阈值此处不用

    def score(self, img):
        return self.det._image_score(img)[0]


class HybridDetector:
    def __init__(self, global_bb, local_bb, local_kw=None):
        self.g = _GlobalBranch(global_bb)
        self.l = _LocalBranch(local_bb, **(local_kw or {}))
        self.branches = [self.g, self.l]
        self.stats = []
        self.weights = []
        self.threshold = None

    def fit_fewshot(self, normals, defects):
        # 1) 留出法估各分支可靠性(对未见正常的可分性),加 floor 防误判归零
        k = max(1, len(normals) // 5)
        val_n, build_n = normals[:k], normals[k:]
        if not build_n:
            build_n, val_n = normals, normals
        self.weights = []
        for b in self.branches:
            b.fit(build_n, defects)
            vn = [b.score(x) for x in val_n]
            ds = [b.score(d) for d in defects]
            sep = auroc(vn + ds, [0] * len(vn) + [1] * len(ds))
            self.weights.append(max(WEIGHT_FLOOR, sep - 0.5))
        # 2) 全量重拟合 + 缓存分数估 μ/σ(每分支每图只算一次)
        self.stats = []
        norm_scores, def_scores = [], []
        for b in self.branches:
            b.fit(normals, defects)
            ns = [b.score(x) for x in normals]
            dsc = [b.score(d) for d in defects]
            m = sum(ns) / len(ns)
            s = (sum((x - m) ** 2 for x in ns) / len(ns)) ** 0.5
            self.stats.append((m, s))
            norm_scores.append(ns)
            def_scores.append(dsc)
        # 3) 在融合分上标阈值
        nf = [self._fuse_vec([norm_scores[j][i] for j in range(len(self.branches))])
              for i in range(len(normals))]
        df = [self._fuse_vec([def_scores[j][i] for j in range(len(self.branches))])
              for i in range(len(defects))]
        self.threshold = FewShotAdapter._calibrate(nf, df)
        return self.threshold

    def _fuse_vec(self, raw):
        tot = sum(self.weights) or 1.0
        num = sum(w * (1.0 / (1.0 + math.exp(-znorm(r, m, s))))
                  for r, (m, s), w in zip(raw, self.stats, self.weights))
        return num / tot

    def _fused(self, img):
        return self._fuse_vec([b.score(img) for b in self.branches])

    def predict(self, img):
        s = self._fused(img)
        return {"score": s, "is_defect": bool(self.threshold is not None and s >= self.threshold)}

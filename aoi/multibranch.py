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
        res.score = fused            # 让 AUROC/阈值都基于融合分(保留最异常分支的 anomaly_map)
        res.defect_type = res.defect_type if is_defect else "normal"
        return res, is_defect

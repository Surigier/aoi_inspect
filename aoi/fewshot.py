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
        """选**平衡准确率**(TPR+TNR)/2 最高的阈值。
        正常多、缺陷少时,最大化原始准确率会偏向"不报警"→漏检;平衡准确率消除该偏向,
        兼顾召回(AOI 早期拦截看重不漏)。并列时取更大阈值(更少误报)。"""
        candidates = sorted(set(normal_scores + defect_scores))
        n_pos = len(defect_scores)
        n_neg = len(normal_scores)
        best_t, best_bal = candidates[0], -1.0
        for t in candidates:
            tpr = sum(s >= t for s in defect_scores) / n_pos
            tnr = sum(s < t for s in normal_scores) / n_neg
            bal = (tpr + tnr) / 2.0
            if bal >= best_bal:
                best_bal, best_t = bal, t
        return best_t

    def predict(self, image: torch.Tensor) -> Tuple[BranchResult, bool]:
        r = self.branch.infer(image)
        is_defect = bool(r.score >= self.threshold)
        r.defect_type = self.branch.defect_type if is_defect else "normal"
        return r, is_defect

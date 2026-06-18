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
        is_defect = bool(r.score >= self.threshold)
        r.defect_type = self.branch.defect_type if is_defect else "normal"
        return r, is_defect

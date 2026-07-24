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


class _MaskAwareAdapter:
    """模拟CompetitionLargeDetector的fit_fewshot(normals, defects, defect_masks=...)
    三参数签名,验证ActiveLearningLoop正确地把掩膜一路带过去(不是只兼容旧的
    两参数记忆库适配器)。"""
    def __init__(self):
        self.fit_calls = []

    def fit_fewshot(self, normals, defects, defect_masks=None):
        self.fit_calls.append((len(normals), len(defects),
                               len(defect_masks) if defect_masks is not None else None))

    def predict(self, image):
        return None, False


def test_feedback_threads_defect_masks():
    import numpy as np
    adapter = _MaskAwareAdapter()
    loop = ActiveLearningLoop(
        adapter,
        normal_images=[torch.zeros(3, 8, 8) for _ in range(2)],
        defect_images=[torch.ones(3, 8, 8)],
        defect_masks=[np.ones((8, 8), "uint8")],
    )
    assert adapter.fit_calls[-1] == (2, 1, 1)          # 初始fit已带掩膜
    n_norm, n_def = loop.feedback(torch.ones(3, 8, 8), is_defect=True,
                                  mask=np.ones((8, 8), "uint8"))
    assert (n_norm, n_def) == (2, 2)
    assert adapter.fit_calls[-1] == (2, 2, 2)          # 反馈后掩膜数同步增长
    assert len(loop.masks) == 2

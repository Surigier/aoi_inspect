"""ActiveLearningLoop 只服务一种适配器契约:生产的 CompetitionLargeDetector。

单元测试不跑真检测器——一次 fit_fewshot 约 20 分钟、要 GPU、8×8 的测试图也撑不起
分块/SAM/DINO。所以用一个**按生产契约实现的最小替身**来锁契约本身:
参数怎么传、掩膜有没有同步增长、反馈有没有真的走快路径。
"""
import numpy as np
import torch
from aoi.active_learning import ActiveLearningLoop
from aoi.fewshot import FewShotAdapter


class _FakeDetector:
    """契约 = CompetitionLargeDetector:
        fit_fewshot(normals, defects, defect_masks=None, retrain_ead=True)
        predict(image) -> dict(score, is_defect, defect_type, _raws)
    打分用图像均值(正常≈0,缺陷≈1),阈值走生产同一套 _calibrate。"""

    def __init__(self):
        self.fit_calls = []
        self.threshold = None
        self._fb_defects = []          # 契约的一部分:操作员反馈的硬约束样本
        self._fb_normals = []

    def fit_fewshot(self, normals, defects, defect_masks=None, retrain_ead=True):
        self.fit_calls.append(dict(
            n=len(normals), d=len(defects),
            m=None if defect_masks is None else len(defect_masks),
            retrain_ead=retrain_ead))
        self.threshold = FewShotAdapter._calibrate(
            [float(x.mean()) for x in normals], [float(x.mean()) for x in defects])

    def predict(self, image):
        s = float(image.mean())
        is_def = bool(self.threshold is not None and s >= self.threshold)
        return {"score": s, "is_defect": is_def,
                "defect_type": "常见外观缺陷" if is_def else "normal", "_raws": None}


def _loop(masks=None):
    return ActiveLearningLoop(
        _FakeDetector(),
        normal_images=[torch.zeros(3, 8, 8) for _ in range(4)],
        defect_images=[torch.ones(3, 8, 8) for _ in range(4)],
        defect_masks=masks)


def test_feedback_recovers_missed_defect():
    loop = _loop()
    weak = torch.full((3, 8, 8), 0.4)                    # 弱缺陷,初始阈值=1 → 漏检
    assert loop.predict(weak.unsqueeze(0))["is_defect"] is False
    n_norm, n_def = loop.feedback(weak, is_defect=True)  # 操作员标记漏检
    assert (n_norm, n_def) == (4, 5)
    assert loop.predict(weak.unsqueeze(0))["is_defect"] is True    # 反馈后被召回


def test_feedback_grows_normal_set():
    loop = _loop()
    assert loop.feedback(torch.zeros(3, 8, 8), is_defect=False) == (5, 4)


def test_feedback_threads_defect_masks():
    loop = _loop(masks=[np.ones((8, 8), "uint8") for _ in range(4)])
    assert loop.adapter.fit_calls[-1]["m"] == 4          # 初始fit已带掩膜
    n_norm, n_def = loop.feedback(torch.ones(3, 8, 8), is_defect=True,
                                  mask=np.ones((8, 8), "uint8"))
    assert (n_norm, n_def) == (4, 5)
    assert loop.adapter.fit_calls[-1]["m"] == 5          # 反馈后掩膜数同步增长
    assert len(loop.masks) == 5


def test_no_masks_passes_none():
    """不传掩膜时按契约显式传 defect_masks=None,而不是省略这个参数。"""
    loop = _loop()
    assert loop.adapter.fit_calls[-1]["m"] is None


def test_feedback_registers_hard_constraint_samples():
    """反馈样本必须登记进适配器的硬约束集合。只进样本集是不够的——实测(cable)
    1张新样本在130张里投不出票,DINO门阈值小数点后五位都没动,那张图依然漏检。"""
    loop = _loop()
    loop.feedback(torch.ones(3, 8, 8), is_defect=True)
    loop.feedback(torch.zeros(3, 8, 8), is_defect=False)
    assert len(loop.adapter._fb_defects) == 1
    assert len(loop.adapter._fb_normals) == 1


def test_feedback_uses_fast_path():
    """赛题要求误检/漏检都能**实时**反馈,所以两条路都跳过EAD学生重训。
    这条实测依据见 active_learning.feedback() 的说明(1193s→251s,且误检反馈上
    快路径的安全边距 +0.326 反而优于完整路径 +0.269),锁死不许回退。"""
    loop = _loop()
    assert loop.adapter.fit_calls[0]["retrain_ead"] is True        # 初始fit是完整的
    loop.feedback(torch.ones(3, 8, 8), is_defect=True)
    assert loop.adapter.fit_calls[-1]["retrain_ead"] is False      # 漏检 → 快路径
    loop.feedback(torch.zeros(3, 8, 8), is_defect=False)
    assert loop.adapter.fit_calls[-1]["retrain_ead"] is False      # 误检 → 也是快路径

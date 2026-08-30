"""操作员反馈硬约束(competition._apply_feedback_constraint)。

不跑真检测器:该方法只依赖 ead.score / gate.score / self._modes / self._dino_thr,
用替身喂进去就能锁住全部判定逻辑,不需要GPU也不需要20分钟的fit。
"""
import numpy as np
import torch

from aoi.competition import CompetitionLargeDetector


class _Scorer:
    """按图像均值给分的替身打分器(正常≈0,缺陷≈1)。"""

    def __init__(self):
        self.last_cls = torch.zeros(4)

    def score(self, img):
        self.last_cls = torch.zeros(4)
        return float(img.mean())


def _det(fb_def=(), fb_norm=(), thr=5.0):
    """只装配 _apply_feedback_constraint 用到的那几个字段,不构造真模型。"""
    d = CompetitionLargeDetector.__new__(CompetitionLargeDetector)
    d._dino_thr = thr
    d._fb_defects = list(fb_def)
    d._fb_normals = list(fb_norm)
    # 单模态:z = max((e-0)/1, (d-0)/1) = 图像均值
    d._modes = [dict(c=None, emu=0.0, esd=1.0, dmu=0.0, dsd=1.0, n=100)]
    return d


#: fit正常图的z分,均匀铺开 0.0~9.9 → "只许10%过线"对应的阈值下限约 8.91。
#: 铺开而不是取两个尖峰,是为了让"救得回 / 救不回"的分界点没有并列值的歧义。
ZN = [i / 10 for i in range(100)]
FP_FLOOR = 8.91


def _img(v):
    return torch.full((3, 8, 8), float(v))


def test_no_feedback_leaves_threshold_untouched():
    """没有反馈时,阈值必须逐位不变——反馈机制不能悄悄改变默认行为。"""
    d = _det(thr=5.0)
    d._apply_feedback_constraint(_Scorer(), _Scorer(), ZN)
    assert d._dino_thr == 5.0
    assert d._fb_unsat is None


def test_missed_defect_pulls_threshold_down_enough_to_catch_it():
    """漏检反馈:阈值必须降到让那张图过线。这正是实测里没能做到的事。"""
    d = _det(fb_def=[_img(9.2)], thr=9.5)      # 9.2 在误报上限之上,救得回
    d._apply_feedback_constraint(_Scorer(), _Scorer(), ZN)
    assert d._dino_thr < 9.2
    assert d._fb_unsat is None


def test_false_positive_feedback_pushes_threshold_up():
    """误检反馈:阈值必须升到让那张图不再过线。"""
    d = _det(fb_norm=[_img(9.8)], thr=5.0)
    d._apply_feedback_constraint(_Scorer(), _Scorer(), ZN)
    assert d._dino_thr > 9.8


def test_unrescuable_feedback_leaves_threshold_untouched():
    """**救不回就一动不动**:实测(cable留出验证)"压到误报上限尽力而为"付出留出
    误报+17.2pp、换回召回+0——所以救不回时阈值必须保持原值,只如实记录不可满足。"""
    d = _det(fb_def=[_img(5.0)], thr=9.5)      # 5.0 远低于误报上限对应的阈值(8.91)
    d._apply_feedback_constraint(_Scorer(), _Scorer(), ZN)
    assert d._dino_thr == 9.5                  # 原值,一分未动
    assert d._fb_unsat and "误报" in d._fb_unsat[0]


def test_contradictory_feedback_is_reported_not_silently_resolved():
    """操作员把一张高分图标成正常、又把一张低分图标成缺陷 —— 两条约束不可能同时
    满足。必须如实记录冲突,不能挑一条偷偷执行。"""
    d = _det(fb_def=[_img(6.0)], fb_norm=[_img(7.0)], thr=9.5)
    d._apply_feedback_constraint(_Scorer(), _Scorer(), ZN)
    assert d._fb_unsat and "不可分" in d._fb_unsat[-1]


def test_normal_feedback_never_lowers_threshold():
    """误检反馈只该让系统更保守,绝不能顺带把阈值降下去。"""
    d = _det(fb_norm=[_img(2.0)], thr=5.0)   # 2.0 已在阈值下方,无需上移
    d._apply_feedback_constraint(_Scorer(), _Scorer(), ZN)
    assert d._dino_thr == 5.0

import torch
from aoi.multibranch import MultiBranchAdapter
from aoi.types import BranchResult


class _FakeBranch:
    def __init__(self, defect_type, score_fn):
        self.defect_type = defect_type
        self.score_fn = score_fn
    def fit(self, images):
        return None
    def infer(self, image):
        return BranchResult(score=float(self.score_fn(image)), defect_type=self.defect_type)


def test_fit_predict_fuses_and_separates():
    # b1 按图像均值打分(正常0/缺陷1);b2 恒为0(静默分支)
    b1 = _FakeBranch("appearance", lambda im: im.mean())
    b2 = _FakeBranch("structural", lambda im: 0.0)
    a = MultiBranchAdapter([b1, b2])
    normals = [torch.zeros(3, 8, 8) for _ in range(4)]
    defects = [torch.ones(3, 8, 8) for _ in range(4)]
    a.fit_fewshot(normals, defects)
    r_n, is_n = a.predict(torch.zeros(1, 3, 8, 8))
    r_d, is_d = a.predict(torch.ones(1, 3, 8, 8))
    assert is_n is False
    assert is_d is True
    assert r_d.defect_type == "appearance"   # 最异常的分支决定缺陷类型
    assert r_n.defect_type == "normal"

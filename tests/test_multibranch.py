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


class _FakeSupervisedBranch:
    """有 fit_supervised(无 fit),验证 MultiBranchAdapter 的监督分支分发路径。"""
    defect_type = "appearance"
    def __init__(self):
        self.saw_supervised = False
    def fit_supervised(self, normals, defects):
        self.saw_supervised = True
    def infer(self, image):
        return BranchResult(score=float(image.mean()), defect_type=self.defect_type)


def test_supervised_branch_dispatch():
    b = _FakeSupervisedBranch()
    a = MultiBranchAdapter([b])
    normals = [torch.zeros(3, 8, 8) for _ in range(10)]
    defects = [torch.ones(3, 8, 8) for _ in range(10)]
    a.fit_fewshot(normals, defects)
    assert b.saw_supervised is True          # 走了 fit_supervised 而非 fit
    _, is_def = a.predict(torch.ones(1, 3, 8, 8))
    assert is_def is True


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


def test_predict_returns_fused_score_not_raw():
    # 回归测试(C1):predict 返回的 score 必须是融合 z 分,而非分支原始分,
    # 否则 run_protocol 的 AUROC 会基于错误的量纲。
    # b1 正常分=10/缺陷分=20(原始量纲大);正常 mean=10,std=0→1.0,缺陷 z=(20-10)/1=10。
    b1 = _FakeBranch("appearance", lambda im: 10.0 + 10.0 * float(im.mean()))
    b2 = _FakeBranch("structural", lambda im: 0.0)
    a = MultiBranchAdapter([b1, b2])
    normals = [torch.zeros(3, 8, 8) for _ in range(4)]
    defects = [torch.ones(3, 8, 8) for _ in range(4)]
    a.fit_fewshot(normals, defects)
    r_d, _ = a.predict(torch.ones(1, 3, 8, 8))
    assert r_d.score == 10.0          # 融合 z 分,不是原始分 20.0

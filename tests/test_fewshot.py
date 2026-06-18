from aoi.fewshot import FewShotAdapter

def test_calibrate_separates_scores():
    t = FewShotAdapter._calibrate(normal_scores=[0.0, 1.0], defect_scores=[5.0, 6.0])
    assert 1.0 < t <= 5.0

class _FakeBranch:
    defect_type = "appearance"
    def __init__(self): self.fitted = False
    def fit(self, imgs): self.fitted = True
    def infer(self, img):
        from aoi.types import BranchResult
        # 用图像均值当分数:>0.5 视为缺陷
        return BranchResult(score=float(img.mean()))

def test_fit_fewshot_sets_threshold_and_predicts():
    import torch
    b = _FakeBranch()
    a = FewShotAdapter(b)
    normals = [torch.zeros(3, 8, 8) for _ in range(4)]      # 均值 0
    defects = [torch.ones(3, 8, 8) for _ in range(4)]       # 均值 1
    a.fit_fewshot(normals, defects)
    assert b.fitted is True
    _, is_def_normal = a.predict(torch.zeros(1, 3, 8, 8))
    _, is_def_defect = a.predict(torch.ones(1, 3, 8, 8))
    assert is_def_normal is False
    assert is_def_defect is True

import torch
from aoi.branches.zeroshot_clip import ZeroShotCLIPBranch
from aoi.types import BranchResult


class FakeEncoder:
    """2 维嵌入:正常轴[1,0],异常轴[0,1]。文本含 defect/damaged/anomaly 归到异常轴;
    图像按均值映射:亮(均值大)→异常轴。"""
    def encode_text(self, prompts):
        embs = []
        for p in prompts:
            if any(k in p for k in ("defect", "damaged", "anomaly")):
                embs.append([0.0, 1.0])
            else:
                embs.append([1.0, 0.0])
        return torch.tensor(embs)

    def encode_image(self, image):
        m = float(image.mean())
        v = torch.tensor([[1.0 - m, m]])
        return v / v.norm()


def test_infer_returns_result_in_unit_range():
    b = ZeroShotCLIPBranch(FakeEncoder(), class_name="bottle")
    r = b.infer(torch.zeros(1, 3, 8, 8))
    assert isinstance(r, BranchResult)
    assert 0.0 <= r.score <= 1.0
    assert r.latency_ms >= 0.0

def test_abnormal_image_scores_higher():
    b = ZeroShotCLIPBranch(FakeEncoder(), class_name="bottle")
    s_normal = b.infer(torch.zeros(1, 3, 8, 8)).score   # 均值0 → 正常
    s_defect = b.infer(torch.ones(1, 3, 8, 8)).score    # 均值1 → 异常
    assert s_defect > s_normal

def test_fit_is_noop():
    b = ZeroShotCLIPBranch(FakeEncoder())
    assert b.fit(torch.zeros(2, 3, 8, 8)) is None

def test_infer_rejects_batch():
    import pytest
    b = ZeroShotCLIPBranch(FakeEncoder())
    with pytest.raises(AssertionError):
        b.infer(torch.zeros(2, 3, 8, 8))

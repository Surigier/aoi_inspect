import pytest
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.types import BranchResult

def _branch():
    return TextureADBranch(backbone=Backbone(pretrained=False), coreset_ratio=1.0)

def test_infer_returns_result_with_map():
    b = _branch()
    normal = torch.full((4, 3, 64, 64), 0.5)
    b.fit(normal)
    r = b.infer(torch.full((1, 3, 64, 64), 0.5))
    assert isinstance(r, BranchResult)
    assert r.anomaly_map is not None and r.anomaly_map.ndim == 2
    assert r.latency_ms >= 0.0

def test_anomaly_scores_higher_than_normal():
    b = _branch()
    normal = torch.full((4, 3, 64, 64), 0.5)
    b.fit(normal)
    s_normal = b.infer(torch.full((1, 3, 64, 64), 0.5)).score
    s_noise = b.infer(torch.rand(1, 3, 64, 64)).score
    assert s_noise > s_normal

def test_infer_rejects_batch():
    b = _branch()
    b.fit(torch.full((2, 3, 64, 64), 0.5))
    with pytest.raises(AssertionError):
        b.infer(torch.full((2, 3, 64, 64), 0.5))

def test_fit_is_idempotent():
    # 重复 fit 同样数据不应让记忆库累积(主动学习多轮反馈依赖此幂等性)
    b = _branch()
    imgs = torch.full((4, 3, 64, 64), 0.5)
    b.fit(imgs)
    n1 = b.bank.bank.shape[0]
    b.fit(imgs)
    n2 = b.bank.bank.shape[0]
    assert n1 == n2

import pytest
import torch
from aoi.backbone import Backbone
from aoi.branches.structural_ad import StructuralADBranch


def _branch():
    return StructuralADBranch(backbone=Backbone(pretrained=False), grid_size=8)

def _with_square():
    """左上角亮块 = 部件存在。"""
    img = torch.full((1, 3, 64, 64), 0.5)
    img[:, :, :16, :16] = 1.0
    return img

def _missing():
    """无亮块 = 部件缺失。"""
    return torch.full((1, 3, 64, 64), 0.5)

def test_flags_missing_component():
    b = _branch()
    b.fit(torch.cat([_with_square() for _ in range(4)], dim=0))   # 正常都带左上块
    s_present = b.infer(_with_square()).score
    s_missing = b.infer(_missing()).score
    assert s_missing > s_present                                   # 缺件 → 高分

def test_infer_map_shape_and_batch_guard():
    b = _branch()
    b.fit(_with_square())
    r = b.infer(_with_square())
    assert r.anomaly_map.shape == (8, 8)
    assert r.defect_type == "structural"
    with pytest.raises(AssertionError):
        b.infer(torch.zeros(2, 3, 64, 64))

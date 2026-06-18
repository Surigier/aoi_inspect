import pytest
import torch
from aoi.branches.dimension_ad import DimensionADBranch


def _square(size):
    img = torch.full((1, 3, 64, 64), 0.5)
    img[:, :, :size, :size] = 1.0
    return img


def test_flags_size_deviation():
    b = DimensionADBranch()
    b.fit(torch.cat([_square(16) for _ in range(4)], dim=0))
    s_normal = b.infer(_square(16)).score
    s_big = b.infer(_square(24)).score
    assert s_big > s_normal

def test_large_foreground_not_inverted():
    # 回归:前景>50% 时,边缘背景估计避免全局中位数翻转(否则大缺陷会测出小面积被漏检)
    b = DimensionADBranch()
    b.fit(torch.cat([_square(16) for _ in range(4)], dim=0))
    s_small = b.infer(_square(16)).score
    s_large = b.infer(_square(56)).score          # 56/64 ≈ 76% 前景
    assert s_large > s_small

def test_result_fields_and_batch_guard():
    b = DimensionADBranch()
    b.fit(torch.cat([_square(16) for _ in range(4)], dim=0))
    r = b.infer(_square(16))
    assert r.defect_type == "dimension"
    assert r.score >= 0.0
    with pytest.raises(AssertionError):
        b.infer(torch.zeros(2, 3, 64, 64))

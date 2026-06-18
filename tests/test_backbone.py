import torch
from aoi.backbone import Backbone

def test_extract_shape():
    bb = Backbone(pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    f = bb.extract(x)
    assert f.ndim == 4
    assert f.shape[0] == 2          # batch 维保持
    assert f.shape[1] > 0           # 通道维 = 多层拼接
    assert f.shape[2] == f.shape[3] # 方形特征图

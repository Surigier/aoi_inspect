import pytest
import torch
from aoi.backbone import Backbone
from aoi.branches.disc_ad import DiscriminativeBranch
from aoi.types import BranchResult


def _blank():
    return torch.full((3, 64, 64), 0.5)


def _blob():
    img = torch.full((3, 64, 64), 0.5)
    img[:, :24, :24] = 1.0
    return img


def test_supervised_fit_then_separate():
    b = DiscriminativeBranch(backbone=Backbone(pretrained=False), epochs=300, lr=0.1)
    b.fit_supervised([_blank() for _ in range(8)], [_blob() for _ in range(8)])
    r = b.infer(_blob().unsqueeze(0))
    assert isinstance(r, BranchResult)
    assert b.infer(_blob().unsqueeze(0)).score > b.infer(_blank().unsqueeze(0)).score


def test_infer_rejects_batch():
    b = DiscriminativeBranch(backbone=Backbone(pretrained=False), epochs=10)
    b.fit_supervised([_blank() for _ in range(4)], [_blob() for _ in range(4)])
    with pytest.raises(AssertionError):
        b.infer(torch.zeros(2, 3, 64, 64))

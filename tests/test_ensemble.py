from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
from aoi.multibranch import MultiBranchAdapter
from aoi.branches.disc_ad import DiscriminativeBranch
from aoi.branches.color_ad import ColorADBranch
from aoi.branches.dimension_ad import DimensionADBranch


def test_default_adapter_covers_five_defect_types():
    a = default_adapter(Backbone(pretrained=False))
    assert isinstance(a, MultiBranchAdapter)
    assert len(a.branches) == 5
    assert any(isinstance(b, DiscriminativeBranch) for b in a.branches)
    assert any(isinstance(b, ColorADBranch) for b in a.branches)     # 色彩变化
    assert any(isinstance(b, DimensionADBranch) for b in a.branches)  # 尺寸偏差
    # 判别头是监督分支,须暴露 fit_supervised 供分发
    assert any(hasattr(b, "fit_supervised") for b in a.branches)

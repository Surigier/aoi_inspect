from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
from aoi.multibranch import MultiBranchAdapter
from aoi.branches.disc_ad import DiscriminativeBranch
from aoi.branches.color_ad import ColorADBranch


def test_default_adapter_has_four_branches_incl_disc_and_color():
    a = default_adapter(Backbone(pretrained=False))
    assert isinstance(a, MultiBranchAdapter)
    assert len(a.branches) == 4
    assert any(isinstance(b, DiscriminativeBranch) for b in a.branches)
    assert any(isinstance(b, ColorADBranch) for b in a.branches)     # 色彩变化覆盖
    # 判别头是监督分支,须暴露 fit_supervised 供分发
    assert any(hasattr(b, "fit_supervised") for b in a.branches)

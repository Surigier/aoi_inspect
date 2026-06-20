from aoi.backbone import Backbone
from aoi.ensemble import default_adapter
from aoi.multibranch import MultiBranchAdapter
from aoi.branches.disc_ad import DiscriminativeBranch


def test_default_adapter_has_three_branches_incl_disc():
    a = default_adapter(Backbone(pretrained=False))
    assert isinstance(a, MultiBranchAdapter)
    assert len(a.branches) == 3
    assert any(isinstance(b, DiscriminativeBranch) for b in a.branches)
    # 判别头是监督分支,须暴露 fit_supervised 供分发
    assert any(hasattr(b, "fit_supervised") for b in a.branches)

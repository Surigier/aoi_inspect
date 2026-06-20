import torch
from aoi.disc_head import DiscriminativeHead


def test_head_separates_two_classes():
    torch.manual_seed(0)
    nf = torch.tensor([[1.0, 0.0]]).repeat(20, 1) + 0.01 * torch.randn(20, 2)
    df = torch.tensor([[0.0, 1.0]]).repeat(20, 1) + 0.01 * torch.randn(20, 2)
    h = DiscriminativeHead(epochs=300, lr=0.1).fit(nf, df)
    s_normal = float(h.score(torch.tensor([[1.0, 0.0]])))
    s_defect = float(h.score(torch.tensor([[0.0, 1.0]])))
    assert s_defect > 0.5 > s_normal


def test_refit_changes_parameters():
    # 反馈累积新缺陷 → 重训 → 权重(模型参数)改变,体现"动态调整模型参数"
    torch.manual_seed(0)
    nf = torch.zeros(10, 2)
    df = torch.tensor([[0.0, 1.0]]).repeat(5, 1)
    h = DiscriminativeHead(epochs=100, lr=0.1).fit(nf, df)
    w1 = h.lin.weight.detach().clone()
    df2 = torch.cat([df, torch.tensor([[1.0, 0.0]]).repeat(5, 1)])   # 加入新方向的缺陷
    h.fit(nf, df2)
    w2 = h.lin.weight.detach()
    assert not torch.allclose(w1, w2)

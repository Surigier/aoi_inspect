import torch
from aoi.memory_bank import MemoryBank

def test_add_and_query_nearest():
    mb = MemoryBank()
    mb.add(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
    d = mb.query(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert d.shape == (2,)
    assert d[0].item() < 1e-5                       # 与 [0,0] 重合
    assert abs(d[1].item() - (2.0 ** 0.5)) < 1e-4   # 到 [0,0] 的距离 sqrt(2)

def test_coreset_reduces_size():
    mb = MemoryBank()
    mb.add(torch.randn(100, 8))
    mb.coreset_subsample(0.25)
    assert mb.bank.shape[0] == 25
    assert mb.bank.shape[1] == 8

def test_query_preserves_input_device():
    # query/add 不再强制 CPU:CPU 输入下结果仍在 CPU(GPU 上则留在 GPU 加速 cdist)
    mb = MemoryBank()
    mb.add(torch.zeros(3, 2))
    d = mb.query(torch.zeros(1, 2))
    assert d.device.type == "cpu"
    assert mb.bank.device.type == "cpu"

# tests/conftest.py
import torch
import pytest


def _normal(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.full((3, 64, 64), 0.5) + 0.01 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1)
            for _ in range(n)]


def _defect(n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        img = torch.full((3, 64, 64), 0.5) + 0.01 * torch.randn(3, 64, 64, generator=g)
        x = int(torch.randint(0, 48, (1,), generator=g))
        y = int(torch.randint(0, 48, (1,), generator=g))
        img[:, y:y + 16, x:x + 16] = 1.0          # 亮斑当缺陷
        out.append(img.clamp(0, 1))
    return out


@pytest.fixture
def synth_dataset():
    return {"normal": _normal(20, seed=0), "defect": _defect(20, seed=1)}

import torch


class MemoryBank:
    """存储正常样本的 patch 特征,查询时返回到最近邻的 L2 距离。"""

    def __init__(self):
        self.bank = None  # (N, C) tensor

    def add(self, features: torch.Tensor) -> None:
        features = features.detach().float()            # 保留所在设备(GPU 上则留 GPU)
        self.bank = features if self.bank is None else torch.cat([self.bank, features], dim=0)

    def coreset_subsample(self, ratio: float) -> None:
        """v1 用随机下采样近似 coreset(后续 Plan 可替换为 k-center-greedy)。"""
        n = self.bank.shape[0]
        k = max(1, int(n * ratio))
        idx = torch.randperm(n)[:k]
        self.bank = self.bank[idx]

    def query(self, features: torch.Tensor) -> torch.Tensor:
        """返回每个 query 特征到 bank 的最小 L2 距离,形状 (M,)。在 bank 所在设备上计算。"""
        d = torch.cdist(features.detach().float().to(self.bank.device), self.bank)
        return d.min(dim=1).values

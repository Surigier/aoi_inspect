import time
import torch
import torch.nn.functional as F
from ..types import BranchResult
from ..backbone import Backbone


class StructuralADBranch:
    """位置感知记忆库:按空间格子分别建正常特征库,捕捉缺件/错位/逻辑异常
    (标准平移不变记忆库捕捉不到"某位置该有的部件没了")。"""
    defect_type = "structural"

    def __init__(self, backbone: Backbone = None, grid_size: int = 8):
        self.backbone = backbone or Backbone()
        self.grid_size = grid_size
        self.bank = None  # (num_cells, N, C)

    def _cell_features(self, image: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (num_cells, B, C);特征图自适应池化到 grid×grid 网格。"""
        fmap = self.backbone.extract(image)                      # (B,C,h,w)
        g = self.grid_size
        pooled = F.adaptive_avg_pool2d(fmap, output_size=g)      # (B,C,g,g)
        b, c, _, _ = pooled.shape
        return pooled.reshape(b, c, g * g).permute(2, 0, 1)      # (num_cells, B, C)

    def fit(self, images: torch.Tensor) -> None:
        # 每次 fit 重建库,保证幂等(多轮反馈重复 fit 不累积重复样本)
        self.bank = self._cell_features(images)                  # (num_cells, B, C)

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        q = self._cell_features(image)                           # (num_cells, 1, C)
        d = torch.cdist(q, self.bank)                            # (num_cells, 1, N)
        cell_dist = d.min(dim=2).values.squeeze(1)              # (num_cells,)
        amap = cell_dist.reshape(self.grid_size, self.grid_size).cpu().numpy()
        score = float(amap.max())
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=amap,
                            defect_type=self.defect_type, latency_ms=lat)

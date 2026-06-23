import time
import torch
import torch.nn.functional as F
from ..types import BranchResult


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB[0,1] (B,3,H,W) -> CIE Lab (B,3,H,W)。L=亮度,a/b=色度(与亮度解耦)。"""
    m = rgb > 0.04045
    lin = torch.where(m, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return torch.where(t > 0.008856, t.clamp(min=1e-6) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.stack([L, a, bb], dim=1)


class ColorADBranch:
    """色彩变化:按空间格子比较 Lab 色度(a*,b*)与正常色度库的最近邻距离。
    用色度(非亮度 L)→ 对纯光照变化鲁棒,只对真实变色/偏色/氧化报警。
    无需骨干网络,纯颜色统计,极轻。"""
    defect_type = "color_change"

    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self.bank = None  # (num_cells, N, 2)

    def _cell_chroma(self, image: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (num_cells, B, 2);每格取 (mean a*, mean b*)。"""
        ab = rgb_to_lab(image)[:, 1:3]                          # (B,2,H,W) 色度
        g = self.grid_size
        pooled = F.adaptive_avg_pool2d(ab, output_size=g)       # (B,2,g,g)
        b = pooled.shape[0]
        return pooled.reshape(b, 2, g * g).permute(2, 0, 1)     # (num_cells, B, 2)

    def fit(self, images: torch.Tensor) -> None:
        self.bank = self._cell_chroma(images)                  # 每次重建,幂等

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        q = self._cell_chroma(image)                           # (num_cells,1,2)
        d = torch.cdist(q, self.bank)                          # (num_cells,1,N)
        cell_dist = d.min(dim=2).values.squeeze(1)             # (num_cells,)
        amap = cell_dist.reshape(self.grid_size, self.grid_size).cpu().numpy()
        score = float(amap.max())
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=amap,
                            defect_type=self.defect_type, latency_ms=lat)

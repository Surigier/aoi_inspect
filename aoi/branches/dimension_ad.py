import time
import torch
from ..types import BranchResult


class DimensionADBranch:
    """尺寸偏差:量前景面积(偏离背景中位数的像素数),与正常面积分布比较。"""
    defect_type = "dimension"

    def __init__(self, dev_thresh: float = 0.2):
        self.dev_thresh = dev_thresh
        self.mean = None
        self.std = None

    def _area(self, image: torch.Tensor) -> float:
        """image: (1,3,H,W) -> 前景像素数。"""
        gray = image[0].mean(dim=0)                       # H,W
        # 背景用边缘像素中位数估计:对大前景(>50%)鲁棒,避免全局中位数被前景翻转
        border = torch.cat([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        bg = border.median()
        return float((gray - bg).abs().gt(self.dev_thresh).sum().item())

    def fit(self, images: torch.Tensor) -> None:
        areas = torch.tensor([self._area(images[i:i + 1]) for i in range(images.shape[0])])
        self.mean = float(areas.mean())
        self.std = float(areas.std()) if images.shape[0] > 1 else 0.0

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        area = self._area(image)
        std = self.std if self.std > 1e-9 else 1.0
        score = abs(area - self.mean) / std               # 仅幅度(偏大/偏小同等视为异常)
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, defect_type=self.defect_type, latency_ms=lat)

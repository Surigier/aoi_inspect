"""判别分支:骨干特征(图像级均值池化)→ 轻量判别头(逻辑回归)。
监督训练于现场正常+缺陷;反馈累积→重训→权重更新(动态调整模型参数)。
缺陷会真正进入模型(不像记忆库只建模正常),故漏检反馈可模型级提召回。"""
import time
import torch
import torch.nn.functional as F
from ..types import BranchResult
from ..backbone import Backbone
from ..disc_head import DiscriminativeHead


class DiscriminativeBranch:
    defect_type = "appearance"

    def __init__(self, backbone: Backbone = None, epochs: int = 200, lr: float = 0.05):
        self.backbone = backbone or Backbone()
        self.head = DiscriminativeHead(epochs=epochs, lr=lr, device=self.backbone.device)

    def _feat(self, image: torch.Tensor) -> torch.Tensor:
        """(1,3,H,W) -> (D,) 图像级 L2 归一化特征。"""
        fmap = self.backbone.extract(image)            # (1,C,h,w)
        f = fmap.mean(dim=(2, 3))[0]                    # (C,)
        return F.normalize(f, dim=0)

    def fit_supervised(self, normal_images, defect_images) -> None:
        nf = torch.stack([self._feat(x.unsqueeze(0)) for x in normal_images])
        df = torch.stack([self._feat(x.unsqueeze(0)) for x in defect_images])
        self.head.fit(nf, df)

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]}")
        t0 = time.perf_counter()
        score = float(self.head.score(self._feat(image).unsqueeze(0)).item())
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, defect_type=self.defect_type, latency_ms=lat)

import time
import torch
from ..types import BranchResult
from ..backbone import Backbone
from ..memory_bank import MemoryBank


def _to_patch_features(fmap: torch.Tensor):
    """(B,C,h,w) -> ((B*h*w, C), (h,w))"""
    b, c, h, w = fmap.shape
    feats = fmap.permute(0, 2, 3, 1).reshape(-1, c)
    return feats, (h, w)


class TextureADBranch:
    """PatchCore 风格:正常 patch 特征入记忆库,推理时取最近邻距离作异常分。"""
    defect_type = "appearance"

    def __init__(self, backbone: Backbone = None, coreset_ratio: float = 0.25):
        self.backbone = backbone or Backbone()
        self.bank = MemoryBank()
        self.coreset_ratio = coreset_ratio

    def fit(self, images: torch.Tensor) -> None:
        fmap = self.backbone.extract(images)
        feats, _ = _to_patch_features(fmap)
        self.bank.add(feats)
        if self.coreset_ratio < 1.0:
            self.bank.coreset_subsample(self.coreset_ratio)

    def infer(self, image: torch.Tensor) -> BranchResult:
        t0 = time.perf_counter()
        fmap = self.backbone.extract(image)
        feats, (h, w) = _to_patch_features(fmap)
        dist = self.bank.query(feats)                 # (h*w,)
        amap = dist.reshape(h, w).numpy()
        score = float(amap.max())
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=amap,
                            defect_type=self.defect_type, latency_ms=lat)

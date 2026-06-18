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
        self.bank = MemoryBank()        # 每次 fit 重建记忆库,保证幂等(多轮反馈重复 fit 不累积)
        fmap = self.backbone.extract(images)
        feats, _ = _to_patch_features(fmap)
        self.bank.add(feats)
        if self.coreset_ratio < 1.0:
            self.bank.coreset_subsample(self.coreset_ratio)

    def infer(self, image: torch.Tensor) -> BranchResult:
        assert image.shape[0] == 1, (
            f"infer 期望单张图 (1,3,H,W),收到 batch={image.shape[0]};"
            " 批量推理请逐张调用。")
        t0 = time.perf_counter()
        fmap = self.backbone.extract(image)
        feats, (h, w) = _to_patch_features(fmap)
        dist = self.bank.query(feats)                 # (h*w,)
        amap = dist.reshape(h, w).cpu().numpy()
        score = float(amap.max())
        lat = (time.perf_counter() - t0) * 1000.0
        return BranchResult(score=score, anomaly_map=amap,
                            defect_type=self.defect_type, latency_ms=lat)

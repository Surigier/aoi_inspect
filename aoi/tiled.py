"""大图分块少样本检测前端(TiledFewShotDetector)。
2500² → 重叠分块 → 小骨干批前向 → 每块最近邻距离 → top-k 块聚合 → 图级判决。
方法依据 Tiled Ensemble(arXiv 2403.04932),扩展到 few-shot + 块批处理 + top-k 聚合。
块内复用 MemoryBank;阈值复用 FewShotAdapter._calibrate。"""
import time
import torch
import torch.nn.functional as F
from .memory_bank import MemoryBank
from .branches.texture_ad import _to_patch_features
from .fewshot import FewShotAdapter


def make_tiles(img: torch.Tensor, tile: int, stride: int):
    """img (3,H,W) → (tiles (N,3,tile,tile), positions [(y,x),...]),末块贴边保证全覆盖。"""
    _, H, W = img.shape

    def starts(L):
        if L <= tile:
            return [0]
        s = list(range(0, L - tile + 1, stride))
        if s[-1] != L - tile:
            s.append(L - tile)
        return s

    ys, xs = starts(H), starts(W)
    tiles, pos = [], []
    for y in ys:
        for x in xs:
            tiles.append(img[:, y:y + tile, x:x + tile])
            pos.append((y, x))
    return torch.stack(tiles), pos


class TiledFewShotDetector:
    def __init__(self, backbone, tile: int = 512, stride: int = 448,
                 patch_top_k: int = 10, tile_top_k: int = 3,
                 coreset_ratio: float = 0.02, position_aware: bool = False,
                 batch: int = 16, feat_grid: int = 0):
        self.bb = backbone
        self.tile = tile
        self.stride = stride
        self.patch_top_k = patch_top_k          # 块内 patch 距离 top-k 均值(抗噪)
        self.tile_top_k = tile_top_k            # 跨块 top-k 均值(稀疏小缺陷敏感)
        self.coreset_ratio = coreset_ratio
        self.position_aware = position_aware
        self.batch = batch
        self.feat_grid = feat_grid              # >0:特征图池化到 grid×grid(减 patch 数提速)
        self.bank = None                        # 共享库
        self.banks = {}                         # 位置感知:pos -> MemoryBank
        self.threshold = None

    def _tile_feats(self, tiles: torch.Tensor):
        """批前向提特征 → 每块 (h*w, C) patch 特征列表;feat_grid>0 时先池化降 patch 数。"""
        out = []
        for i in range(0, len(tiles), self.batch):
            fmap = self.bb.extract(tiles[i:i + self.batch])     # (b,C,h,w)
            if self.feat_grid > 0 and fmap.shape[-1] > self.feat_grid:
                fmap = F.adaptive_avg_pool2d(fmap, self.feat_grid)
            for f in fmap:
                feats, _ = _to_patch_features(f.unsqueeze(0))
                out.append(feats)
        return out

    def fit_fewshot(self, normals, defects):
        per_pos = {}
        for img in normals:
            tiles, pos = make_tiles(img, self.tile, self.stride)
            for p, f in zip(pos, self._tile_feats(tiles)):
                per_pos.setdefault(p, []).append(f)
        if self.position_aware:
            self.banks = {}
            for p, fs in per_pos.items():
                b = MemoryBank()
                b.add(torch.cat(fs))
                b.coreset_subsample(self.coreset_ratio)
                self.banks[p] = b
        else:
            self.bank = MemoryBank()
            self.bank.add(torch.cat([f for fs in per_pos.values() for f in fs]))
            self.bank.coreset_subsample(self.coreset_ratio)
        ns = [self._image_score(i)[0] for i in normals]
        ds = [self._image_score(i)[0] for i in defects]
        self.threshold = FewShotAdapter._calibrate(ns, ds)
        return self.threshold

    def _image_score(self, img: torch.Tensor):
        """返回 (图级分数, 最异常块位置)。"""
        if img.dim() == 4:
            img = img[0]
        tiles, pos = make_tiles(img, self.tile, self.stride)
        scores = []
        for p, f in zip(pos, self._tile_feats(tiles)):
            bank = self.banks.get(p) if self.position_aware else self.bank
            if bank is None:                    # 位置在 fit 时未见 → 跳过
                continue
            d = bank.query(f)
            k = max(1, min(self.patch_top_k, d.numel()))
            scores.append((float(d.topk(k).values.mean()), p))
        scores.sort(reverse=True)
        kk = max(1, min(self.tile_top_k, len(scores)))
        img_score = sum(s for s, _ in scores[:kk]) / kk
        worst = scores[0][1] if scores else None
        return img_score, worst

    def predict(self, img: torch.Tensor):
        t0 = time.perf_counter()
        s, worst = self._image_score(img)
        lat = (time.perf_counter() - t0) * 1000.0
        is_def = bool(self.threshold is not None and s >= self.threshold)
        return {"score": s, "is_defect": is_def, "worst_tile": worst, "latency_ms": lat}

"""大图(2500²)EfficientAD:分块 → 训一个学生于所有正常块 → 测时逐块(批)打分 → top-k 聚合。
分块既保小缺陷(块内原生分辨率),又给 EfficientAD 更多训练样本(块)缓解少样本欠训;
推理无记忆库 → 延时恒定。接口与其他 adapter 一致(fit_fewshot/predict)。"""
import time
from .efficientad import EfficientADDetector
from .tiled import make_tiles
from .fewshot import FewShotAdapter


class TiledEfficientAD:
    def __init__(self, model_size="small", device="cuda", train_steps=8000,
                 tile=512, stride=512, tile_top_k=3, batch=16,
                 whole_infer=False, max_size=1536):
        self.det = EfficientADDetector(model_size=model_size, device=device,
                                       train_steps=train_steps)
        self.tile = tile
        self.stride = stride
        self.tile_top_k = tile_top_k
        self.batch = batch
        self.whole_infer = whole_infer       # True:推理走整图全卷积(快,保细节);训练仍用块
        self.max_size = max_size
        self.threshold = None

    def _tiles(self, img):
        return list(make_tiles(img, self.tile, self.stride)[0])

    def fit_fewshot(self, normals, defects):
        norm_tiles = []
        for im in normals:
            norm_tiles.extend(self._tiles(im))
        self.det.fit_fewshot(norm_tiles, None)          # 训学生于所有正常块
        ns = [self._image_score(im) for im in normals]
        ds = [self._image_score(im) for im in defects]
        self.threshold = FewShotAdapter._calibrate(ns, ds)
        return self.threshold

    def _image_score(self, img):
        if self.whole_infer:
            return self.det.score_large(img, max_size=self.max_size)
        scores = self.det.score_images(self._tiles(img), batch=self.batch)
        scores.sort(reverse=True)
        k = max(1, min(self.tile_top_k, len(scores)))
        return sum(scores[:k]) / k

    def predict(self, img):
        t0 = time.perf_counter()
        s = self._image_score(img)
        lat = (time.perf_counter() - t0) * 1000.0
        return {"score": s, "is_defect": bool(self.threshold is not None and s >= self.threshold),
                "latency_ms": lat}

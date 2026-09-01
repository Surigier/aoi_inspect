"""大图(2500²)EfficientAD:分块 → 训一个学生于所有正常块 → 测时逐块(批)打分 → top-k 聚合。
分块既保小缺陷(块内原生分辨率),又给 EfficientAD 更多训练样本(块)缓解少样本欠训;
推理无记忆库 → 延时恒定。接口与其他 adapter 一致(fit_fewshot/predict)。"""
import time
from .efficientad import EfficientADDetector
from .tiled import make_tiles
from .fewshot import FewShotAdapter


class TiledEfficientAD:
    def __init__(self, model_size="small", device="cuda", train_steps=10000,
                 tile=256, stride=256, tile_top_k=3, batch=16,
                 whole_infer=True, max_size=1152, max_pixels=1_400_000,
                 compile_infer=False, n_students=1, ead_smooth_k=1):
        # max_size 卡长边(1152)为主;max_pixels 仅作超大图安全网(1152²≈1.33M)。
        # FP16 整图卷积:1152²方图~110ms@4060→~187ms@2060,达标且留余量;
        # 1152 vs 1280 仅0.9×线性,精度基本保住(对比650k降分辨率会塌成0.49)。
        # compile_infer:fit 后 torch.compile 加速推理(方形-24%、细长-56%)。
        # n_students:多种子学生集成(检测分),教师共享;fit不计时→训练免费。
        self.det = EfficientADDetector(model_size=model_size, device=device,
                                       train_steps=train_steps, compile_infer=compile_infer,
                                       n_students=n_students)
        self.tile = tile
        self.stride = stride
        self.tile_top_k = tile_top_k
        self.batch = batch
        self.whole_infer = whole_infer       # True:推理走整图全卷积(快,保细节);训练仍用块
        self.max_size = max_size
        self.max_pixels = max_pixels
        self.ead_smooth_k = ead_smooth_k    # 见 EfficientADDetector.score_large 的 smooth_k
        self.threshold = None

    def _tiles(self, img):
        return list(make_tiles(img, self.tile, self.stride)[0])

    def fit_fewshot(self, normals, defects, retrain_student=True):
        """retrain_student=False:跳过学生训练,只重标阈值。
        依据:学生只在正常块上训(下面这行传的是norm_tiles和None),缺陷图**只参与
        阈值标定**——所以操作员反馈的是缺陷图(漏检)时,重训学生纯属浪费:实测学生
        训练占整个fit_fewshot的绝大部分耗时(~17-20分钟),跳过后反馈只需秒级到分钟级,
        这是"实时反馈"能不能成立的关键。新增的是正常图(误检反馈)时必须重训(有新
        正常数据),调用方负责判断。"""
        if retrain_student:
            norm_tiles = []
            for im in normals:
                norm_tiles.extend(self._tiles(im))
            self.det.fit_fewshot(norm_tiles, None)      # 训学生于所有正常块
        ns = [self._image_score(im) for im in normals]
        ds = [self._image_score(im) for im in defects]
        self.threshold = FewShotAdapter._calibrate(ns, ds)
        return self.threshold

    def _image_score(self, img):
        if self.whole_infer:
            return self.det.score_large(img, max_size=self.max_size, max_pixels=self.max_pixels,
                                        smooth_k=self.ead_smooth_k)
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

"""赛场大图统一检测器(CompetitionLargeDetector):2500² 输入的生产路径。
核心 = EfficientAD 整图卷积(强检测 + 低延时);辅以色彩/尺寸/结构分支补齐 5 类缺陷覆盖。
可靠性加权 z-norm 软融合 → 二值判决 + 缺陷类型输出(最强分支=类型)。
延时 ≈ EfficientAD(~106ms@2060)+ 几个轻分支(几 ms),仍 <200ms。"""
import torch
import torch.nn.functional as F
from .fusion import znorm
from .fewshot import FewShotAdapter
from .backbone import Backbone
from .tiled_efficientad import TiledEfficientAD
from .branches.color_ad import ColorADBranch
from .branches.dimension_ad import DimensionADBranch
from .branches.structural_ad import StructuralADBranch
from .seg_head import SupervisedSegHead, map_to_boxes


def _down(img, size):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)


class _EADBranch:
    """EfficientAD 核心(整图卷积),映射到'外观缺陷'(一般异常的默认归类)。"""
    defect_type = "外观缺陷"

    def __init__(self, **kw):
        self.det = TiledEfficientAD(**kw)

    def fit(self, normals, defects):
        self.det.fit_fewshot(normals, defects)

    def score(self, img):
        return self.det._image_score(img)


class _AuxBranch:
    """轻量辅助分支(色彩/尺寸/结构),在下采样大图上跑,负责对应缺陷类型。"""
    def __init__(self, branch, defect_type, size):
        self.branch = branch
        self.defect_type = defect_type
        self.size = size

    def fit(self, normals, defects):
        self.branch.fit(torch.cat([_down(n, self.size) for n in normals], 0))

    def score(self, img):
        return self.branch.infer(_down(img, self.size)).score


class CompetitionLargeDetector:
    def __init__(self, device="cuda", aux_size=320, train_steps=10000, seg_eval_hw=(256, 256),
                 compile_infer=False, sam_refine=True, roi_zoom=False):
        # roi_zoom 默认关:AD2真大图实测负面(0.131→0.072,裁块尺度失配+阈值不匹配),留待修复
        dev = device if torch.cuda.is_available() else "cpu"
        bb = Backbone(device=dev)
        self.branches = [
            _EADBranch(device=dev, train_steps=train_steps, compile_infer=compile_infer),
            _AuxBranch(ColorADBranch(grid_size=16), "色彩变化", aux_size),
            _AuxBranch(DimensionADBranch(), "尺寸偏差", aux_size),
            _AuxBranch(StructuralADBranch(backbone=bb, grid_size=16), "缺件/逻辑", aux_size),
        ]
        self.stats = []
        self.weights = []
        self.threshold = None
        # 定位专用浅层骨干:WRN50 layers(1,2) @512 → 128²特征格(现状40²比微小缺陷粗)。
        # 扫描实测(run_feat_res.py):IoU均值0.305→0.449(+47%),pcb+53%/battery+74%/pill+72%,
        # 且浅层更快(8ms vs 36ms)。640过犹不及。结构分支仍用默认bb(layers 2,3)不受影响。
        self._bb_loc = Backbone(layers=(1, 2), device=dev)
        self._seg_in = 512
        self.seg_head = SupervisedSegHead(device=dev, extractor=self._wrn_feats)
        self.seg_eval_hw = seg_eval_hw
        self.pix_thr = None                                # 像素图二值阈值(正常分位标定)
        from .sam_refine import SamRefiner
        self.sam = SamRefiner() if sam_refine else None    # SAM边界精化(仅判缺陷图触发)
        self.roi_zoom = roi_zoom                           # 2500²大图:粗定位→原生裁块→局部重分割

    @torch.no_grad()
    def _wrn_feats(self, img):
        """img(3,H,W)[0,1] → WRN50 浅层(1,2)特征 (C,128,128)。
        先搬 GPU 再下采样(大图 CPU interpolate 慢),再提特征(~8ms)。"""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self._bb_loc.device)
        img = F.interpolate(img, size=(self._seg_in, self._seg_in), mode="bilinear", align_corners=False)
        return self._bb_loc.extract(img)[0]

    def fit_fewshot(self, normals, defects, defect_masks=None):
        """检测由 EAD 核心(branches[0])单独负责;辅助分支只为'类型归属'拟合并估 μ/σ。
        实测从少样本估融合权重不可靠(弱分支overfit→拖垮强EAD),故不做检测层融合。
        defect_masks:每张缺陷的 (H,W){0,1} 掩膜(赛题迁移图带标注)→ 训监督分割头提定位精度。"""
        self.stats = []
        for b in self.branches:
            b.fit(normals, defects)
            ns = [b.score(x) for x in normals]
            m = sum(ns) / len(ns)
            s = (sum((x - m) ** 2 for x in ns) / len(ns)) ** 0.5
            self.stats.append((m, s))
        ead = self.branches[0]
        ns = [ead.score(x) for x in normals]
        ds = [ead.score(d) for d in defects]
        self.threshold = FewShotAdapter._calibrate(ns, ds)    # 阈值标在 EAD 分上
        if defect_masks is not None:
            d_imgs, d_masks = list(defects), list(defect_masks)
            if self.roi_zoom:
                ci, cm = self._native_crops(defects, defect_masks)   # 原生裁块样本(教头认放大视角)
                d_imgs += ci; d_masks += cm
            self.seg_head.fit(self._eff(), d_imgs, d_masks, normals[:30])
        self._calibrate_pixel(normals)
        return self.threshold

    def _native_crops(self, imgs, masks, pad=1.0, min_c=256, max_c=1024):
        """对每张缺陷图:按掩膜连通域在原生分辨率裁块(带上下文),供分割头训练。
        小图(长边<700)原生≈全局视角,跳过。返回 (crop_imgs, crop_masks)。"""
        import numpy as np
        try:
            import cv2
        except Exception:
            return [], []
        ci, cm = [], []
        for img, mk in zip(imgs, masks):
            H, W = img.shape[-2:]
            if max(H, W) < 700:
                continue
            mh, mw = mk.shape
            n, _, stats, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
            for i in range(1, min(n, 4)):
                x, y, w, h, a = stats[i]
                if a < 2:
                    continue
                # 掩膜坐标 → 原图坐标
                cx, cy = (x + w / 2) * W / mw, (y + h / 2) * H / mh
                side = int(min(max(max(w * W / mw, h * H / mh) * (1 + 2 * pad), min_c), max_c))
                x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
                y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
                x1, y1 = min(W, x0 + side), min(H, y0 + side)
                crop = img[:, y0:y1, x0:x1]
                # 掩膜对应子区域(掩膜空间)
                mx0, my0 = int(x0 * mw / W), int(y0 * mh / H)
                mx1, my1 = max(mx0 + 1, int(x1 * mw / W)), max(my0 + 1, int(y1 * mh / H))
                sub = mk[my0:my1, mx0:mx1]
                if sub.sum() == 0:
                    continue
                ci.append(crop); cm.append(sub.astype(np.uint8))
        return ci, cm

    def _eff(self):
        """底层 EfficientADDetector(residual_map_large/anomaly_map_large 在它上面)。"""
        return self.branches[0].det.det

    def _calibrate_pixel(self, normals):
        """像素二值阈值:优先用监督头在fit缺陷掩膜上标的F1最优阈值(实测IoU +58%);
        无监督头(无掩膜)则回退正常p99.5分位。"""
        import numpy as np
        if self.seg_head.thr is not None:
            self.pix_thr = self.seg_head.thr
            return
        vals = [self.segment(n).ravel() for n in normals[:20]]
        self.pix_thr = float(np.quantile(np.concatenate(vals), 0.995))

    def segment(self, img):
        """像素级异常图(原始尺度,不逐图标准化→阈值语义清晰)。
        有监督头(迁移带掩膜)→用它的 logit(BCE训,>0≈缺陷,实测均值0.890,救弱项);
        无掩膜→回退无监督 EAD 异常图。"""
        eff = self._eff()
        sup = self.seg_head.map(eff, img, self.seg_eval_hw)
        return sup if sup is not None else eff.anomaly_map_large(img, out_hw=self.seg_eval_hw)

    def locate(self, img):
        """完整定位输出:图级分(EAD)+ 判决 + 类型 + 像素图 + 检测框。"""
        import numpy as np
        res = self.predict(img)
        amap = self.segment(img)
        thr = self.pix_thr if self.pix_thr is not None else float(amap.mean() + 3 * amap.std())
        res["anomaly_map"] = amap
        if res["is_defect"]:
            mask = (amap >= thr).astype(np.uint8)
            if self.roi_zoom:
                mask = self._zoom_refine(img if img.dim() == 3 else img[0], mask, thr)  # 原生裁块重分割
            if self.sam is not None:
                mask = self.sam.refine(img if img.dim() == 3 else img[0], mask)  # SAM粗到细,IoU均值+23%
            res["mask"] = mask
            # 掩膜已阈值化+SAM精化,面积门槛放宽到~13px(默认52px会滤掉pcb类5×5微小缺陷)
            res["boxes"] = map_to_boxes(mask.astype(np.float32), 0.5, min_area_frac=0.0002, close=0)
        else:
            res["mask"] = None
            res["boxes"] = []
        return res

    def _zoom_refine(self, img, mask, thr, pad=1.0, min_c=256, max_c=1024, max_regions=6):
        """2500²大图定位精化:粗掩膜连通域→原生分辨率裁块→分割头重打分→贴回。
        消除'全图下采样512把微小缺陷抹掉'的结构损失;小图(长边<700)直接返回。"""
        import numpy as np
        try:
            import cv2
        except Exception:
            return mask
        H, W = img.shape[-2:]
        if max(H, W) < 700 or self.seg_head.head is None:
            return mask
        mh, mw = mask.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask
        out = mask.copy()
        for i in range(1, min(n, max_regions + 1)):
            x, y, w, h, a = stats[i]
            if a < 2:
                continue
            cx, cy = (x + w / 2) * W / mw, (y + h / 2) * H / mh
            side = int(min(max(max(w * W / mw, h * H / mh) * (1 + 2 * pad), min_c), max_c))
            x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
            y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
            x1, y1 = min(W, x0 + side), min(H, y0 + side)
            sub_amap = self.seg_head.map(self._eff(), img[:, y0:y1, x0:x1], (128, 128))
            sub_mask = (sub_amap >= thr).astype(np.uint8)
            # 贴回掩膜空间对应区域
            mx0, my0 = int(x0 * mw / W), int(y0 * mh / H)
            mx1, my1 = max(mx0 + 1, int(x1 * mw / W)), max(my0 + 1, int(y1 * mh / H))
            sub_rs = cv2.resize(sub_mask, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_NEAREST)
            out[my0:my1, mx0:mx1] = sub_rs
        return out

    def _ztype(self, raws):
        """各分支 z 分,最高者定缺陷类型(z 归一后可比)。"""
        zs = [znorm(r, m, s) for r, (m, s) in zip(raws, self.stats)]
        return self.branches[max(range(len(zs)), key=lambda i: zs[i])].defect_type

    def predict(self, img):
        raws = [b.score(img) for b in self.branches]
        score = raws[0]                                   # 检测分 = EAD 核心
        is_def = bool(self.threshold is not None and score >= self.threshold)
        return {"score": score, "is_defect": is_def,
                "defect_type": self._ztype(raws) if is_def else "normal"}

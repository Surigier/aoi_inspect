"""SAM 边界精化(粗到细定位,Segment Any Anomaly 同族思路):
分割头粗掩膜 → 连通域框提示 → MobileSAM 出锐利边界。

2026-07:从"总是接受SAM"改成"受控精化"——每个连通域独立OOF验证过的接受规则,不是整图
一起accept/reject。诊断(深裁A/B)发现SAM在反光/透明表面等场景可能整体伤害定位,门粗放的
"整图开/关"丢了信息(有的区域SAM真的帮忙、有的真的帮倒忙)。新逻辑:
  - box padding 候选(5%/10%/15%/25%)+ 4个逐区域特征(面积比、raw/SAM IoU、SAM是否覆盖
    raw异常图的峰值点、SAM边界处的图像梯度支持度)做阈值规则(不训练分类器——4维小规则,
    OOF在30张fit掩膜上搜索,比训练一个新模型更抗小样本过拟合,呼应RAMS教训)。
  - 只有OOF验证"用SAM比用raw平均IoU更高"才在生产启用该规则;否则退回原始"整图4倍面积"
    heuristic(不引入回归)。
  - 每个连通域独立套用规则,不是整图统一accept/reject。
延时不变(~19-38ms,imgsz=512),仅判缺陷图触发。SAM 不可用时优雅回退原掩膜。
"""
from pathlib import Path
import itertools
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "mobile_sam.pt"
PAD_CANDS = (0.05, 0.10, 0.15, 0.25)


class SamRefiner:
    def __init__(self, imgsz=512):
        self.imgsz = imgsz
        self._m = None
        self._failed = False
        self.padding = 0.15                  # 默认padding(未做OOF标定时的旧值)
        # gate 三态:"uncalibrated"(未调用calibrate,如无掩膜)→退回旧heuristic(总是接受除非
        # 4倍面积爆炸,历史默认行为保留);"reject_all"(OOF标定发现连最优接受规则都打不过完全
        # 不用SAM)→每个区域都拒绝SAM(等价于raw,不能退回"总是接受"——那个已被证明更差);
        # (t_area,t_iou,need_peak,t_grad)元组→OOF验证过净正的逐区域接受规则。
        self.gate = "uncalibrated"

    def _model(self):
        if self._m is None and not self._failed:
            try:
                from ultralytics import SAM
                self._m = SAM(str(_WEIGHTS) if _WEIGHTS.exists() else "mobile_sam.pt")
            except Exception:
                self._failed = True
        return self._m

    def _region_boxes(self, raw_mask, padding):
        H, W = raw_mask.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8), connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 3:
                continue
            px, py = max(2, int(w * padding)), max(2, int(h * padding))
            boxes.append((max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py), i))
        return boxes, stats, n

    def refine(self, img_chw, raw_mask, amap=None):
        """img(3,H,W)[0,1] tensor(CPU) + 二值掩膜(h,w) [+ 原始异常图amap,可选,供峰值特征用]
        → 精化掩膜(h,w)。失败/无模型回退原掩膜。逐连通域独立判断是否接受SAM结果。"""
        if self.gate == "reject_all":
            return raw_mask                                # OOF判SAM无净值:短路,连SAM推理都不跑(缺陷图省~40-60ms)
        m = self._model()
        if m is None or not _HAS_CV2:
            return raw_mask
        H, W = raw_mask.shape
        padding = self.padding
        boxes, stats, n = self._region_boxes(raw_mask, padding)
        if not boxes:
            return raw_mask
        arr = (img_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ih, iw = arr.shape[:2]
        sx, sy = iw / W, ih / H
        native_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
        try:
            r = m.predict(arr, bboxes=native_boxes, imgsz=self.imgsz, verbose=False)[0]
        except Exception:
            return raw_mask
        if r.masks is None:
            return raw_mask
        n_masks = r.masks.data.shape[0]
        out = np.zeros((H, W), np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx ** 2 + gy ** 2)
        for k, b in enumerate(boxes):
            x0, y0, x1, y1, comp_id = b
            raw_region = raw_mask[y0:y1, x0:x1]
            if k >= n_masks:
                out[y0:y1, x0:x1] |= raw_region
                continue
            mk_native = r.masks.data[k].cpu().numpy().astype(np.uint8)
            mk = cv2.resize(mk_native, (W, H), interpolation=cv2.INTER_NEAREST)
            sam_region = mk[y0:y1, x0:x1]
            feats = self._region_features(raw_region, sam_region, amap, x0, y0, x1, y1, gmag)
            if self._accept(feats):
                out[y0:y1, x0:x1] |= sam_region
            else:
                out[y0:y1, x0:x1] |= raw_region
        return out

    def _region_features(self, raw_region, sam_region, amap, x0, y0, x1, y1, gmag):
        """逐区域4特征:面积比、raw/SAM的IoU、SAM是否覆盖raw异常图峰值点、SAM边界梯度支持度。"""
        box_area = max(1, (x1 - x0) * (y1 - y0))
        area_ratio = float(sam_region.sum()) / box_area
        inter = int((raw_region.astype(bool) & sam_region.astype(bool)).sum())
        union = int((raw_region.astype(bool) | sam_region.astype(bool)).sum())
        iou_raw_sam = inter / max(union, 1)
        peak_covered = True
        if amap is not None:
            sub = amap[y0:y1, x0:x1]
            if sub.size > 0:
                py, px = np.unravel_index(np.argmax(sub), sub.shape)
                peak_covered = bool(sam_region[py, px]) if sam_region.size > 0 else False
        # 边界梯度支持度:SAM掩膜边界像素处的图像梯度均值(边界与真实边缘对齐→高;
        # 边界落在平坦/噪声区→低,提示SAM可能把无关区域当物体边界)
        boundary_grad = 0.0
        if sam_region.any():
            edge = sam_region.astype(np.uint8) - cv2.erode(sam_region.astype(np.uint8), np.ones((3, 3), np.uint8))
            gsub = gmag[y0:y1, x0:x1]
            if edge.sum() > 0 and gsub.shape == edge.shape:
                boundary_grad = float(gsub[edge > 0].mean())
        return {"area_ratio": area_ratio, "iou_raw_sam": iou_raw_sam,
                "peak_covered": peak_covered, "boundary_grad": boundary_grad}

    def _accept(self, feats):
        """生产接受规则,按self.gate三态分派(见__init__注释):
        'uncalibrated'→旧heuristic(面积比>4→拒绝,历史默认行为,未标定时保底);
        'reject_all'→OOF发现SAM在此产品上净负(见run_sam_gate_ab.py实测:sheet_metal/
        walnuts/fruit_jelly上"总是接受"比完全不用SAM还差),每区域都拒绝,等价于raw;
        元组→OOF验证过净正的逐区域接受规则。"""
        if self.gate == "reject_all":
            return False
        if self.gate == "uncalibrated" or self.gate is None:
            return feats["area_ratio"] <= 4.0 and feats["area_ratio"] > 0
        t_area, t_iou, need_peak, t_grad = self.gate
        ok = feats["area_ratio"] <= t_area and feats["iou_raw_sam"] >= t_iou and feats["boundary_grad"] >= t_grad
        if need_peak:
            ok = ok and feats["peak_covered"]
        return ok

    def calibrate(self, det, defect_imgs, defect_masks, seg_map_fn, k=5, seed=0):
        """OOF标定:在30张fit缺陷图上搜索padding(5/10/15/25%)+ 接受规则阈值网格,
        只有OOF验证'用规则后平均IoU > 全raw基线'才启用该(padding,gate);否则保持None
        (=旧heuristic,零回退)。seg_map_fn(img)->原始(logit)异常图(未阈值化,供峰值特征/
        阈值化用),按det.seg_head.thr做二值化得到raw_mask。
        每个连通域是一条OOF样本(不是每张图一条),小样本(30图,通常几十到上百个连通域)下
        用简单阈值规则而非训练分类器,呼应RAMS教训(小样本别上复杂可学习模块)。"""
        m = self._model()
        if m is None or not _HAS_CV2 or len(defect_imgs) < 6:
            return
        n = len(defect_imgs)
        kk = min(k, max(2, n // 3))
        order = list(range(n))
        import random as _r
        _r.Random(seed).shuffle(order)
        folds = [order[i::kk] for i in range(kk)]

        # 先算全raw基线(padding无关,用当前raw_mask做参照)
        thr = getattr(det.seg_head, "thr", None) if hasattr(det, "seg_head") else None

        # seg_map_fn(WRN分割头前向)与padding无关,4个padding候选共享——只算一次/图,避免
        # calibrate()慢4倍(SAM predict本身依赖padding算出的box坐标,不可省,仍需per-padding重跑)。
        from PIL import Image
        cache = {}
        for i in range(n):
            img, mk_native = defect_imgs[i], defect_masks[i]
            amap = seg_map_fn(img)
            if amap is None or thr is None:
                continue
            raw_mask = (amap >= thr).astype(np.uint8)
            H, W = raw_mask.shape
            gt = (np.array(Image.fromarray(mk_native).resize((W, H), Image.NEAREST)) > 0).astype(np.uint8)
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) if img.dim() == 3 else \
                  (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gmag = np.sqrt(gx ** 2 + gy ** 2)
            cache[i] = (amap, raw_mask, gt, arr, gmag)

        best_overall = None
        for padding in PAD_CANDS:
            # 收集该padding下所有(fold, region)的特征+是否SAM优于raw的标签,按fold分组供OOF
            fold_regions = {fi: [] for fi in range(kk)}   # fi -> [(feats, sam_better:bool, raw_iou, sam_iou)]
            for fi in range(kk):
                idxs = folds[fi]
                for i in idxs:
                    if i not in cache:
                        continue
                    amap, raw_mask, gt, arr, gmag = cache[i]
                    H, W = raw_mask.shape
                    boxes, stats, _n = self._region_boxes(raw_mask, padding)
                    if not boxes:
                        continue
                    native_boxes = [[b[0], b[1], b[2], b[3]] for b in boxes]
                    try:
                        r = m.predict(arr, bboxes=native_boxes, imgsz=self.imgsz, verbose=False)[0]
                    except Exception:
                        continue
                    if r.masks is None:
                        continue
                    n_masks = r.masks.data.shape[0]
                    for kidx, b in enumerate(boxes):
                        if kidx >= n_masks:
                            continue
                        x0, y0, x1, y1, _ = b
                        raw_region = raw_mask[y0:y1, x0:x1]; gt_region = gt[y0:y1, x0:x1]
                        sam_region = cv2.resize(r.masks.data[kidx].cpu().numpy().astype(np.uint8), (W, H),
                                                interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]
                        feats = self._region_features(raw_region, sam_region, amap, x0, y0, x1, y1, gmag)

                        def _iou(a, b):
                            inter = int((a.astype(bool) & b.astype(bool)).sum())
                            union = int((a.astype(bool) | b.astype(bool)).sum())
                            return inter / max(union, 1)
                        raw_iou = _iou(raw_region, gt_region); sam_iou = _iou(sam_region, gt_region)
                        fold_regions[fi].append((feats, sam_iou > raw_iou, raw_iou, sam_iou))

            all_regions = [r for fr in fold_regions.values() for r in fr]
            if len(all_regions) < 6:
                continue
            # 阈值网格(小,4×4×2):对每个候选规则做OOF(每fold的规则由其余fold的regions搜索得到)
            area_cands = [1.0, 2.0, 4.0, 8.0]
            iou_cands = [0.0, 0.1, 0.2, 0.3]
            grad_cands = [0.0, 5.0, 10.0]
            best_rule, best_iou_gain = None, -1e9
            for t_area, t_iou, need_peak, t_grad in itertools.product(area_cands, iou_cands, (True, False), grad_cands):
                oof_raw_ious, oof_chosen_ious = [], []
                for fi in range(kk):
                    for feats, sam_better, raw_iou, sam_iou in fold_regions[fi]:
                        ok = feats["area_ratio"] <= t_area and feats["iou_raw_sam"] >= t_iou and \
                             feats["boundary_grad"] >= t_grad
                        if need_peak:
                            ok = ok and feats["peak_covered"]
                        oof_raw_ious.append(raw_iou)
                        oof_chosen_ious.append(sam_iou if ok else raw_iou)
                gain = float(np.mean(oof_chosen_ious)) - float(np.mean(oof_raw_ious))
                if gain > best_iou_gain:
                    best_iou_gain = gain
                    best_rule = (t_area, t_iou, need_peak, t_grad)
            # "全部拒绝"(=raw,gain恒为0)必须显式作为候选参与比较——网格里最严阈值组合可能因
            # 边界情况(如area_ratio恰好等于1.0)仍漏放几个区域进来,算出的gain可能略负于0,
            # 若不显式兜底,后面会误落到"退回旧总接受"分支(已证明比raw还差,见sheet_metal
            # 0.324<0.486的教训)。
            if best_iou_gain < 0.0:
                best_iou_gain = 0.0
                best_rule = "reject_all"
            if best_overall is None or best_iou_gain > best_overall[1]:
                best_overall = (padding, best_iou_gain, best_rule)

        if best_overall is None:
            self.gate = "uncalibrated"                          # 数据不足以标定→保留历史默认行为
            return
        padding, gain, rule = best_overall
        if gain <= 0.005:                                       # 没有显著优于"完全不用SAM"
            self.gate = "reject_all"                             # 保守兜底=raw,绝不比raw差
            self.calib_gain = gain
            return
        self.padding = padding                                  # gain>margin时rule必为学到的元组
        self.gate = rule
        self.calib_gain = gain

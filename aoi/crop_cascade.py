"""独立crop-head级联(微小缺陷ROI,重做roi_zoom治其两个缺陷):
①候选来自ECC对齐模板残差(Lab色差+梯度差,原生分辨率,training-free),不是已经在512
下采样时看不见微缺陷的全图粗分割头——高召回,不依赖粗头能不能先看到。
②crop-head独立训练(SupervisedSegHead新实例)+独立mu/sd/阈值,不复用全图头的统计量,
避免全图/裁块尺度分布漂移。
只在fit留出OOF验证confirmed净正增益的产品上启用(self.enabled),否则locate()原样返回
全图结果,零回退——同SAM门控/DINO门控一致的安全哲学。
"""
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from .seg_head import SupervisedSegHead, _mask_to, _per_image_iou


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def _lab_grad_candidates(img, ref, topk=6, min_c=192, max_c=768, pad=1.5):
    """ECC对齐模板残差 → 高召回候选框(原生分辨率,training-free,不依赖粗分割头)。
    Lab色差(对光照相对鲁棒)+梯度差(对结构/纹理边缘敏感)各自标准化后取max融合,
    连通域候选按残差强度排序取top-k。"""
    if not _HAS_CV2:
        return []
    H, W = img.shape[-2:]
    a = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    b = (ref.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_diff = np.linalg.norm(lab_a - lab_b, axis=-1)
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gxa = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3); gya = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
    gxb = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3); gyb = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
    grad_diff = np.abs(np.sqrt(gxa ** 2 + gya ** 2) - np.sqrt(gxb ** 2 + gyb ** 2))
    combined = np.clip(np.maximum(_z(lab_diff), _z(grad_diff)), 0, None)
    thr = combined.mean() + 2 * combined.std()
    binm = (combined >= thr).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4:
            continue
        score = float(combined[y:y + h, x:x + w].max())
        cx, cy = x + w / 2, y + h / 2
        side = int(min(max(max(w, h) * pad, min_c), max_c))
        x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
        y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
        x1, y1 = min(W, x0 + side), min(H, y0 + side)
        cands.append((x0, y0, x1, y1, score))
    cands.sort(key=lambda c: -c[4])
    return cands[:topk]


def _gt_crops(img, mk, min_c=192, max_c=768, pad=2.5, max_regions=4):
    """训练阶段用GT掩膜位置裁块(有标注,不依赖候选生成器召回率——生成器只用于无GT的推理)。"""
    if not _HAS_CV2:
        return []
    H, W = img.shape[-2:]
    mh, mw = mk.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, min(n, max_regions + 1)):
        x, y, w, h, a = stats[i]
        if a < 2:
            continue
        cx, cy = (x + w / 2) * W / mw, (y + h / 2) * H / mh
        side = int(min(max(max(w * W / mw, h * H / mh) * pad, min_c), max_c))
        x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
        y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
        x1, y1 = min(W, x0 + side), min(H, y0 + side)
        out.append((x0, y0, x1, y1))
    return out


class CropHeadCascade:
    """独立训练的crop-head(候选来自模板残差非全图粗分割),救全图下采样抹掉的微小缺陷。
    fit时按原图分折做OOF留出验证:只有确认净正增益才生产启用,否则locate()原样返回全图
    结果,零回退。预计缺陷图延时+15~35ms(仅判缺陷图触发,正常图不受影响)。"""

    def __init__(self, device="cuda", crop_seg_in=512):
        self.device = device
        self.crop_seg_in = crop_seg_in
        self.head = None
        self.enabled = False
        self.ref_bank = None
        self.gain = None

    def fit(self, det, ref_bank, defect_imgs, defect_masks, normal_imgs, min_native=700):
        """defect_imgs/masks原生尺寸。围绕GT位置裁块训crop-head(独立mu/sd/阈值),
        按原图(非按裁块)分折OOF验证候选生成器+crop-head合并后是否比纯全图头更好。
        小图(长边<700,原生≈全局视角)直接跳过,同_native_crops判据。"""
        import os as _os
        _dbg = _os.environ.get("CC_DEBUG")
        self.ref_bank = ref_bank
        native = [(img, mk) for img, mk in zip(defect_imgs, defect_masks)
                 if max(img.shape[-2:]) >= min_native]
        if _dbg: print(f"[cc] native={len(native)}/{len(defect_imgs)} (min_native={min_native})", flush=True)
        if len(native) < 8:
            self.enabled = False
            return

        crop_imgs, crop_masks, crop_src = [], [], []
        for idx, (img, mk) in enumerate(native):
            H, W = img.shape[-2:]; mh, mw = mk.shape
            gtc = _gt_crops(img, mk)
            if _dbg and idx < 3:
                print(f"[cc] img{idx} shape={tuple(img.shape)} mask_shape={mk.shape} mask_sum={int(mk.sum())} gt_crops={len(gtc)}", flush=True)
            for x0, y0, x1, y1 in gtc:
                crop = img[:, y0:y1, x0:x1]
                mx0, my0 = int(x0 * mw / W), int(y0 * mh / H)
                mx1, my1 = max(mx0 + 1, int(x1 * mw / W)), max(my0 + 1, int(y1 * mh / H))
                sub = mk[my0:my1, mx0:mx1]
                if sub.sum() == 0:
                    continue
                crop_imgs.append(crop); crop_masks.append(sub.astype(np.uint8)); crop_src.append(idx)
        if _dbg: print(f"[cc] crop_imgs={len(crop_imgs)}", flush=True)
        if len(crop_imgs) < 6:
            self.enabled = False
            return

        def extractor(img):
            x = (img.unsqueeze(0) if img.dim() == 3 else img).to(self.device)
            x = F.interpolate(x, size=(self.crop_seg_in, self.crop_seg_in),
                              mode="bilinear", align_corners=False)
            return det._bb_loc.extract(x)[0]

        head = SupervisedSegHead(device=self.device, steps=200, extractor=extractor)
        neg_crops = [im[:, :min(self.crop_seg_in, im.shape[-2]), :min(self.crop_seg_in, im.shape[-1])]
                    for im in normal_imgs[:10]]
        ok = head.fit(det, crop_imgs, crop_masks, neg_crops)
        if _dbg: print(f"[cc] head.fit ok={ok} thr={getattr(head,'thr',None)}", flush=True)
        if not ok:
            self.enabled = False
            return

        uniq_src = sorted(set(crop_src))
        if _dbg: print(f"[cc] uniq_src={len(uniq_src)}", flush=True)
        if len(uniq_src) < 4:
            self.enabled = False
            return
        hold_src = set(uniq_src[::3])
        gains = []
        for i in hold_src:
            img, mk = native[i]
            H, W = img.shape[-2:]
            full_amap = det.segment(img)
            fh, fw = full_amap.shape
            base_thr = det.pix_thr if det.pix_thr is not None else float(full_amap.mean() + 3 * full_amap.std())
            full_mask = (full_amap >= base_thr).astype(np.uint8)
            gt_full = _mask_to(mk, fh, fw)
            base_iou = _per_image_iou([full_mask], [gt_full])
            merged = self._merge_once(det, head, img, full_mask, (fh, fw))
            merged_iou = _per_image_iou([merged], [gt_full])
            gains.append(merged_iou - base_iou)
        gain = float(np.mean(gains)) if gains else -1.0
        if _dbg: print(f"[cc] hold_src={len(hold_src)} gains={gains} gain={gain}", flush=True)
        self.gain = gain
        self.head = head
        self.enabled = gain > 0.01                     # 需明显正增益(裁块推理有延时代价)才开

    def _merge_once(self, det, head, img, full_mask, out_hw):
        H, W = out_hw
        native_img = img if img.dim() == 3 else img[0]
        ref = self.ref_bank.aligned_ref(native_img)
        cands = _lab_grad_candidates(native_img, ref)
        out = full_mask.copy()
        iH, iW = native_img.shape[-2:]
        for x0, y0, x1, y1, _ in cands:
            crop = native_img[:, y0:y1, x0:x1]
            amap = head.map(det, crop, (max(1, y1 - y0), max(1, x1 - x0)))
            if amap is None or head.thr is None:
                continue
            crop_mask = (amap >= head.thr).astype(np.uint8)
            mx0, my0 = int(x0 * W / iW), int(y0 * H / iH)
            mx1, my1 = max(mx0 + 1, int(x1 * W / iW)), max(my0 + 1, int(y1 * H / iH))
            rs = cv2.resize(crop_mask, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_NEAREST)
            out[my0:my1, mx0:mx1] |= rs
        return out

    def refine(self, det, img, full_mask, out_hw):
        """生产入口:未启用(OOF未验证正增益)直接原样返回,零回退。"""
        if not self.enabled or self.head is None:
            return full_mask
        return self._merge_once(det, self.head, img, full_mask, out_hw)

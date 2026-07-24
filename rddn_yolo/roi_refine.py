"""Top-1参考ROI精修(WRN-LoRA封存后的新优先级,承接RDDN-YOLO冻结候选框的正向发现):
EAD+DINO判异常→现有WRN粗定位选Top-K可疑连通域→原始分辨率裁剪同一ROI(原图+ECC对齐
模板)→resize 640→差分通道(diff_channels.build_6ch)→冻结的预训练YOLO候选框(不用
LoRA微调——wrn_lora/WRN-LoRA、rddn_yolo/eval_fit.py YOLO-LoRA两次实验都已判负/收益
不广泛,已排除微调方向)→框内置信度过阈值才并入mask,低置信时该ROI原样保留WRN结果
(不是替换,是并集,避免crop_cascade早期"残差直接OR进结果"式的假阳性——这里的YOLO是
监督学出来的,不是原始残差阈值)。

接口与aoi/crop_cascade.py的CropHeadCascade完全对齐(.fit(det,ref_bank,...)/.refine(det,
img,full_mask,out_hw)),方便若验证净正后原样接入competition.py的locate()级联链。
**当前仍是独立子工程,未改动aoi/下任何文件**——按本项目零回退纪律,需先在
eval_roi_refine.py里真实fit/OOF数据验证net positive,阈值只能在fit数据上标定,
在真正独立的test集上报框命中率+严格IoU,才能考虑转正。

【已验证,暂不转正——见eval_roi_refine.py/eval_roi_refine_loco.py】三条真实数据路线
都没能给出净正结果:①Real-IAD pcb/phone_battery原生只有256×256(比WRN分割用的512还
小),min_native门槛正确地整体自禁用(0/40两类都是),这个数据集本身没有分辨率headroom
可供机制验证,不是bug。②MVTec LOCO(breakfast_box/juice_bottle/splicing_connectors,
原生800~1700px,真有分辨率余量)3/3类OOF gain全负(-0.735/-0.289/-0.565),阈值扫描
全部收敛到网格下限0.05——YOLO是Real-IAD电子件上预训练的,LOCO是日用品,候选框置信度
分布本身不可靠,fit()的OOF门控正确识别并全部禁用(enabled=False×3),test集结果原样
未受影响(Δ=0.000)。③pku_pcb(原生2000~3000px,真PCB,域匹配度最好)缺少clean/normal
参考图,无法套用few-shot结构,未测。**结论:安全门控本身工作正常(3种情形下都没有让
未经验证的机制碰到test结果),但至今没有在任何真实数据上验证出Top-1 ROI净正——不是
机制被证伪,而是"同域+大分辨率+有normal参考"这三个条件本地数据凑不齐,YOLO-LoRA/
WRN-LoRA之外这是第三次在这个方向上拿不到validated positive。** 默认不接入
competition.py,代码留opt-in。"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from .diff_channels import build_6ch
from .model_surgery import make_defect_yolo

SIZE = 640


def _resize_chw_np(x, size=SIZE):
    t = torch.from_numpy(x)[None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0].numpy()


@torch.no_grad()
def _predict_boxes(model, ch6_640, conf_thr, nms_iou=0.5, max_keep=50):
    """同rddn_yolo/eval_fit.py predict_boxes:输出640像素尺度[(x1,y1,x2,y2,conf)],已过NMS。"""
    model.eval()
    x = torch.from_numpy(ch6_640)[None].float()
    if next(model.parameters()).is_cuda:
        x = x.cuda()
    out = model(x)[0]
    pred = out[0].T
    conf = pred[:, 4]
    keep = conf >= conf_thr
    boxes = []
    for cx, cy, w, h, c in pred[keep].cpu().numpy():
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        boxes.append((x1, y1, x2, y2, float(c)))
    boxes.sort(key=lambda b: -b[4])
    kept = []
    for b in boxes:
        if all(_iou_xyxy(b[:4], k[:4]) < nms_iou for k in kept):
            kept.append(b)
        if len(kept) >= max_keep:
            break
    return kept


def _iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def _top_regions_native(full_mask, out_hw, native_hw, topk, pad, min_c, max_c):
    """粗mask连通域按面积取Top-K,转换成native分辨率的裁剪框(与_zoom_refine/crop_cascade
    同款坐标换算,side自适应长边×(1+2pad),夹在[min_c,max_c])。"""
    mh, mw = full_mask.shape
    H, W = native_hw
    n, _, stats, _ = cv2.connectedComponentsWithStats(full_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    order = sorted(range(1, n), key=lambda i: -stats[i][4])[:topk]
    out = []
    for i in order:
        x, y, w, h, a = stats[i]
        if a < 2:
            continue
        cx, cy = (x + w / 2) * W / mw, (y + h / 2) * H / mh
        side = int(min(max(max(w * W / mw, h * H / mh) * (1 + 2 * pad), min_c), max_c))
        x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
        y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
        x1, y1 = min(W, x0 + side), min(H, y0 + side)
        out.append((x0, y0, x1, y1))
    return out


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


class Top1ROIRefine:
    """topk=1主实验/2挑战组(latency分层用,由外部latency ladder注入,不在这里写死)。"""

    def __init__(self, device="cuda", ckpt="rddn_yolo/defect_yolo.pt", topk=1,
                 min_c=384, max_c=960, pad=1.5):
        self.device = device
        self.ckpt = ckpt
        self.topk = topk
        self.min_c, self.max_c, self.pad = min_c, max_c, pad
        self.model = None
        self.thr = None
        self.enabled = False
        self.gain = None

    def _load_model(self):
        if self.model is not None:
            return
        m = make_defect_yolo()
        m.model.load_state_dict(torch.load(self.ckpt, map_location="cpu"))
        m.model.to(self.device).eval()
        self.model = m

    def _refine_once(self, det, img, full_mask, out_hw, thr):
        native_img = img if img.dim() == 3 else img[0]
        H, W = native_img.shape[-2:]
        fh, fw = out_hw
        regions = _top_regions_native(full_mask, out_hw, (H, W), self.topk,
                                      self.pad, self.min_c, self.max_c)
        if not regions:
            return full_mask
        ref = det._ref_bank.aligned_ref(native_img)
        if ref.shape[-2:] != native_img.shape[-2:]:
            ref = F.interpolate(ref[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        out = full_mask.copy()
        for x0, y0, x1, y1 in regions:
            crop_img = native_img[:, y0:y1, x0:x1]
            crop_ref = ref[:, y0:y1, x0:x1]
            ch6 = build_6ch(crop_img, crop_ref)
            ch6_640 = _resize_chw_np(ch6)
            boxes = _predict_boxes(self.model.model, ch6_640, conf_thr=thr)
            if not boxes:
                continue                                    # 低置信:该ROI原样保留WRN结果,不动
            cw, ch_ = x1 - x0, y1 - y0
            for bx0, by0, bx1, by1, _ in boxes:
                nx0, ny0 = bx0 * cw / SIZE + x0, by0 * ch_ / SIZE + y0
                nx1, ny1 = bx1 * cw / SIZE + x0, by1 * ch_ / SIZE + y0
                mx0, my0 = int(nx0 * fw / W), int(ny0 * fh / H)
                mx1, my1 = max(mx0 + 1, int(nx1 * fw / W)), max(my0 + 1, int(ny1 * fh / H))
                mx0, my0 = max(0, mx0), max(0, my0); mx1, my1 = min(fw, mx1), min(fh, my1)
                if mx1 > mx0 and my1 > my0:
                    out[my0:my1, mx0:mx1] = 1                # 并集,不替换——低置信区不touch
        return out

    def fit(self, det, ref_bank, defect_imgs, defect_masks, normal_imgs, min_native=700):
        """阈值只在fit数据(30张)上标定:F1-optimal扫描;OOF增益按原图3取1留出折估计,
        只有净正增益(>0.01,同crop_cascade门槛)才self.enabled=True,否则locate()原样
        返回,零回退。"""
        native = [(img, mk) for img, mk in zip(defect_imgs, defect_masks)
                 if max(img.shape[-2:]) >= min_native]
        if len(native) < 6:
            self.enabled = False
            return
        self._load_model()

        hold_idx = set(range(0, len(native), 3))
        train_idx = [i for i in range(len(native)) if i not in hold_idx]

        # 阈值标定:只用train折,对每个候选区域缓存低阈值全量框,扫F1最优阈值(只看fit数据)
        cached = []
        for i in train_idx:
            img, mk = native[i]
            H, W = img.shape[-2:]
            regions = _top_regions_native(mk, mk.shape, (H, W), self.topk, self.pad, self.min_c, self.max_c)
            ref = ref_bank.aligned_ref(img if img.dim() == 3 else img[0])
            if ref.shape[-2:] != img.shape[-2:]:
                ref = F.interpolate(ref[None], size=(H, W), mode="bilinear", align_corners=False)[0]
            for x0, y0, x1, y1 in regions:
                crop_img = img[:, y0:y1, x0:x1]; crop_ref = ref[:, y0:y1, x0:x1]
                gt_crop = mk[int(y0 * mk.shape[0] / H):int(y1 * mk.shape[0] / H),
                             int(x0 * mk.shape[1] / W):int(x1 * mk.shape[1] / W)]
                ch6_640 = _resize_chw_np(build_6ch(crop_img, crop_ref))
                boxes = _predict_boxes(self.model.model, ch6_640, conf_thr=0.02)
                has_gt = gt_crop.sum() > 0
                cached.append((boxes, has_gt))
        if not cached:
            self.enabled = False
            return
        thr_grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
        best_thr, best_f1 = 0.25, -1.0
        for t in thr_grid:
            tp = fp = fn = 0
            for boxes, has_gt in cached:
                fired = any(c >= t for *_, c in boxes)
                if has_gt and fired:
                    tp += 1
                elif has_gt and not fired:
                    fn += 1
                elif not has_gt and fired:
                    fp += 1
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        self.thr = best_thr

        # OOF增益:hold折上,WRN粗mask本身 vs +ROI精修,逐图IoU对比(全独立于阈值标定用的train折)
        gains = []
        for i in hold_idx:
            img, mk = native[i]
            full_amap = det.segment(img)
            fh, fw = full_amap.shape
            base_thr = det.pix_thr if det.pix_thr is not None else float(full_amap.mean() + 3 * full_amap.std())
            full_mask = (full_amap >= base_thr).astype(np.uint8)
            gt = (F.interpolate(torch.from_numpy(mk.astype(np.float32))[None, None],
                                size=(fh, fw), mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
            base_iou = _per_image_iou(full_mask, gt)
            refined = self._refine_once(det, img, full_mask, (fh, fw), self.thr)
            refined_iou = _per_image_iou(refined, gt)
            gains.append(refined_iou - base_iou)
        self.gain = float(np.mean(gains)) if gains else -1.0
        self.enabled = self.gain > 0.01

    def refine(self, det, img, full_mask, out_hw):
        """生产入口:未启用(OOF未验证净正)直接原样返回,零回退。"""
        if not self.enabled or self.model is None or self.thr is None:
            return full_mask
        return self._refine_once(det, img, full_mask, out_hw, self.thr)

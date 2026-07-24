"""Top-1参考ROI精修——大图机制验证(MVTec LOCO,structural_anomalies子集)。
pcb/phone_battery在Real-IAD里原生只有256×256(比WRN分割用的512还小),"恢复大图
下采样丢失的分辨率"这个前提在那两类上不成立,min_native门槛已如实全禁用(0/40触发,
非bug)。LOCO(breakfast_box/juice_bottle/splicing_connectors)原生800~1700px,真正
有超过512/640的分辨率headroom可供ROI机制验证,且有正规train/good正常图(fit_fewshot
few-shot结构可直接套用)。

⚠️域不匹配说明(诚实标注,同AD2的警示一致):RDDN-YOLO候选框是在Real-IAD手机/电子件
12类上预训练的,LOCO(早餐盒/果汁瓶/订书钉袋)不是电子件质检场景,候选框质量本身没有
目标域代表性。这里只验证"给定这个候选检测器,原生分辨率裁剪相对全图降采样是否有净
增益"这个机制假设,不代表对手机部件目标域的精度证据。只用structural_anomalies(物理/
像素级缺陷),不用logical_anomalies(逻辑异常,ROI局部裁剪反而丢失全局上下文,不是
Top-1 ROI设计要解决的问题类型)。

用法:PYTHONPATH=. python rddn_yolo/eval_roi_refine_loco.py
"""
import glob
import random
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.seg_head import map_to_boxes, merge_boxes
from aoi.imageio import load_fast
from rddn_yolo.roi_refine import Top1ROIRefine

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("data/_dl/mvtec_loco")


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def _gt_boxes(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in
            (stats[i] for i in range(1, n)) if a >= 4]


def _box_hit(pred_boxes, gtbs, thr=0.5):
    if not gtbs:
        return None
    hit = sum(1 for g in gtbs if any(_box_iou(p[:4], g) >= thr for p in pred_boxes))
    return hit / len(gtbs)


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def _union_mask(gt_dir):
    m = None
    for mp in sorted(gt_dir.glob("*.png")):
        arr = (np.array(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        m = arr if m is None else (m | arr)
    return m


def prep_loco(cat, n_norm=100, n_fit=15, seed=0):
    root = ROOT / cat
    normals = [load_fast(p) for p in sorted((root / "train" / "good").glob("*.png"))[:n_norm]]
    struct_imgs = sorted((root / "test" / "structural_anomalies").glob("*.png"))
    random.Random(seed).shuffle(struct_imgs)
    fit_p, test_p = struct_imgs[:n_fit], struct_imgs[n_fit:]
    fit_i = [load_fast(p) for p in fit_p]
    fit_m = [_union_mask(root / "ground_truth" / "structural_anomalies" / p.stem) for p in fit_p]
    test_defs = [(load_fast(p), _union_mask(root / "ground_truth" / "structural_anomalies" / p.stem))
                for p in test_p]
    return normals, fit_i, fit_m, test_defs


def evaluate(cat):
    normals, fit_i, fit_m, test_defs = prep_loco(cat)
    print(f"{cat}: normals={len(normals)} fit={len(fit_i)} test={len(test_defs)} "
          f"原生尺寸样例={tuple(fit_i[0].shape[-2:])}", flush=True)
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)

    roi = Top1ROIRefine(device=DEV)
    roi.fit(det, det._ref_bank, fit_i, fit_m, normals)
    print(f"{cat}: ROI-refine fit阶段 thr={roi.thr} OOF gain={roi.gain} enabled={roi.enabled}", flush=True)

    base_ious, base_hits, ref_ious, ref_hits, lat_base, lat_add = [], [], [], [], [], []
    n_fired = 0
    for img, gt in test_defs:
        t0 = time.perf_counter(); o = det.locate(img); t1 = time.perf_counter()
        lat_base.append((t1 - t0) * 1000)
        gtb = _gt_boxes(gt)
        if o.get("mask") is None:
            base_ious.append(0.0); base_hits.append(0.0)
            ref_ious.append(0.0); ref_hits.append(0.0)
            lat_add.append(0.0)
            continue
        mask = o["mask"]
        gt_r = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        base_ious.append(_per_image_iou(mask, gt_r))
        base_hits.append(_box_hit(o["boxes"], gtb) or 0.0)

        t2 = time.perf_counter(); refined = roi.refine(det, img, mask, mask.shape); t3 = time.perf_counter()
        lat_add.append((t3 - t2) * 1000)
        if not np.array_equal(refined, mask):
            n_fired += 1
        boxes = merge_boxes(map_to_boxes(refined.astype(np.float32), 0.5, min_area_frac=0.0002, close=0),
                            getattr(det, "box_merge_d", 0))
        ref_ious.append(_per_image_iou(refined, gt_r))
        ref_hits.append(_box_hit(boxes, gtb) or 0.0)

    n = len(test_defs)
    print(f"{cat} baseline   IoU={np.mean(base_ious):.3f} 框命中@0.5={np.mean(base_hits):.3f} "
          f"locate={np.mean(lat_base):.0f}ms", flush=True)
    print(f"{cat} +ROI精修   IoU={np.mean(ref_ious):.3f} 框命中@0.5={np.mean(ref_hits):.3f} "
          f"Δ(IoU)={np.mean(ref_ious)-np.mean(base_ious):+.3f} Δ(框命中)={np.mean(ref_hits)-np.mean(base_hits):+.3f}",
          flush=True)
    fired_lat = [l for l in lat_add if l > 0]
    print(f"{cat} ROI分支启用比例={n_fired}/{n}={n_fired/max(n,1):.2f} "
          f"每图新增时延(仅触发图均值)={np.mean(fired_lat) if fired_lat else 0:.1f}ms "
          f"(全部test图均值)={np.mean(lat_add):.1f}ms", flush=True)
    return dict(cat=cat, base_iou=np.mean(base_ious), base_hit=np.mean(base_hits),
               ref_iou=np.mean(ref_ious), ref_hit=np.mean(ref_hits), enabled=roi.enabled)


def main():
    torch.manual_seed(0)
    print("=== ⚠️域不匹配警示:YOLO在Real-IAD电子件预训练,LOCO是日用品——只验证ROI机制"
          "本身(原生分辨率headroom下是否有净增益),不代表目标域精度证据 ===", flush=True)
    rows = [evaluate(cat) for cat in ["breakfast_box", "juice_bottle", "splicing_connectors"]]
    print("\n=== 汇总(逐类看Δ,不看均值——历史教训:均值会掩盖反号) ===", flush=True)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()

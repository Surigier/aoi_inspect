"""Top-1参考ROI精修——真实竞赛口径验证(pcb/phone_battery,待办#2主拉分点两类)。
复用scripts/run_scorecard.py同款生产harness(真实CompetitionLargeDetector.fit_fewshot+
locate(),load_fast同口径),在locate()产出的mask基础上再接一层Top1ROIRefine.refine(),
对比baseline(生产原样)vs +ROI精修:
①阈值只在fit(30张)侧标定(Top1ROIRefine.fit()内部完成,test集全程不碰)
②独立test集(留出40张,从未参与fit)报告框命中率@0.5 + 严格IoU
③报告含SAM/crop_cascade/comp_graph/低置信回退的完整管线结果(直接用det.locate()的
  最终mask再叠ROI精修,不是单独测候选框proposer)
④对每张test图强制走完整异常路径计时(test_defs全部是真缺陷图,locate()天然不会走
  正常图早退分支)
⑤报告ROI分支实际启用比例(命中候选框的图占比)+ 每图新增时延(locate耗时 vs
  locate+refine总耗时之差)

用法:PYTHONPATH=. python rddn_yolo/eval_roi_refine.py
"""
import glob
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from aoi.competition import CompetitionLargeDetector
from aoi.seg_head import map_to_boxes, merge_boxes
from aoi.imageio import load_fast
from rddn_yolo.roi_refine import Top1ROIRefine

RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def prep_realiad(cat, hw=(256, 256)):
    def _read(p):
        if not Path(p).exists():
            return np.zeros(hw, np.uint8)
        return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)
    from PIL import Image
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [load_fast(R / x["image_path"]) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit_i = [load_fast(R / x["image_path"]) for x in ng[:30]]
    fit_m = [_read(R / x["mask_path"]) for x in ng[:30]]
    test_defs = [(load_fast(R / x["image_path"]), _read(R / x["mask_path"])) for x in ng[30:70]]
    return normals, fit_i, fit_m, test_defs


def evaluate(cat):
    normals, fit_i, fit_m, test_defs = prep_realiad(cat)
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)

    roi = Top1ROIRefine(device=DEV)
    roi.fit(det, det._ref_bank, fit_i, fit_m, normals)          # 阈值+enabled门槛只碰fit(30张)
    print(f"{cat}: ROI-refine fit阶段 thr={roi.thr} OOF gain={roi.gain} enabled={roi.enabled}", flush=True)

    base_ious, base_hits = [], []
    ref_ious, ref_hits = [], []
    lat_base, lat_add = [], []
    n_fired = 0
    for img, gt in test_defs:
        t0 = time.perf_counter()
        o = det.locate(img)
        t1 = time.perf_counter()
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

        t2 = time.perf_counter()
        refined = roi.refine(det, img, mask, mask.shape)
        t3 = time.perf_counter()
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
    print(f"{cat} ROI分支启用比例={n_fired}/{n}={n_fired/max(n,1):.2f} "
          f"每图新增时延(仅触发图均值)={np.mean([l for l in lat_add if l>0]) if any(l>0 for l in lat_add) else 0:.1f}ms "
          f"(全部test图均值,含未触发图0ms)={np.mean(lat_add):.1f}ms", flush=True)
    return dict(cat=cat, base_iou=np.mean(base_ious), base_hit=np.mean(base_hits),
               ref_iou=np.mean(ref_ious), ref_hit=np.mean(ref_hits), enabled=roi.enabled)


def main():
    torch.manual_seed(0)
    rows = [evaluate(cat) for cat in ["pcb", "phone_battery"]]
    print("\n=== 汇总(仅供参考,决策以enabled门控+逐类Δ为准,不看均值掩盖反号) ===", flush=True)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()

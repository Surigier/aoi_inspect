"""TTA(测试时增强,水平翻转取logit均值)真实数据验证:同一次fit,post-hoc切换
det.use_tta对比raw vs TTA的纯定位IoU/框命中,外加缺陷图locate()延时代价(TTA只在
判缺陷时多跑一次翻转前向,正常图零开销)。
覆盖:Real-IAD pcb/battery(本项目历史最弱两类,TTA的边界稳定性假说最该在这兑现)+
AD2 sheet_metal/walnuts/fruit_jelly(广度检查,不能只在弱类上测,要看是否普遍安全)。
用法:PYTHONPATH=. python scripts/run_tta_ab.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_realiad, gt_boxes, box_hit
from scripts.run_seg_head_ab import prep_ad2

JOBS = [
    ("pcb", lambda: prep_realiad("pcb")),
    ("phone_battery", lambda: prep_realiad("phone_battery")),
    ("sheet_metal", lambda: (*prep_ad2("sheet_metal"),)),
    ("walnuts", lambda: (*prep_ad2("walnuts"),)),
    ("fruit_jelly", lambda: (*prep_ad2("fruit_jelly"),)),
]


def eval_mode(det, test_defs, use_tta):
    det.use_tta = use_tta
    ious, hits, lats = [], [], []
    for img, gt in test_defs:
        t0 = time.perf_counter()
        o = det.locate(img)
        lats.append((time.perf_counter() - t0) * 1000)
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        ious.append(TP / max(TP + FP + FN, 1))
        if o["is_defect"]:
            h = box_hit(o["boxes"], gt_boxes(gt))
            hits.append(h if h is not None else 0.0)
        else:
            hits.append(0.0)
    return float(np.mean(ious)), float(np.mean(hits)), float(np.mean(lats))


def main():
    torch.manual_seed(0)
    rows = []
    for name, prep in JOBS:
        data = prep()
        normals, fit_i, fit_m, test_defs = data[0], data[1], data[2], data[3]
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        iou_r, hit_r, lat_r = eval_mode(det, test_defs, use_tta=False)
        iou_t, hit_t, lat_t = eval_mode(det, test_defs, use_tta=True)
        rows.append((iou_r, iou_t, hit_r, hit_t))
        print(f"{name:14s} raw纯定位={iou_r:.3f} +TTA={iou_t:.3f} Δ={iou_t-iou_r:+.3f} | "
              f"raw框={hit_r:.3f} +TTA框={hit_t:.3f} Δ框={hit_t-hit_r:+.3f} | "
              f"locate延时 raw={lat_r:.0f}ms TTA={lat_t:.0f}ms", flush=True)
    m = np.array(rows).mean(axis=0)
    print(f"\n=== 均值 === raw纯定位={m[0]:.3f} +TTA={m[1]:.3f} Δ={m[1]-m[0]:+.3f} | "
          f"raw框={m[2]:.3f} +TTA框={m[3]:.3f} Δ框={m[3]-m[2]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

"""大图版竞赛成绩单(补小图640代理的缺口):AD2真大图(2232×1024,带GT掩膜)走生产
load_fast(native解码,2500²同口径)+ 全量locate()(EAD+DINO门+WRN定位+SAM)。
报与run_scorecard.py同口径:图级acc/含漏检IoU/纯定位IoU/框命中@0.5/延时。
注:AD2是MVTec2025地狱级对抗数据(反光/背光/透明),是大图下限;隐藏手机件域更接近Real-IAD。
用法:PYTHONPATH=. python scripts/run_scorecard_large.py [类名...]
"""
import sys
import glob
import random
import time
from pathlib import Path
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import img_iou, gt_boxes, box_hit, _read

HW = (256, 256)
AD2 = Path("data/mvtec_ad_2")
CLASSES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def prep_ad2(cat, n_norm=100, n_fit=30):
    root = AD2 / cat
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    gt = root / "test_public/ground_truth/bad"
    def m(p):
        return _read(str(gt / (Path(p).stem + "_mask.png")), HW)
    fit_b, test_b = bad[:n_fit], bad[n_fit:n_fit + 40]
    fit_i = [load_fast(p) for p in fit_b]; fit_m = [m(p) for p in fit_b]
    test_defs = [(load_fast(p), m(p)) for p in test_b]
    goods = [(load_fast(p), None) for p in sorted(glob.glob(str(root / "test_public/good/*.png")))[:40]]
    return normals, fit_i, fit_m, test_defs, goods


SEG_IN = int(__import__("os").environ.get("SEG_IN", "512"))


def evaluate(name, normals, fit_i, fit_m, test_defs, test_goods):
    det = CompetitionLargeDetector(seg_in=SEG_IN)
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    ious_g, ious_p, hits, lats = [], [], [], []
    n_ok = 0; total = 0
    for img, gt in test_defs:
        t0 = time.perf_counter(); o = det.locate(img)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        lats.append((time.perf_counter() - t0) * 1000); total += 1
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (o["anomaly_map"] >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        iou = TP / max(TP + FP + FN, 1); ious_p.append(iou)
        if o["is_defect"]:
            n_ok += 1; ious_g.append(iou)
            h = box_hit(o["boxes"], gt_boxes(gt))
            if h is not None:
                hits.append(h)
        else:
            ious_g.append(0.0); hits.append(0.0)
    for img, _ in test_goods:
        total += 1
        if not det.locate(img)["is_defect"]:
            n_ok += 1
    print(f"{name:14s} 图级acc={n_ok/max(total,1):.3f} | 含漏检IoU={np.mean(ious_g):.3f} 纯定位={np.mean(ious_p):.3f} "
          f"| 框命中@0.5={np.mean(hits):.3f} | DINO={det._dino is not None} | locate={np.mean(lats):.0f}ms", flush=True)
    return np.mean(ious_g), np.mean(ious_p), np.mean(hits)


def main():
    torch.manual_seed(0)
    cats = sys.argv[1:] or CLASSES
    print(f"=== 大图成绩单 AD2真大图(2232×1024,对抗下限;{len(cats)}类,seg_in={SEG_IN},无AUROC)===", flush=True)
    rows = []
    for cat in cats:
        rows.append(evaluate(cat, *prep_ad2(cat)))
    g, p, h = (np.mean([r[i] for r in rows]) for i in range(3))
    print(f"\n均值: 含漏检IoU={g:.3f}  纯定位IoU={p:.3f}  框命中@0.5={h:.3f}", flush=True)


if __name__ == "__main__":
    main()

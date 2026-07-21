"""逻辑缺陷定位真口径基线(方向#2起点,治缺件/错序死穴)。
LOCO 5类逻辑异常(带GT掩膜)按赛题协议:100正常+30标注缺陷 fit,余下 test。
量真口径:含漏检IoU/纯定位IoU/框命中@0.5/延时(无AUROC)。看现状纹理seg-head在
"局部都正常、缺件/错位"的逻辑缺陷上到底烂到什么程度、错在哪 → 决定组件级特征是否对症。
用法:PYTHONPATH=. python scripts/run_logic_scorecard.py
"""
import glob
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from eval.mvtec import _load_img
from scripts.run_scorecard import img_iou, gt_boxes, box_hit

LOCO = Path("data/_dl/mvtec_loco")
HW = (256, 256)


def _union_mask(folder, hw):
    """LOCO 每图一文件夹,内含1+掩膜PNG → 并集。"""
    m = np.zeros(hw, np.uint8)
    for p in sorted(glob.glob(str(folder / "*.png"))):
        a = np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0
        m |= a.astype(np.uint8)
    return m


def prep_logic(cat, n_norm=100, n_fit=30):
    root = LOCO / cat
    normals = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    defs = sorted(glob.glob(str(root / "test/logical_anomalies/*.png")))
    random.Random(0).shuffle(defs)
    gt_dir = root / "ground_truth/logical_anomalies"
    def load(p):
        stem = Path(p).stem
        return _load_img(p, 640), _union_mask(gt_dir / stem, HW)
    fit = [load(p) for p in defs[:n_fit]]
    test = [load(p) for p in defs[n_fit:n_fit + 40]]
    fit_i = [im for im, _ in fit]; fit_m = [mk for _, mk in fit]
    goods = [(_load_img(p, 640), None) for p in
             sorted(glob.glob(str(root / "test/good/*.png")))[:40]]
    return normals, fit_i, fit_m, test, goods


def evaluate(name, normals, fit_i, fit_m, test_defs, test_goods):
    det = CompetitionLargeDetector(train_steps=3000, ead_students=1)   # 快跑基线(与组件图A/B同预算对齐)
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    ious_gated, ious_pure, hits, lats = [], [], [], []
    n_ok = 0; total = 0
    for img, gt in test_defs:
        t0 = time.perf_counter()
        o = det.locate(img)
        lats.append((time.perf_counter() - t0) * 1000)
        total += 1
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        iou = TP / max(TP + FP + FN, 1)
        ious_pure.append(iou)
        if o["is_defect"]:
            n_ok += 1; ious_gated.append(iou)
            h = box_hit(o["boxes"], gt_boxes(gt))
            if h is not None:
                hits.append(h)
        else:
            ious_gated.append(0.0); hits.append(0.0)
    for img, _ in test_goods:
        total += 1
        if not det.locate(img)["is_defect"]:
            n_ok += 1
    print(f"{name:20s} 图级acc={n_ok/max(total,1):.3f} | 含漏检IoU={np.mean(ious_gated):.3f} 纯定位={np.mean(ious_pure):.3f} "
          f"| 框命中@0.5={np.mean(hits):.3f} | DINO={det._dino is not None} | locate={np.mean(lats):.0f}ms", flush=True)
    return np.mean(ious_gated), np.mean(ious_pure), np.mean(hits)


def main():
    torch.manual_seed(0)
    print("=== 逻辑缺陷定位真口径基线(LOCO 5类,含漏检IoU/框,无AUROC)===")
    rows = []
    for cat in ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]:
        rows.append(evaluate(cat, *prep_logic(cat)))
    g, p, h = (np.mean([r[i] for r in rows]) for i in range(3))
    print(f"\n均值: 含漏检IoU={g:.3f}  纯定位IoU={p:.3f}  框命中@0.5={h:.3f}")


if __name__ == "__main__":
    main()

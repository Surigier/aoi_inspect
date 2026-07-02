"""ROI原生放大 × 真2500²级大图定位(AD2=唯一有掩膜的大图数据)。
对比 roi_zoom 关/开 的逐图IoU + locate延时。原生加载(不压1152,保留细节供裁块)。
用法:python scripts/run_roi_zoom.py [cat...]  默认 sheet_metal
"""
import sys
import glob
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = (256, 256)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def iou(pred, gt):
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def run(cat, zoom):
    root = Path(f"data/mvtec_ad_2/{cat}")
    gn = sorted(glob.glob(str(root / "train/good/*.png")))
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png")))
    random.Random(0).shuffle(bad)
    def mpath(p):
        return str(root / "test_public/ground_truth/bad" / (Path(p).stem + "_mask.png"))
    LOAD = 4096                                            # 原生加载(保细节供裁块)
    normals = [load_fast(p, max_size=LOAD) for p in gn[:100]]
    fit_i = [load_fast(p, max_size=LOAD) for p in bad[:30]]
    fit_m = [_read(mpath(p), HW) for p in bad[:30]]
    det = CompetitionLargeDetector(train_steps=200, roi_zoom=zoom)
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    ious, lats = [], []
    for p in bad[30:]:
        img = load_fast(p, max_size=LOAD)
        t0 = time.perf_counter()
        o = det.locate(img)
        lats.append((time.perf_counter() - t0) * 1000)
        pred = o["mask"].astype(bool) if o.get("mask") is not None else (o["anomaly_map"] >= det.pix_thr)
        ious.append(iou(pred, _read(mpath(p), HW)))
    return np.mean(ious), np.mean(lats), len(ious)


def main():
    torch.manual_seed(0)
    cats = sys.argv[1:] or ["sheet_metal"]
    print("=== ROI原生放大 × AD2真大图定位 ===")
    for cat in cats:
        i0, l0, n = run(cat, zoom=False)
        i1, l1, _ = run(cat, zoom=True)
        print(f"{cat:14s} 关zoom: IoU={i0:.3f}/{l0:.0f}ms  开zoom: IoU={i1:.3f}/{l1:.0f}ms  Δ={i1-i0:+.3f} (n={n})", flush=True)


if __name__ == "__main__":
    main()

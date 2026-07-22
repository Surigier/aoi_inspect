"""端到端联合验证:今天全部改动(DINO永远融合+延时梯队重排+seg_head回退)在赛题
真实2500²尺度形状的图上,延时(真实自适应裁剪梯队)+精度(真实per-pixel掩膜)一起
跑一遍。此前延时用AD2/PKU探针测、精度用MVTec/RealIAD/LOCO测,两条线一直分开验,
从未用当前最终代码在真2500²大图上联合确认过。

AD2(sheet_metal/vial等)是本项目里原生分辨率最大(1400²~4224×1056)且有真实
per-pixel掩膜的数据集,是2500²赛题域的最佳可用代理。
probe_paths传真实原生文件路径(不是重建张量),延时口径与submit.py一致。
生产满配置:ead_students=2(默认)/compile_infer=True(对齐submit.py)/sam_refine=True
(默认)/boundary_refine=False(今天验证判负,默认关)/comp_graph=False(默认严格门控)。
用法:PYTHONPATH=. python scripts/run_e2e_2500_check.py
"""
import glob
import random
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast

AD2 = Path("data/mvtec_ad_2")
HW = (256, 256)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def prep(cat, n_norm=100, n_fit=30, n_test=40):
    root = AD2 / cat
    norm_p = sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    gt = root / "test_public/ground_truth/bad"
    m = lambda p: _read(str(gt / (Path(p).stem + "_mask.png")), HW)
    fit_p, test_p = bad[:n_fit], bad[n_fit:n_fit + n_test]
    normals = [load_fast(p) for p in norm_p]
    fit_i = [load_fast(p) for p in fit_p]; fit_m = [m(p) for p in fit_p]
    test_defs = [(load_fast(p), m(p)) for p in test_p]
    all_paths = norm_p + fit_p + test_p                      # 真实原生文件路径(延时探针用)
    return normals, fit_i, fit_m, test_defs, all_paths


def gt_boxes(mask):
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in (stats[i] for i in range(1, n)) if a >= 4]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def box_hit(pred_boxes, gtbs, thr=0.5):
    if not gtbs:
        return None
    hit = sum(1 for g in gtbs if any(box_iou(p[:4], g) >= thr for p in pred_boxes))
    return hit / len(gtbs)


def main():
    torch.manual_seed(0)
    for cat in ["sheet_metal", "vial"]:
        print(f"\n=== {cat} ===", flush=True)
        normals, fit_i, fit_m, test_defs, all_paths = prep(cat)
        sizes = set(tuple(Image.open(p).size) for p in all_paths[:5])
        print(f"原生尺寸样例: {sizes}", flush=True)

        det = CompetitionLargeDetector(compile_infer=True)    # 生产满配置(默认ead_students=2/sam_refine=True)
        det.probe_paths = all_paths                           # 真实原生文件路径,延时口径同submit.py
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)

        print(f"延时裁剪梯队: lat_trimmed={getattr(det, 'lat_trimmed', None)} "
              f"lat_probe_ms={getattr(det, 'lat_probe_ms', None):.0f}ms", flush=True)
        print(f"DINO门={'开' if det._dino is not None else '关'}  SAM={'开' if det.sam is not None else '关'}", flush=True)

        ious, hits, lats = [], [], []
        n_ok = 0
        for img, gt in test_defs:
            t0 = time.perf_counter()
            o = det.locate(img)
            lats.append((time.perf_counter() - t0) * 1000)
            pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
            TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
            iou = TP / max(TP + FP + FN, 1)
            ious.append(iou)
            if o["is_defect"]:
                n_ok += 1
                h = box_hit(o["boxes"], gt_boxes(gt))
                hits.append(h if h is not None else 0.0)
            else:
                hits.append(0.0)
        print(f"图级acc={n_ok/max(len(test_defs),1):.3f}  纯定位IoU={np.mean(ious):.3f}  "
              f"框命中={np.mean(hits):.3f}  单图locate均值={np.mean(lats):.0f}ms p90={np.percentile(lats,90):.0f}ms",
              flush=True)


if __name__ == "__main__":
    main()

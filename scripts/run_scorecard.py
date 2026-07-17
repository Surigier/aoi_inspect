"""终极竞赛口径成绩单(从此只看这张表,无AUROC):
每类报:①图级二值准确率(校准阈值,门控质量)②逐图像素IoU(含漏检=0惩罚/不含 两版)
③框命中率@IoU0.5(预测框vs GT连通域框)④端到端延时。
生产路径 CompetitionLargeDetector(全量train_steps,图级判决真实)。
用法:python scripts/run_scorecard.py
"""
import glob
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def img_iou(amap, gt, thr):
    pred = amap >= thr
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def gt_boxes(mask):
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in
            (stats[i] for i in range(1, n)) if a >= 4]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def box_hit(pred_boxes, gtbs, thr=0.5):
    """图级命中:每个GT框被某预测框IoU≥thr覆盖 → 按GT框召回率计。"""
    if not gtbs:
        return None
    hit = 0
    for g in gtbs:
        if any(box_iou(p[:4], g) >= thr for p in pred_boxes):
            hit += 1
    return hit / len(gtbs)


def evaluate(name, normals, fit_i, fit_m, test_defs, test_goods):
    det = CompetitionLargeDetector()                       # 全量train_steps=10000
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    # 图级二值 + 定位
    n_img_ok = 0; total = 0
    ious_gated, ious_pure, hits, lats = [], [], [], []
    for img, gt in test_defs:
        t0 = time.perf_counter()
        o = det.locate(img)
        lats.append((time.perf_counter() - t0) * 1000)
        total += 1
        if o.get("mask") is not None:                       # 交付掩膜(SAM精化)
            pred = o["mask"].astype(bool)
            TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
            iou = TP / max(TP + FP + FN, 1)
        else:
            iou = img_iou(det.segment(img), gt, det.pix_thr)
        ious_pure.append(iou)
        if o["is_defect"]:
            n_img_ok += 1
            ious_gated.append(iou)
            h = box_hit(o["boxes"], gt_boxes(gt))
            if h is not None:
                hits.append(h)
        else:
            ious_gated.append(0.0)                          # 漏检=定位0分
            hits.append(0.0)
    for img, _ in test_goods:
        o = det.locate(img)
        total += 1
        if not o["is_defect"]:
            n_img_ok += 1
    acc = n_img_ok / max(total, 1)
    print(f"{name:18s} 图级acc={acc:.3f} | 逐图IoU 含漏检={np.mean(ious_gated):.3f} 纯定位={np.mean(ious_pure):.3f} "
          f"| 框命中@0.5={np.mean(hits):.3f} | locate={np.mean(lats):.0f}ms", flush=True)
    return acc, np.mean(ious_gated), np.mean(ious_pure), np.mean(hits)


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [(_load_img(p, 640), None) for p in sorted(glob.glob(str(root / "test/good/*.png")))[:40]]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit_i = [_load_img(p, 640) for p, _ in df[:k]]
    fit_m = [_read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW) for p, fo in df[:k]]
    test_defs = [(_load_img(p, 640), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    return normals, fit_i, fit_m, test_defs, goods


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 640) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit_i = [_load_img(R / x["image_path"], 640) for x in ng[:30]]
    fit_m = [_read(R / x["mask_path"], HW) for x in ng[:30]]
    test_defs = [(_load_img(R / x["image_path"], 640), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    goods = [(_load_img(R / x["image_path"], 640), None) for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, fit_i, fit_m, test_defs, goods


def main():
    torch.manual_seed(0)
    print("=== 竞赛口径成绩单(图级acc/含漏检IoU/框命中@0.5/延时,无AUROC)===")
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("电子 pcb", lambda: prep_realiad("pcb")),
        ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    rows = []
    for name, prep in jobs:
        rows.append(evaluate(name, *prep()))
    a, g, p, h = (np.mean([r[i] for r in rows]) for i in range(4))
    print(f"\n均值: 图级acc={a:.3f}  含漏检IoU={g:.3f}  纯定位IoU={p:.3f}  框命中@0.5={h:.3f}")


if __name__ == "__main__":
    main()

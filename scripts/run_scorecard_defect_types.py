"""按赛题官方缺陷类型分层的真实成绩单(补此前只按产品类目报数的完整度缺口)。
赛题原文:"处理尺寸偏差、缺件少件、逻辑错误(如顺序错误)、色彩变化与常见外观缺陷"。
已有production scorecard(run_scorecard.py)按产品类目报数,能对上号的3类:
  常见外观缺陷→hazelnut  缺件少件→cable  色彩变化→pill
本脚本补另外两类,用MVTec LOCO的官方细分子集(真实数据,非代理猜测):
  logical_anomalies →"缺件/错位/组合错误"(计数错/位置错/搭配错,如splicing_connectors
    的线缆长度错——这类里包含尺寸偏差的样本,但LOCO官方定义整体上不等于赛题例子
    "顺序错误"字面意思,可能是空间拼接顺序或时序装配顺序,见邮件问出题人第1条)
  structural_anomalies →"常见外观缺陷"的另一路真实数据(补充hazelnut)

用法:PYTHONPATH=. python scripts/run_scorecard_defect_types.py
"""
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
ROOT = Path("data/_dl/mvtec_loco")
HW = (256, 256)


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def gt_boxes(mask):
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in
            (stats[i] for i in range(1, n)) if a >= 4]


def box_hit(pred_boxes, gtbs, thr=0.5):
    if not gtbs:
        return None
    hit = sum(1 for g in gtbs if any(_box_iou(p[:4], g) >= thr for p in pred_boxes))
    return hit / len(gtbs)


def img_iou(amap, gt, thr):
    pred = amap >= thr
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def _union_mask(gt_dir, hw=HW):
    m = None
    for mp in sorted(gt_dir.glob("*.png")):
        arr = np.array(Image.open(mp).convert("L").resize((hw[1], hw[0]))) > 0
        m = arr.astype(np.uint8) if m is None else (m | arr.astype(np.uint8))
    return m if m is not None else np.zeros(hw, np.uint8)


def prep_loco(cat, anomaly_type, n_norm=100, n_fit=15, seed=0):
    """anomaly_type: 'logical_anomalies' 或 'structural_anomalies'。"""
    root = ROOT / cat
    normals = [load_fast(p) for p in sorted((root / "train" / "good").glob("*.png"))[:n_norm]]
    imgs = sorted((root / "test" / anomaly_type).glob("*.png"))
    random.Random(seed).shuffle(imgs)
    fit_p, test_p = imgs[:n_fit], imgs[n_fit:]
    fit_i = [load_fast(p) for p in fit_p]
    fit_m = [_union_mask(root / "ground_truth" / anomaly_type / p.stem) for p in fit_p]
    goods = [(load_fast(p), None) for p in sorted((root / "test" / "good").glob("*.png"))[:40]]
    test_defs = [(load_fast(p), _union_mask(root / "ground_truth" / anomaly_type / p.stem))
                for p in test_p]
    return normals, fit_i, fit_m, test_defs, goods


def evaluate(name, normals, fit_i, fit_m, test_defs, test_goods):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    n_img_ok = 0; total = 0
    ious_gated, ious_pure, hits, lats = [], [], [], []
    for img, gt in test_defs:
        t0 = time.perf_counter()
        o = det.locate(img)
        lats.append((time.perf_counter() - t0) * 1000)
        total += 1
        if o.get("mask") is not None:
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
            ious_gated.append(0.0)
            hits.append(0.0)
    for img, _ in test_goods:
        o = det.locate(img)
        total += 1
        if not o["is_defect"]:
            n_img_ok += 1
    acc = n_img_ok / max(total, 1)
    print(f"{name:28s} 图级acc={acc:.3f} | 逐图IoU 含漏检={np.mean(ious_gated):.3f} 纯定位={np.mean(ious_pure):.3f} "
          f"| 框命中@0.5={np.mean(hits):.3f} | locate={np.mean(lats):.0f}ms", flush=True)
    return acc, np.mean(ious_gated), np.mean(ious_pure), np.mean(hits)


def main():
    torch.manual_seed(0)
    print("=== 缺陷类型分层成绩单(补足赛题5类缺陷里未验证的\"尺寸偏差/逻辑错误\") ===", flush=True)
    jobs = [
        ("常见外观缺陷(LOCO breakfast_box结构)", "breakfast_box", "structural_anomalies"),
        ("常见外观缺陷(LOCO juice_bottle结构)", "juice_bottle", "structural_anomalies"),
        ("缺件/错位/组合(LOCO breakfast_box逻辑)", "breakfast_box", "logical_anomalies"),
        ("缺件/错位/组合(LOCO splicing_connectors逻辑)", "splicing_connectors", "logical_anomalies"),
        ("缺件/错位/组合(LOCO screw_bag逻辑)", "screw_bag", "logical_anomalies"),
    ]
    rows = []
    for name, cat, atype in jobs:
        rows.append(evaluate(name, *prep_loco(cat, atype)))
    a, g, p, h = (np.mean([r[i] for r in rows]) for i in range(4))
    print(f"\n均值: 图级acc={a:.3f}  含漏检IoU={g:.3f}  纯定位IoU={p:.3f}  框命中@0.5={h:.3f}")
    print("\n⚠️注:LOCO logical_anomalies是\"计数/位置/搭配错误\",不完全等同赛题原文\"顺序错误\"")
    print("字面意思(可能指拼接顺序或装配时序)——这条已列入给出题人的邮件问题清单。")


if __name__ == "__main__":
    main()

"""A/B 记分卡:同一 detector(fit一次)上对比 EAD-only 门 vs DINOv2 融合门。
像素路径与基线完全相同(DINO只改图级判决),故一趟拿到逐类净增益 Δ含漏检/Δ图级acc/Δ延时。
用法:PYTHONPATH=. python scripts/run_scorecard_ab.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import (prep_mvtec, prep_realiad, img_iou, gt_boxes, box_hit)


def evaluate_ab(name, normals, fit_i, fit_m, test_defs, test_goods):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    dino_on = det._dino is not None
    # 逐缺陷图:纯定位IoU(不受门影响)+ 两门各自的判决
    g_ead, g_fus, lats = [], [], []
    for img, gt in test_defs:
        t0 = time.perf_counter()
        o = det.locate(img)                                 # 走融合门(生产)
        lats.append((time.perf_counter() - t0) * 1000)
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        iou = TP / max(TP + FP + FN, 1)
        ead_def = det.threshold is not None and o["score"] >= det.threshold   # EAD-only 反事实
        g_ead.append(iou if ead_def else 0.0)
        g_fus.append(iou if o["is_defect"] else 0.0)
    # 正常图:两门各自误报
    ng_ead = ng_fus = 0
    for img, _ in test_goods:
        o = det.locate(img)
        ead_def = det.threshold is not None and o["score"] >= det.threshold
        ng_ead += (not ead_def); ng_fus += (not o["is_defect"])
    nD, nG = len(test_defs), len(test_goods)
    recD_e = np.mean([v > 0 for v in g_ead]); recD_f = np.mean([v > 0 for v in g_fus])
    acc_e = (np.sum([v > 0 for v in g_ead]) + ng_ead) / max(nD + nG, 1)
    acc_f = (np.sum([v > 0 for v in g_fus]) + ng_fus) / max(nD + nG, 1)
    print(f"{name:18s} DINO启用={dino_on} | EAD门: 图级acc={acc_e:.3f} 召回={recD_e:.3f} 含漏检IoU={np.mean(g_ead):.3f}"
          f"  →  融合门: 图级acc={acc_f:.3f} 召回={recD_f:.3f} 含漏检IoU={np.mean(g_fus):.3f}"
          f"  | Δacc={acc_f-acc_e:+.3f} Δ含漏检={np.mean(g_fus)-np.mean(g_ead):+.3f} | locate={np.mean(lats):.0f}ms", flush=True)
    return acc_e, acc_f, np.mean(g_ead), np.mean(g_fus)


def main():
    torch.manual_seed(0)
    print("=== A/B 记分卡:EAD门 vs DINOv2融合门(纯定位路径相同)===")
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("电子 pcb", lambda: prep_realiad("pcb")),
        ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    rows = []
    for name, prep in jobs:
        rows.append(evaluate_ab(name, *prep()))
    ae, af, ge, gf = (np.mean([r[i] for r in rows]) for i in range(4))
    print(f"\n均值: 图级acc {ae:.3f}→{af:.3f} (Δ{af-ae:+.3f}) | 含漏检IoU {ge:.3f}→{gf:.3f} (Δ{gf-ge:+.3f})")


if __name__ == "__main__":
    main()

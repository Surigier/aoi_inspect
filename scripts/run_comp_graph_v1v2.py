"""component_graph v1(纯z-score) vs v2(Hungarian匹配)受控对比:同一次fit,只切
ComponentGraph.use_hungarian开关,隔离算法差异,不受EAD/DINO/seg_head训练随机性
混杂(run_comp_graph_ab.py两次独立fit的结果混杂了这个,juice_bottle基线从0.431→
0.170跳变,不能直接比较v1/v2孰优孰劣)。
用法:PYTHONPATH=. python scripts/run_comp_graph_v1v2.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.component_graph import ComponentGraph
from scripts.run_logic_scorecard import prep_logic
from scripts.run_scorecard import gt_boxes, box_hit

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]


def eval_mode(det, test_defs, cg):
    det.comp_graph = cg
    ious_gated, hits = [], []
    for img, gt in test_defs:
        o = det.locate(img)
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        iou = TP / max(TP + FP + FN, 1)
        if o["is_defect"]:
            ious_gated.append(iou)
            h = box_hit(o["boxes"], gt_boxes(gt))
            if h is not None:
                hits.append(h)
        else:
            ious_gated.append(0.0); hits.append(0.0)
    det.comp_graph = None
    return float(np.mean(ious_gated)), float(np.mean(hits))


def main():
    torch.manual_seed(0)
    rows = []
    for cat in CATS:
        normals, fit_i, fit_m, test_defs, goods = prep_logic(cat)
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)

        cg1 = ComponentGraph(device=det._bb_loc.device, use_hungarian=False)
        cg1.comp_masks, cg1.tmpl_gray = None, None
        cg2 = ComponentGraph(device=det._bb_loc.device, use_hungarian=True)
        # 只fit一次(cg2),把学到的组件/统计量直接复用给cg1(纯换算法,同一份组件/统计)
        cg2.fit(det, normals, defect_imgs=fit_i, defect_masks=fit_m)
        n_comp = len(cg2.comp_masks) if cg2.comp_masks else 0
        if n_comp < 2:
            print(f"{cat:20s} 组件数<2,跳过", flush=True)
            continue
        cg1.comp_masks, cg1.tmpl_gray = cg2.comp_masks, cg2.tmpl_gray
        cg1.mu, cg1.d_mu, cg1.d_sd = cg2.mu, cg2.d_mu, cg2.d_sd

        g_b, h_b = eval_mode(det, test_defs, None)               # base(无组件图)
        g_v1, h_v1 = eval_mode(det, test_defs, cg1)               # 强制开,v1算法
        g_v2, h_v2 = eval_mode(det, test_defs, cg2)               # 强制开,v2算法(同一组件/统计)
        print(f"{cat:20s} 组件数={n_comp}", flush=True)
        print(f"  base       : 含漏检={g_b:.3f} 框={h_b:.3f}", flush=True)
        print(f"  +cg v1(z) : 含漏检={g_v1:.3f} 框={h_v1:.3f}  Δ含漏检={g_v1-g_b:+.3f} Δ框={h_v1-h_b:+.3f}", flush=True)
        print(f"  +cg v2(H) : 含漏检={g_v2:.3f} 框={h_v2:.3f}  Δ含漏检={g_v2-g_b:+.3f} Δ框={h_v2-h_b:+.3f}", flush=True)
        rows.append((g_b, g_v1, g_v2, h_b, h_v1, h_v2))
    if rows:
        m = np.array(rows).mean(axis=0)
        print(f"\n=== 均值(受控,同一fit) === base含漏检={m[0]:.3f} v1={m[1]:.3f} v2={m[2]:.3f} | "
              f"Δv1={m[1]-m[0]:+.3f} Δv2={m[2]-m[0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

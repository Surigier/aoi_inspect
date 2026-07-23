"""component_graph v1(纯z-score,固定槽位) vs v3(use_local_search,槽位周围局部
平移搜索)受控对比:同一次fit,只切算法开关,隔离训练随机性(同run_comp_graph_v1v2.py
的方法论)。v3测试v2判负时点出的假设——单一全局刚性ECC warp限制了错位检测,是否
真的能靠局部搜索找回来;同时要警惕局部搜索的多重比较风险(候选越多越容易凑巧误判)。
用法:PYTHONPATH=. python scripts/run_comp_graph_v1v3.py
"""
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
            hits.append(h if h is not None else 0.0)
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

        cg3 = ComponentGraph(device=det._bb_loc.device, use_local_search=True)
        cg3.fit(det, normals, defect_imgs=fit_i, defect_masks=fit_m)
        n_comp = len(cg3.comp_masks) if cg3.comp_masks else 0
        if n_comp < 2:
            print(f"{cat:20s} 组件数<2,跳过", flush=True)
            continue
        cg1 = ComponentGraph(device=det._bb_loc.device, use_local_search=False)
        cg1.comp_masks, cg1.tmpl_gray = cg3.comp_masks, cg3.tmpl_gray
        cg1.mu, cg1.d_mu, cg1.d_sd = cg3.mu, cg3.d_mu, cg3.d_sd

        g_b, h_b = eval_mode(det, test_defs, None)
        g_v1, h_v1 = eval_mode(det, test_defs, cg1)
        g_v3, h_v3 = eval_mode(det, test_defs, cg3)
        print(f"{cat:20s} 组件数={n_comp}", flush=True)
        print(f"  base          : 含漏检={g_b:.3f} 框={h_b:.3f}", flush=True)
        print(f"  +cg v1(固定槽位): 含漏检={g_v1:.3f} 框={h_v1:.3f}  Δ含漏检={g_v1-g_b:+.3f} Δ框={h_v1-h_b:+.3f}", flush=True)
        print(f"  +cg v3(局部搜索): 含漏检={g_v3:.3f} 框={h_v3:.3f}  Δ含漏检={g_v3-g_b:+.3f} Δ框={h_v3-h_b:+.3f}", flush=True)
        rows.append((g_b, g_v1, g_v3, h_b, h_v1, h_v3))
    if rows:
        m = np.array(rows).mean(axis=0)
        print(f"\n=== 均值(受控,同一fit) === base={m[0]:.3f} v1={m[1]:.3f} v3={m[2]:.3f} | "
              f"Δv1={m[1]-m[0]:+.3f} Δv3={m[2]-m[0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

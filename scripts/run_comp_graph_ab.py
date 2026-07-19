"""优先级4验证:组件图逻辑异常分支(component_graph.py)真实A/B,LOCO 5类逻辑缺陷。
每类fit一次(3000步,run_logic_scorecard.py同预算),script层手动构造ComponentGraph,
post-hoc切换det.comp_graph测 base(无组件图) vs cg(强制开启,无视门控) 两路——
同时报OOF门控自己的决策(enabled/gain),对照test真实Δ看门控判得对不对。
用法:PYTHONPATH=. python scripts/run_comp_graph_ab.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.component_graph import ComponentGraph
from scripts.run_logic_scorecard import prep_logic
from scripts.run_scorecard import gt_boxes, box_hit

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]


def eval_mode(det, test_defs, test_goods, cg):
    det.comp_graph = cg
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
    det.comp_graph = None
    return (n_ok / max(total, 1), float(np.mean(ious_gated)), float(np.mean(ious_pure)),
            float(np.mean(hits)), float(np.mean(lats)))


def main():
    torch.manual_seed(0)
    rows = []
    for cat in CATS:
        normals, fit_i, fit_m, test_defs, goods = prep_logic(cat)
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        cg = ComponentGraph(device=det._bb_loc.device)
        cg.fit(det, normals, defect_imgs=fit_i, defect_masks=fit_m)
        n_comp = len(cg.comp_masks) if cg.comp_masks else 0
        acc_b, g_b, p_b, h_b, l_b = eval_mode(det, test_defs, goods, None)
        if n_comp >= 2:
            acc_c, g_c, p_c, h_c, l_c = eval_mode(det, test_defs, goods, cg)
        else:
            acc_c, g_c, p_c, h_c, l_c = acc_b, g_b, p_b, h_b, l_b
        rows.append((g_b, p_b, h_b, g_c, p_c, h_c))
        print(f"{cat:20s} 组件数={n_comp} 门控gain={cg.gain if cg.gain is not None else float('nan'):+.3f} "
              f"enabled={cg.enabled}", flush=True)
        print(f"  base: 含漏检IoU={g_b:.3f} 纯定位={p_b:.3f} 框={h_b:.3f} acc={acc_b:.3f} {l_b:.0f}ms", flush=True)
        print(f"  +cg : 含漏检IoU={g_c:.3f} 纯定位={p_c:.3f} 框={h_c:.3f} acc={acc_c:.3f} {l_c:.0f}ms "
              f"Δ含漏检={g_c-g_b:+.3f} Δ框={h_c-h_b:+.3f}", flush=True)
    m = np.array(rows).mean(axis=0)
    print(f"\n=== 均值 === base含漏检={m[0]:.3f} 纯定位={m[1]:.3f} 框={m[2]:.3f} | "
          f"+cg含漏检={m[3]:.3f} 纯定位={m[4]:.3f} 框={m[5]:.3f} | Δ含漏检={m[3]-m[0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

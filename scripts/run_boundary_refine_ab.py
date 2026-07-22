"""DCP-SFR边界残差头真实数据验证:生产配置fit(内部已跑k折OOF自动门控),报告
①门控自己在fit留出上的判断(enabled/gain)②不管门控判断如何,都强制训一份边界头
在独立test集上评估真实Δ(纯定位IoU/框命中),对照门控判断是否可信。
DCP-SFR目标场景是"微小缺陷/边界丢失"——Real-IAD pcb/battery(本项目历史最弱两类)+
AD2 sheet_metal(细划痕)最贴近。
用法:PYTHONPATH=. python scripts/run_boundary_refine_ab.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.boundary_refine import BoundaryRefiner
from scripts.run_scorecard import prep_realiad, gt_boxes, box_hit
from scripts.run_seg_head_ab import prep_ad2


def eval_iou_hit(det, test_defs, br):
    orig = det.boundary_refiner
    det.boundary_refiner = br
    ious, hits = [], []
    for img, gt in test_defs:
        o = det.locate(img)
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        ious.append(TP / max(TP + FP + FN, 1))
        if o["is_defect"]:
            h = box_hit(o["boxes"], gt_boxes(gt))
            hits.append(h if h is not None else 0.0)
        else:
            hits.append(0.0)
    det.boundary_refiner = orig
    return float(np.mean(ious)), float(np.mean(hits))


def main():
    torch.manual_seed(0)
    jobs = [
        ("pcb", lambda: prep_realiad("pcb")),
        ("phone_battery", lambda: prep_realiad("phone_battery")),
        ("sheet_metal(AD2)", lambda: prep_ad2("sheet_metal")),
    ]
    rows = []
    for name, prep in jobs:
        data = prep()
        normals, fit_i, fit_m, test_defs = data[0], data[1], data[2], data[3]
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1, boundary_refine=True)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        gated_br = det.boundary_refiner                      # 门控自己的决定(可能是None)
        gate_gain = getattr(gated_br, "gain", None)

        # 无论门控判断如何,强制训一份边界头(全fit数据)做test集真实对照
        forced = BoundaryRefiner(device=det._bb_loc.device)
        base_logits = [det.segment(img) for img in fit_i]
        forced.net = forced._train_one(det, fit_i, base_logits, fit_m, steps=forced.steps, seed=forced.seed)
        forced.enabled = True

        iou_raw, hit_raw = eval_iou_hit(det, test_defs, None)
        iou_ref, hit_ref = eval_iou_hit(det, test_defs, forced)
        rows.append((iou_raw, iou_ref, hit_raw, hit_ref))
        print(f"{name:20s} 门控gain(fit留出)={gate_gain} enabled={gated_br is not None} | "
              f"test集: raw纯定位={iou_raw:.3f} +boundary={iou_ref:.3f} Δ={iou_ref-iou_raw:+.3f} | "
              f"raw框={hit_raw:.3f} +boundary框={hit_ref:.3f} Δ框={hit_ref-hit_raw:+.3f}", flush=True)
    m = np.array(rows).mean(axis=0)
    print(f"\n=== 均值 === raw纯定位={m[0]:.3f} +boundary={m[1]:.3f} Δ={m[1]-m[0]:+.3f} | "
          f"raw框={m[2]:.3f} +boundary框={m[3]:.3f} Δ框={m[3]-m[2]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

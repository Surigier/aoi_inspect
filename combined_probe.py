"""把今天验证过的两个"配置太保守"候选(seg_head核心配方OOF门控 + GCAD-EmbedAE
激进配方)组合起来,在真正的生产5类目scorecard(hazelnut/cable/pill/pcb/
phone_battery)上跑一遍,看合起来的真实效果——不是分别验证时的单独数字,是叠加
后对同一批production评测口径的实际影响。

LoRA门控本次不合并:它要换的是WRN骨干特征本身,和生产_select_feat_mode(768通道
_wrn_feats vs 1536通道_wrn_feats_diff模板差分)有耦合,合并前需要单独确认不会
静默出错,这次先跳过,只测seg_head门控+GCAD两个互相独立、已知安全的候选。

用法:PYTHONPATH=. python combined_probe.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from global_context.eval_global_branch import (
    prep_mvtec, prep_realiad, gt_boxes, box_hit, fit_global_branches, evaluate_variant,
    DinoCLS,
)
from seghead_tuning.gated_train import pick_gated_head


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def baseline_eval(det, test_defs):
    ious, hits = [], []
    for img, gt in test_defs:
        o = det.locate(img)
        if o.get("mask") is None:
            ious.append(0.0); hits.append(0.0); continue
        mask = o["mask"]
        gt_r = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        ious.append(_per_image_iou(mask, gt_r))
        hits.append(box_hit(o["boxes"], gt_boxes(gt)) or 0.0)
    return float(np.mean(ious)), float(np.mean(hits))


def run_one(name, normals, fit_i, fit_m, test_defs):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det.seg_head.head is None:
        print(f"{name}: 生产seg_head未训成功,跳过", flush=True)
        return None
    base_iou, base_hit = baseline_eval(det, test_defs)

    # ① seg_head OOF门控:决定保守/激进,把选中的head/mu/sd/thr真正写回det.seg_head
    picked = pick_gated_head(name, det.seg_head.extractor, fit_i, fit_m)
    if picked is None:
        print(f"{name}: seg_head门控失败,跳过", flush=True)
        return None
    _, (h_g, mu_g, sd_g, thr_g, gate_agg) = picked
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd = h_g, mu_g, sd_g
    det.seg_head.thr = thr_g; det.seg_head.rams = None

    # ② GCAD-EmbedAE(激进配方,900步/3e-3),融合进图级检测
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)
    dino = DinoCLS(device="cuda" if torch.cuda.is_available() else "cpu")
    fns = fit_global_branches(det, normals, fit_i, dino, ae_steps=900, ae_lr=3e-3)
    combo_iou, combo_hit = evaluate_variant(det, *fns["emb"], test_defs)

    print(f"{name:20s} baseline IoU={base_iou:.3f}/hit={base_hit:.3f}  "
          f"+seg_head门控+GCAD-EmbedAE IoU={combo_iou:.3f}/hit={combo_hit:.3f}  "
          f"Δ(IoU)={combo_iou-base_iou:+.3f}", flush=True)
    return dict(base=(base_iou, base_hit), combo=(combo_iou, combo_hit))


def main():
    torch.manual_seed(0)
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("生产:pcb", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        row = run_one(name, *prep())
        if row is None:
            continue
        names.append(name)
        deltas.append(row["combo"][0] - row["base"][0])
    if deltas:
        d = np.array(deltas)
        passed = (np.median(d) >= 0.005 and np.mean(d) > 0
                 and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
        print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
              f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

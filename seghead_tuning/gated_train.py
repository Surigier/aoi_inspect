"""给seg_head核心训练配方(保守300/5e-3 vs 激进900/1e-2)加OOF门控,和wrn_lora/
gated_validate.py同一套思路:30张fit缺陷图内部切train_sub/val_sub,两种配置都只在
train_sub上训,val_sub上比谁强就用谁,决定后再看真正held-out test_defs的表现。
动机:probe_aggressive_train.py测出来是6/10类受益、2类(wood/pcb)因为过拟合明显
变差(wood -0.089破了止损线)——和LoRA那次一模一样的模式,同样的门控解法应该适用。

用法:PYTHONPATH=. python seghead_tuning/gated_train.py
"""
import random
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import _mask_to
from seghead_tuning.probe_aggressive_train import (
    _train_head, _calibrate_thr, _per_image_iou, BASE_STEPS, BASE_LR, AGG_STEPS, AGG_LR, DEV,
)
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit
from scripts.run_scorecard_5types import prep_mvtec_color


@torch.no_grad()
def _eval_head_on(extractor, head, mu, sd, thr, defs, device=DEV):
    ious = []
    for img, mk in defs:
        f = extractor(img)[None].to(device).float()
        logit = head((f - mu) / sd)
        gh, gw = logit.shape[-2:]
        pred = (logit[0, 0].cpu().numpy() >= thr).astype(np.uint8)
        ious.append(_per_image_iou(pred, _mask_to(mk, gh, gw)))
    return float(np.mean(ious)) if ious else 0.0


def run_gated(name, normals, fit_i, fit_m, test_defs, val_frac=0.3, seed=0, device=DEV):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det.seg_head.head is None:
        print(f"{name}: 生产seg_head未训成功,跳过", flush=True)
        return None
    extractor = det.seg_head.extractor

    n = len(fit_i)
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_i = [fit_i[i] for i in train_idx]; train_m = [fit_m[i] for i in train_idx]
    val_defs = [(fit_i[i], fit_m[i]) for i in val_idx]
    print(f"  [{name}] fit={n} -> train_sub={len(train_idx)} val_sub={len(val_idx)}", flush=True)

    with torch.no_grad():
        feats = [extractor(im) for im in train_i]
    grid_hw = feats[0].shape[-2:]
    gts = [torch.from_numpy(_mask_to(m, grid_hw[0], grid_hw[1]).astype(np.float32)) for m in train_m]
    pos_total = sum(float(g.sum()) for g in gts)
    neg_total = len(gts) * grid_hw[0] * grid_hw[1] - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=device)

    def train_and_val(steps, lr):
        head, mu, sd = _train_head(feats, gts, pos_w, steps, lr, device=device)
        thr = _calibrate_thr(extractor, head, mu, sd, train_i, train_m)
        if thr is None:
            return None, None, None, None, 0.0
        val_iou = _eval_head_on(extractor, head, mu, sd, thr, val_defs, device=device)
        return head, mu, sd, thr, val_iou

    h_base, mu_b, sd_b, thr_b, val_base = train_and_val(BASE_STEPS, BASE_LR)
    h_agg, mu_a, sd_a, thr_a, val_agg = train_and_val(AGG_STEPS, AGG_LR)
    if h_base is None or h_agg is None:
        print(f"{name}: 阈值标定失败,跳过", flush=True)
        return None
    gate_agg = val_agg > val_base

    def _test_iou(head, mu, sd, thr):
        orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
        det.seg_head.head, det.seg_head.mu, det.seg_head.sd = head, mu, sd
        det.seg_head.thr = thr; det.seg_head.rams = None
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
        det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams = orig
        return float(np.mean(ious)), float(np.mean(hits))

    test_base, hit_base = _test_iou(h_base, mu_b, sd_b, thr_b)
    test_agg, hit_agg = _test_iou(h_agg, mu_a, sd_a, thr_a)
    test_gated, hit_gated = (test_agg, hit_agg) if gate_agg else (test_base, hit_base)
    d = test_gated - test_base
    print(f"{name}: val(base={val_base:.3f} agg={val_agg:.3f}) gate={'激进' if gate_agg else '保守'}  "
          f"test(base={test_base:.3f} agg={test_agg:.3f} gated={test_gated:.3f}) Δ(gated-base)={d:+.3f}", flush=True)
    return d


def main():
    torch.manual_seed(0)
    print("=== OOF门控seg_head训练配方:留出集自检,过拟合自动退回保守配置 ===", flush=True)
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("色彩 carpet", lambda: prep_mvtec_color("carpet")[:4]),
        ("色彩 leather", lambda: prep_mvtec_color("leather")[:4]),
        ("色彩 metal_nut", lambda: prep_mvtec_color("metal_nut")[:4]),
        ("色彩 wood", lambda: prep_mvtec_color("wood")[:4]),
        ("生产:pcb(微小缺陷)", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        d = run_gated(name, *prep())
        if d is None:
            continue
        names.append(name); deltas.append(d)

    d = np.array(deltas)
    passed = (np.median(d) >= 0.005 and np.mean(d) > 0
              and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
    print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
          f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
    print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

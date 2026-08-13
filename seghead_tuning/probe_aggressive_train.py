"""查"是不是整体都保守了"的最高杠杆一处:生产SupervisedSegHead自己的训练配方
(steps=300/lr=5e-3,aoi/_seg_head_old_ae5fbbb.py:_train_one)有没有偏保守。
动机:WRN-LoRA判负结论被证实是配置太保守的假象(steps 150→300、lr 2e-4→1e-3后
从打平变真实提升),FocalDice用的正是这套steps=300/lr=5e-3的baseline当对照——
如果seg_head自己就没训到位,会同时拖累好几个已判负的旁支实验的baseline,不是
旁支机制真的没用。

对比(同一fit数据、同一det、同一损失函数BCEWithLogitsLoss,唯一变量是steps/lr):
  baseline: steps=300, lr=5e-3(生产现状)
  激进    : steps=900, lr=1e-2(3倍步数+2倍学习率,量级参照LoRA那次从150→300、
            2e-4→1e-3的倍数)

复用det.locate()完整下游(mask/box/IoU/框命中),只替换det.seg_head的
head/mu/sd/thr,其余(EAD/DINO/SAM等)完全不变——和focal_dice_seghead/
eval_focal_dice.py同一套方法论,唯一变量从"损失函数"换成"训练步数/学习率"。

用法:PYTHONPATH=. python seghead_tuning/probe_aggressive_train.py
"""
import numpy as np
import torch
import torch.nn as nn
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit
from scripts.run_scorecard_5types import prep_mvtec_color

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH, SEED = 8, 0
LIGHT_STEPS, LIGHT_LR = 100, 5e-3  # diag_pcb.py测出来的:微小缺陷(pcb)真正的峰值比base还早
BASE_STEPS, BASE_LR = 300, 5e-3
AGG_STEPS, AGG_LR = 900, 1e-2


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def _train_head(feats, gts, pos_w, steps, lr, device=DEV):
    """和生产_train_one同一套loss/优化器/batch/种子,唯一变量是steps/lr。"""
    C = feats[0].shape[0]
    mu = torch.stack([f.float().mean(dim=(1, 2)) for f in feats]).mean(0).view(1, -1, 1, 1).to(device)
    sd = (torch.stack([f.float().std(dim=(1, 2)) for f in feats]).mean(0) + 1e-6).view(1, -1, 1, 1).to(device)
    all_feats = [f.to(device) for f in feats]
    all_gts = [g.to(device) for g in gts]

    torch.manual_seed(SEED)
    lin = _ProdHead._linear_head(C).to(device)
    cnv = _ProdHead._conv_head(C).to(device)
    head = _Ensemble(lin, cnv)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(SEED)
    n = len(all_feats)
    for _ in range(steps):
        sel = torch.randperm(n, generator=g)[:BATCH].tolist()
        X = torch.stack([all_feats[i] for i in sel])
        X = (X - mu) / sd
        Y = torch.stack([all_gts[i] for i in sel])
        opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
    head.eval()
    return head, mu, sd


@torch.no_grad()
def _calibrate_thr(extractor, head, mu, sd, defect_imgs, defect_masks, out_hw=(256, 256)):
    S, L = [], []
    for img, mk in zip(defect_imgs, defect_masks):
        f = extractor(img)[None].to(DEV).float()
        logit = head((f - mu) / sd)
        amap = torch.nn.functional.interpolate(logit, size=out_hw, mode="bilinear", align_corners=False)
        S.append(amap[0, 0].cpu().numpy().ravel())
        L.append(_mask_to(mk, out_hw[0], out_hw[1]).ravel())
    s = np.concatenate(S); l = np.concatenate(L)
    order = np.argsort(-s); ls = l[order]; ss = s[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    if P == 0:
        return None
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(ss[int(np.argmax(f1))])


def evaluate_with_head(det, test_defs):
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
    # DINO门可能被_calibrate_latency在GPU瞬时负载下砍掉(已知风险),不补会导致
    # baseline/激进两次评测用的判定路径跨进程不一致,IoU数字跟着运气跳。
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)
    base_iou, base_hit = evaluate_with_head(det, test_defs)

    extractor = det.seg_head.extractor
    with torch.no_grad():
        feats = [extractor(im) for im in fit_i]
    grid_hw = feats[0].shape[-2:]
    gts = [torch.from_numpy(_mask_to(m, grid_hw[0], grid_hw[1]).astype(np.float32)) for m in fit_m]
    pos_total = sum(float(g.sum()) for g in gts)
    neg_total = len(gts) * grid_hw[0] * grid_hw[1] - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=DEV)

    agg_head, mu, sd = _train_head(feats, gts, pos_w, AGG_STEPS, AGG_LR)
    thr = _calibrate_thr(extractor, agg_head, mu, sd, fit_i, fit_m)
    if thr is None:
        print(f"{name}: 激进配置阈值标定失败,跳过", flush=True)
        return None

    orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd = agg_head, mu, sd
    det.seg_head.thr = thr
    det.seg_head.rams = None
    agg_iou, agg_hit = evaluate_with_head(det, test_defs)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams = orig

    print(f"{name:28s} baseline(300/5e-3) IoU={base_iou:.3f}/hit={base_hit:.3f}  "
          f"激进(900/1e-2) IoU={agg_iou:.3f}/hit={agg_hit:.3f}  Δ(IoU)={agg_iou-base_iou:+.3f}", flush=True)
    return dict(base=(base_iou, base_hit), agg=(agg_iou, agg_hit))


def main():
    torch.manual_seed(0)
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
        row = run_one(name, *prep())
        if row is None:
            continue
        names.append(name)
        deltas.append(row["agg"][0] - row["base"][0])
    if deltas:
        d = np.array(deltas)
        passed = (np.median(d) >= 0.005 and np.mean(d) > 0
                 and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
        print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
              f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

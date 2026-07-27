"""FocalLoss+DiceLoss替换现有BCE+pos_weight,验证能不能提升seg_head定位精度
(动机见losses.py:专门为极端类别不平衡设计,理论上比简单pos_weight缩放更适合
pcb/phone_battery这种微小缺陷)。

对比(同一fit数据、同一det、同一架构,唯一变量是损失函数):
  baseline : 生产SupervisedSegHead.fit()直接训(BCEWithLogitsLoss+pos_weight)
  +FocalDice: 完全相同的数据/架构/steps/lr,只把损失函数换成FocalLoss+DiceLoss

复用det.locate()完整下游(mask/box/IoU/框命中),只替换det.seg_head的
head/mu/sd/thr,其余(EAD/DINO/SAM等)完全不变——和今天ahl_seghead/eval_ahl.py
同一套方法论。

用法:PYTHONPATH=. python focal_dice_seghead/eval_focal_dice.py
"""
import numpy as np
import torch
import torch.nn as nn
from aoi.competition import CompetitionLargeDetector
from aoi.seg_head import map_to_boxes, merge_boxes
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to
from focal_dice_seghead.losses import FocalDiceLoss
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR, BATCH, SEED = 300, 5e-3, 8, 0


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def _train_head(feats, gts, lossf, device=DEV):
    """和生产_train_one完全相同的steps/lr/batch/优化器/种子,只换损失函数。"""
    C = feats[0].shape[0]
    mu = torch.stack([f.float().mean(dim=(1, 2)) for f in feats]).mean(0).view(1, -1, 1, 1).to(device)
    sd = (torch.stack([f.float().std(dim=(1, 2)) for f in feats]).mean(0) + 1e-6).view(1, -1, 1, 1).to(device)
    all_feats = [f.to(device) for f in feats]
    all_gts = [g.to(device) for g in gts]

    torch.manual_seed(SEED)
    lin = _ProdHead._linear_head(C).to(device)
    cnv = _ProdHead._conv_head(C).to(device)
    head = _Ensemble(lin, cnv)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    g = torch.Generator().manual_seed(SEED)
    n = len(all_feats)
    for _ in range(STEPS):
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
    base_iou, base_hit = evaluate_with_head(det, test_defs)

    extractor = det.seg_head.extractor
    with torch.no_grad():
        feats = [extractor(im) for im in fit_i]
    grid_hw = feats[0].shape[-2:]
    gts = [torch.from_numpy(_mask_to(m, grid_hw[0], grid_hw[1]).astype(np.float32)) for m in fit_m]

    fd_head, mu, sd = _train_head(feats, gts, FocalDiceLoss())
    thr = _calibrate_thr(extractor, fd_head, mu, sd, fit_i, fit_m)
    if thr is None:
        print(f"{name}: FocalDice头阈值标定失败,跳过", flush=True)
        return None

    orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd = fd_head, mu, sd
    det.seg_head.thr = thr
    det.seg_head.rams = None
    fd_iou, fd_hit = evaluate_with_head(det, test_defs)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams = orig

    print(f"{name:28s} baseline IoU={base_iou:.3f}/hit={base_hit:.3f}  "
          f"+FocalDice IoU={fd_iou:.3f}/hit={fd_hit:.3f}  Δ(IoU)={fd_iou-base_iou:+.3f}", flush=True)
    return dict(base=(base_iou, base_hit), fd=(fd_iou, fd_hit))


def main():
    torch.manual_seed(0)
    jobs = [
        ("生产:pcb(微小缺陷,最该受益)", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("生产:cable(回归检查)", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        row = run_one(name, *prep())
        if row is None:
            continue
        names.append(name)
        deltas.append(row["fd"][0] - row["base"][0])
    if deltas:
        d = np.array(deltas)
        passed = (np.median(d) >= 0.005 and np.mean(d) > 0
                 and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
        print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
              f"min(Δ)={np.min(d):+.3f}  {'✅通过' if passed else '❌不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

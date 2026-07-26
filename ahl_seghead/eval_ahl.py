"""AHL适配seg_head——真实数据验证(先小规模摸底,确认核心机制有没有戏,再决定
要不要扩大到今天GCAD那种9类目规模的充分对比)。

对比(同一fit数据,同一det,唯一变量是seg_head怎么训出来的):
  baseline : 生产SupervisedSegHead.fit()直接在全部fit缺陷上训(现状)
  +AHL     : 本文件method.py的异态代理集+协同训练流程训出的统一头

复用det.locate()的完整下游(mask/box/IoU/框命中),只替换det.seg_head的
head/mu/sd/thr这几个属性,其余(EAD/DINO/SAM等)完全不变——保证唯一变量真的
只是seg_head怎么训的。

用法:PYTHONPATH=. python ahl_seghead/eval_ahl.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.seg_head import map_to_boxes, merge_boxes
from ahl_seghead.method import fit_ahl_unified, _mask_to
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


@torch.no_grad()
def _calibrate_thr(extractor, head, mu, sd, defect_imgs, defect_masks, out_hw=(256, 256)):
    """F1最优像素阈值(和生产_calibrate_thr同公式),给换上去的AHL头单独标定。"""
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

    extractor = det.seg_head.extractor                  # 用production实际选中的特征模式(single或tmpl_diff),
                                                          # 不能硬编码_wrn_feats——否则通道数可能对不上已训好的头
    with torch.no_grad():
        normal_feats = [extractor(n) for n in normals[:40]]
        defect_feats = [extractor(d) for d in fit_i]
    grid_hw = defect_feats[0].shape[-2:]
    unified_head, mu, sd = fit_ahl_unified(normal_feats, defect_feats, fit_m, grid_hw,
                                           device=DEV, seed=0)
    if unified_head is None:
        print(f"{name}: AHL统一头训练跳过(缺陷样本太少)", flush=True)
        return None
    thr = _calibrate_thr(extractor, unified_head, mu, sd, fit_i, fit_m)
    if thr is None:
        print(f"{name}: AHL头阈值标定失败,跳过", flush=True)
        return None

    # 换上AHL统一头,复用det.locate()完整下游(mask/box/IoU/框命中全走production同一套逻辑)
    orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd = unified_head, mu, sd
    det.seg_head.thr = thr
    det.seg_head.rams = None                                # RAMS修正支是针对生产头训的,不适用于AHL头
    ahl_iou, ahl_hit = evaluate_with_head(det, test_defs)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams = orig

    print(f"{name:28s} baseline IoU={base_iou:.3f}/hit={base_hit:.3f}  "
          f"+AHL IoU={ahl_iou:.3f}/hit={ahl_hit:.3f}  Δ(IoU)={ahl_iou-base_iou:+.3f}", flush=True)
    return dict(base=(base_iou, base_hit), ahl=(ahl_iou, ahl_hit))


def main():
    torch.manual_seed(0)
    jobs = [
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
        ("生产:cable(回归检查)", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        row = run_one(name, *prep())
        if row is None:
            continue
        names.append(name)
        deltas.append(row["ahl"][0] - row["base"][0])
    if deltas:
        d = np.array(deltas)
        print(f"\n=== 摸底汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
              f"min(Δ)={np.min(d):+.3f}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

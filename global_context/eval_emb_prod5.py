"""GCAD-EmbedAE(激进配方900步/3e-3)单独(不接seg_head门控)在真正的5类生产
scorecard(hazelnut/cable/pill/pcb/phone_battery)上验证——之前只在9类LOCO/
cable/pcb上测过、不需门控直接过严格判据,这里换到真正要交的成绩单上确认是否
依然成立。用真正的OR门(复用det.locate()真实判定,不联合重标定阈值)。

用法:PYTHONPATH=. python global_context/eval_emb_prod5.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.seg_head import map_to_boxes, merge_boxes
from global_context.eval_global_branch import (
    prep_mvtec, prep_realiad, gt_boxes, box_hit, fit_global_branches, DinoCLS,
)


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


def locate_with_emb_or(det, img, z_emb_fn, thr_emb_only):
    """同combined_probe.py:base该抓的用det.locate()真实判定原样返回,只有base说
    "正常"时才额外查EmbedAE是否独立超过自己的阈值,触发的话手动补分割/框逻辑。"""
    res = det.locate(img)
    if res["is_defect"]:
        return res
    if z_emb_fn(img) < thr_emb_only:
        return res
    amap = det.segment(img)
    thr = det.pix_thr if det.pix_thr is not None else float(amap.mean() + 3 * amap.std())
    mask = (amap >= thr).astype(np.uint8)
    native = img if img.dim() == 3 else img[0]
    if det.roi_zoom:
        mask = det._zoom_refine(native, mask, thr)
    if det.sam is not None:
        mask = det.sam.refine(native, mask, amap=amap)
    if det.crop_cascade is not None:
        mask = det.crop_cascade.refine(det, native, mask, mask.shape)
    if det.comp_graph is not None:
        mask = det.comp_graph.refine(det, img, mask)
    boxes = map_to_boxes(mask.astype(np.float32), 0.5, min_area_frac=0.0002, close=0)
    res["is_defect"] = True
    res["anomaly_map"] = amap
    res["mask"] = mask
    res["boxes"] = merge_boxes(boxes, getattr(det, "box_merge_d", 0))
    return res


def combo_eval(det, z_emb_fn, thr_emb_only, test_defs):
    ious, hits = [], []
    for img, gt in test_defs:
        o = locate_with_emb_or(det, img, z_emb_fn, thr_emb_only)
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
    print(f"{name}: lat_probe_ms={getattr(det, 'lat_probe_ms', None)} "
          f"lat_trimmed={getattr(det, 'lat_trimmed', None)}", flush=True)
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)
    base_iou, base_hit = baseline_eval(det, test_defs)

    dino = DinoCLS(device="cuda" if torch.cuda.is_available() else "cpu")
    fns = fit_global_branches(det, normals, fit_i, dino, ae_steps=900, ae_lr=3e-3)
    z_emb_fn, thr_emb_only = fns["emb_only"]
    combo_iou, combo_hit = combo_eval(det, z_emb_fn, thr_emb_only, test_defs)

    print(f"{name:20s} baseline IoU={base_iou:.3f}/hit={base_hit:.3f}  "
          f"+GCAD-EmbedAE(单独) IoU={combo_iou:.3f}/hit={combo_hit:.3f}  "
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
        names.append(name)
        deltas.append(row["combo"][0] - row["base"][0])
    d = np.array(deltas)
    passed = (np.median(d) >= 0.005 and np.mean(d) > 0
             and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
    print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
          f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
    print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

"""验证优先级3:独立crop-head级联(候选来自ECC模板残差,非全图粗分割头;独立训练/独立
mu-sd-阈值)能否救回全图512下采样抹掉的微小缺陷,用户点名"最有希望继续救PCB"。
用ViSA pcb1(原生1404×1070,有正常/异常图+像素级mask,非MDPI/学术造假来源)。

两路对比(同一次fit,fit_fewshot(crop_cascade=True)内部OOF留出验证net gain>0.01才enabled):
  base = det.crop_cascade临时置空 → 走纯全图(含SAM)结果
  cc   = det.crop_cascade保持fit()学到的状态(未通过OOF门槛则等于base,零回退)

用法:PYTHONPATH=. python scripts/run_crop_cascade_ab.py
"""
import glob
import random
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from eval.mvtec import _load_img_native

HW = (256, 256)


def _read_mask(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def prep_visa_pcb(cat, n_norm=100, n_fit=30, n_test=40):
    root = Path("data/visa") / cat / "Data"
    norm_p = sorted(glob.glob(str(root / "Images/Normal/*.JPG")))
    anom_p = sorted(glob.glob(str(root / "Images/Anomaly/*.JPG")))
    random.Random(0).shuffle(norm_p)
    random.Random(1).shuffle(anom_p)
    normals = [_load_img_native(p) for p in norm_p[:n_norm]]
    mask_p = lambda p: root / "Masks/Anomaly" / (Path(p).stem + ".png")
    fit_p = anom_p[:n_fit]
    fit_i = [_load_img_native(p) for p in fit_p]
    fit_m = [_read_mask(mask_p(p), HW) for p in fit_p]
    test_p = anom_p[n_fit:n_fit + n_test]
    test_defs = [(_load_img_native(p), _read_mask(mask_p(p), HW)) for p in test_p]
    return normals, fit_i, fit_m, test_defs


def iou(pred, gt):
    p = pred.astype(bool)
    TP = int((p & (gt == 1)).sum()); FP = int((p & (gt == 0)).sum()); FN = int((~p & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def eval_pure_iou(det, test_defs, use_cc):
    orig_cc = det.crop_cascade
    if not use_cc:
        det.crop_cascade = None
    ious, lats = [], []
    for img, gt in test_defs:
        t0 = time.perf_counter()
        o = det.locate(img)
        lats.append((time.perf_counter() - t0) * 1000)
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        ious.append(iou(pix, gt))
    det.crop_cascade = orig_cc
    return float(np.mean(ious)), float(np.mean(lats))


def main():
    torch.manual_seed(0)
    cats = ["pcb1", "pcb2", "pcb3", "pcb4"]
    results = {}
    for cat in cats:
        normals, fit_i, fit_m, test_defs = prep_visa_pcb(cat)
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1, crop_cascade=True)
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        cc = det.crop_cascade
        enabled = cc is not None
        gain_fit = getattr(cc, "gain", None) if enabled else None
        base_iou, base_lat = eval_pure_iou(det, test_defs, use_cc=False)
        cc_iou, cc_lat = eval_pure_iou(det, test_defs, use_cc=True)
        results[cat] = (base_iou, cc_iou)
        print(f"{cat:8s} enabled={enabled} fit留出gain={gain_fit} | "
              f"base(全图+SAM)={base_iou:.3f} cc(+crop级联)={cc_iou:.3f} Δ={cc_iou-base_iou:+.3f} | "
              f"lat base={base_lat:.0f}ms cc={cc_lat:.0f}ms", flush=True)
    print("\n=== 均值 ===")
    b = np.mean([v[0] for v in results.values()])
    c = np.mean([v[1] for v in results.values()])
    print(f"base={b:.3f}  cc={c:.3f}  Δ={c-b:+.3f}", flush=True)


if __name__ == "__main__":
    main()

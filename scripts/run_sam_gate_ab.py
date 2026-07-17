"""验证优先级2:SAM受控精化(per-region OOF门控)能否修复Pareto扫描发现的问题——
SAM在walnuts/fruit_jelly/sheet_metal(非对抗性正常类)上系统性拖累纯定位IoU
(0.266<0.414 / 0.556<0.701 / 0.486<0.679)。

三路对比(同一fit,post-hoc切换):
  raw       = 不用SAM(分割头原始输出)
  old_sam   = 旧"整图4倍面积"heuristic,总是接受(gate强制None)
  new_sam   = 新逐区域OOF标定门控(gate=calibrate()学到的规则,若OOF无净增益会自动=None即等于old_sam)

用法:PYTHONPATH=. python scripts/run_sam_gate_ab.py
"""
import glob
import random
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import _read

AD2 = __import__("pathlib").Path("data/mvtec_ad_2")
HW = (256, 256)


def prep_ad2(cat, n_norm=100, n_fit=30):
    root = AD2 / cat
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    gt = root / "test_public/ground_truth/bad"
    m = lambda p: _read(str(gt / (__import__("pathlib").Path(p).stem + "_mask.png")), HW)
    fit_i = [load_fast(p) for p in bad[:n_fit]]; fit_m = [m(p) for p in bad[:n_fit]]
    test_defs = [(load_fast(p), m(p)) for p in bad[n_fit:n_fit + 40]]
    return normals, fit_i, fit_m, test_defs


def iou(pred, gt):
    p = pred.astype(bool)
    TP = int((p & (gt == 1)).sum()); FP = int((p & (gt == 0)).sum()); FN = int((~p & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def eval_pure_iou(det, test_defs, sam_mode):
    """sam_mode: 'raw' | 'old' | 'new'"""
    orig_sam = det.sam
    orig_gate = det.sam.gate if det.sam is not None else None
    if sam_mode == "raw":
        det.sam = None
    elif sam_mode == "old":
        det.sam.gate = "uncalibrated"           # 强制退回旧heuristic(总是接受,除非4倍面积爆炸)
    # 'new' 保持calibrate()学到的gate不动
    ious = []
    for img, gt in test_defs:
        o = det.locate(img)
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        ious.append(iou(pix, gt))
    det.sam = orig_sam
    if det.sam is not None:
        det.sam.gate = orig_gate
    return float(np.mean(ious))


def main():
    torch.manual_seed(0)
    cats = ["sheet_metal", "walnuts", "fruit_jelly"]
    results = {}
    for cat in cats:
        normals, fit_i, fit_m, test_defs = prep_ad2(cat)
        det = CompetitionLargeDetector(train_steps=3000, ead_students=1)   # 减步数快跑,只测精度不测延时
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        gate_info = getattr(det.sam, "gate", None) if det.sam else None
        gain = getattr(det.sam, "calib_gain", None) if det.sam else None
        raw_iou = eval_pure_iou(det, test_defs, "raw")
        old_iou = eval_pure_iou(det, test_defs, "old")
        new_iou = eval_pure_iou(det, test_defs, "new")
        results[cat] = (raw_iou, old_iou, new_iou)
        print(f"{cat:14s} raw(无SAM)={raw_iou:.3f}  old_sam(旧总接受)={old_iou:.3f}  "
              f"new_sam(OOF门控)={new_iou:.3f}  gate={gate_info}  OOF标定gain={gain}", flush=True)
    print("\n=== 均值 ===")
    r = np.mean([v[0] for v in results.values()])
    o = np.mean([v[1] for v in results.values()])
    n = np.mean([v[2] for v in results.values()])
    print(f"raw={r:.3f}  old_sam={o:.3f}  new_sam={n:.3f}  "
          f"Δ(new-raw)={n-r:+.3f}  Δ(new-old)={n-o:+.3f}", flush=True)


if __name__ == "__main__":
    main()

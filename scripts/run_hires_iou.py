"""提定位精度:分辨率 × 阈值对严格 IoU 的影响(全程用 IoU/F1,不用 AUROC)。
诊断:小缺陷IoU崩=异常图分辨率太低(320输入→残差72²,对不准小缺陷边界)。
对比输入分辨率 320/512/768 在弱电子件上的 best-IoU 与校准-IoU。用法:python scripts/run_hires_iou.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.efficientad import EfficientADDetector
from aoi.seg_head import SupervisedSegHead
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def best_and_cal(S, L, cal_thr):
    s = np.asarray(S); l = np.asarray(L)
    order = np.argsort(-s); ls = l[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(np.argmax(f1))
    best_iou = tp[bi] / max(tp[bi] + fp[bi] + (P - tp[bi]), 1)
    pred = s >= cal_thr
    TP = int((pred & (l == 1)).sum()); FP = int((pred & (l == 0)).sum()); FN = int((~pred & (l == 1)).sum())
    cal_iou = TP / max(TP + FP + FN, 1)
    return float(best_iou), float(cal_iou)


def prep_realiad(cat, size):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], size) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    d_imgs = [_load_img(R / x["image_path"], size) for x in ng[:30]]
    hw = (size, size)
    d_masks = [_read(R / x["mask_path"], hw) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], size), _read(R / x["mask_path"], hw)) for x in ng[30:70]]
    tests += [(_load_img(R / x["image_path"], size), np.zeros(hw, np.uint8))
              for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, d_imgs, d_masks, tests, hw


def run(cat, size):
    normals, d_imgs, d_masks, tests, hw = prep_realiad(cat, size)
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(normals, None)
    head = SupervisedSegHead(device=DEV); head.fit(det, d_imgs, d_masks, normals[:30])
    nv = np.concatenate([head.map(det, n, hw).ravel() for n in normals[:15]])
    cal = float(np.quantile(nv, 0.995))
    S, L = [], []
    for img, mp in tests:
        S.append(head.map(det, img, hw).ravel()); L.append(mp.ravel())
    bi, ci = best_and_cal(np.concatenate(S), np.concatenate(L), cal)
    # 残差源分辨率
    res_h = det.residual_map_large(tests[0][0]).shape[1]
    return bi, ci, res_h


def main():
    torch.manual_seed(0)
    print("=== 分辨率对严格定位IoU的影响(弱电子件)===")
    for cat in ["pcb", "phone_battery"]:
        print(f"\n[{cat}]")
        for size in [320, 512, 768]:
            bi, ci, rh = run(cat, size)
            print(f"  输入{size}(残差{rh}²): 最佳IoU={bi:.3f}  校准IoU={ci:.3f}", flush=True)


if __name__ == "__main__":
    main()

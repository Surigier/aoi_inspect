"""提IoU杠杆①:阈值标定——正常分位数 vs 用30张缺陷掩膜的F1最优阈值(监督)。
诊断:校准IoU普遍只有best-IoU的1/3,阈值是最大漏损。赛题给了缺陷掩膜→应监督标阈值。
全程IoU。用法:python scripts/run_thresh_iou.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.efficientad import EfficientADDetector
from aoi.seg_head import SupervisedSegHead
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def iou_at(s, l, thr):
    pred = s >= thr
    TP = int((pred & (l == 1)).sum()); FP = int((pred & (l == 0)).sum()); FN = int((~pred & (l == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def f1_opt_thr(s, l):
    """在(缺陷)像素上找最大化F1的阈值。"""
    order = np.argsort(-s); ls = l[order]; ss = s[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(ss[int(np.argmax(f1))])


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "test/good/*.png")))]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit = [(_load_img(p, 320), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[:k]]
    tests = [(_load_img(p, 320), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    tests += [(g, np.zeros(HW, np.uint8)) for g in goods[:len(df) - k]]
    return normals, fit, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 320) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit = [(_load_img(R / x["image_path"], 320), _read(R / x["mask_path"], HW)) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 320), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    tests += [(_load_img(R / x["image_path"], 320), np.zeros(HW, np.uint8))
              for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, fit, tests


def run(name, normals, fit, tests):
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(normals, None)
    head = SupervisedSegHead(device=DEV)
    head.fit(det, [f[0] for f in fit], [f[1] for f in fit], normals[:30])
    # 阈值A:正常p99.5
    nv = np.concatenate([head.map(det, n, HW).ravel() for n in normals[:15]])
    thrA = float(np.quantile(nv, 0.995))
    # 阈值B:fit缺陷上F1最优(监督)
    fs = np.concatenate([head.map(det, im, HW).ravel() for im, _ in fit])
    fl = np.concatenate([m.ravel() for _, m in fit])
    thrB = f1_opt_thr(fs, fl)
    # 测试集评
    ts = np.concatenate([head.map(det, im, HW).ravel() for im, _ in tests])
    tl = np.concatenate([m.ravel() for _, m in tests])
    iA = iou_at(ts, tl, thrA); iB = iou_at(ts, tl, thrB)
    print(f"{name:22s} 正常分位阈值IoU={iA:.3f}  监督F1阈值IoU={iB:.3f}  Δ={iB-iA:+.3f}", flush=True)
    return iA, iB


def main():
    torch.manual_seed(0)
    print("=== 阈值标定对IoU的影响(正常分位 vs 监督掩膜F1)===")
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("电子 realiad/pcb", lambda: prep_realiad("pcb")),
        ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    A, B = [], []
    for name, prep in jobs:
        a, b = run(name, *prep()); A.append(a); B.append(b)
    print(f"\n均值: 正常分位={np.mean(A):.3f}  监督阈值={np.mean(B):.3f}  Δ={np.mean(B)-np.mean(A):+.3f}")


if __name__ == "__main__":
    main()

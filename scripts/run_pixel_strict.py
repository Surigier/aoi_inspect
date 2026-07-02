"""严格像素级定位指标(评委真实口径):AUROC(参考) + AP + 最佳F1/IoU + 校准阈值F1/IoU。
赛题准确率按分割/检测定位评,IoU/F1比AUROC严得多(小缺陷背景压倒→AUROC虚高)。
用监督分割头(生产定位)在真实数据上跑。用法:python scripts/run_pixel_strict.py
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
from eval.protocol import image_auroc
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
EVAL_HW = (256, 256)


def _mmask(cat, folder, stem, hw):
    p = GT / cat / "ground_truth" / folder / (stem + "_mask.png")
    return _read(p, hw)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def strict_metrics(scores, labels, cal_thr=None):
    """scores,labels: 1D pooled 像素。返回 AUROC/AP/最佳F1&IoU/校准F1&IoU。"""
    s = np.asarray(scores); l = np.asarray(labels)
    au = image_auroc(s, l)
    order = np.argsort(-s); ls = l[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls)
    P = int(ls.sum())                                   # 正样本(缺陷像素)总数
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:])) if len(rec) > 1 else 0.0
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(np.argmax(f1))
    best_f1 = float(f1[bi]); best_iou = float(tp[bi] / max(tp[bi] + fp[bi] + (P - tp[bi]), 1))
    out = dict(auroc=au, ap=ap, bestF1=best_f1, bestIoU=best_iou)
    if cal_thr is not None:
        pred = s >= cal_thr
        TP = int((pred & (l == 1)).sum()); FP = int((pred & (l == 0)).sum()); FN = int((~pred & (l == 1)).sum())
        pr = TP / max(TP + FP, 1); rc = TP / max(TP + FN, 1)
        out["calF1"] = 2 * pr * rc / max(pr + rc, 1e-9); out["calIoU"] = TP / max(TP + FP + FN, 1)
    return out


def eval_cat(name, det, head, tests, cal_thr):
    S, L = [], []
    for img, mp in tests:
        amap = head.map(det, img, EVAL_HW)
        S.append(amap.ravel()); L.append(mp.ravel())
    m = strict_metrics(np.concatenate(S), np.concatenate(L), cal_thr)
    print(f"{name:20s} AUROC={m['auroc']:.3f} | AP={m['ap']:.3f} 最佳F1={m['bestF1']:.3f} IoU={m['bestIoU']:.3f} "
          f"| 校准F1={m.get('calF1',0):.3f} IoU={m.get('calIoU',0):.3f}", flush=True)
    return m


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "test/good/*.png")))]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df)
    k = max(5, len(df) // 3)
    d_imgs = [_load_img(p, 320) for p, _ in df[:k]]
    d_masks = [_mmask(cat, fo, Path(p).stem, EVAL_HW) for p, fo in df[:k]]
    tests = [(_load_img(p, 320), _mmask(cat, fo, Path(p).stem, EVAL_HW)) for p, fo in df[k:]]
    tests += [(g, np.zeros(EVAL_HW, np.uint8)) for g in goods[:len(df) - k]]
    return normals, d_imgs, d_masks, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 320) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    d_imgs = [_load_img(R / x["image_path"], 320) for x in ng[:30]]
    d_masks = [_read(R / x["mask_path"], EVAL_HW) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 320), _read(R / x["mask_path"], EVAL_HW)) for x in ng[30:70]]
    tests += [(_load_img(R / x["image_path"], 320), np.zeros(EVAL_HW, np.uint8))
              for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, d_imgs, d_masks, tests


def run(name, normals, d_imgs, d_masks, tests):
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(normals, None)
    head = SupervisedSegHead(device=DEV); head.fit(det, d_imgs, d_masks, normals[:30])
    # 校准阈值:正常图像素图 p99.5
    nv = np.concatenate([head.map(det, n, EVAL_HW).ravel() for n in normals[:15]])
    cal = float(np.quantile(nv, 0.995))
    return eval_cat(name, det, head, tests, cal)


def main():
    torch.manual_seed(0)
    print("=== 严格像素级定位(AP/F1/IoU=评委真口径,AUROC仅参考)===")
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("尺寸 metal_nut", lambda: prep_mvtec("metal_nut", ["flip"])),
        ("电子 realiad/pcb", lambda: prep_realiad("pcb")),
        ("电池 realiad/phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    ms = []
    for name, prep in jobs:
        ms.append(run(name, *prep()))
    print("\n均值: AP={:.3f} 最佳F1={:.3f} IoU={:.3f} | 校准F1={:.3f} IoU={:.3f}".format(
        np.mean([m['ap'] for m in ms]), np.mean([m['bestF1'] for m in ms]),
        np.mean([m['bestIoU'] for m in ms]), np.mean([m.get('calF1', 0) for m in ms]),
        np.mean([m.get('calIoU', 0) for m in ms])))


if __name__ == "__main__":
    main()

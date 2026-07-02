"""生产链路严格IoU验收:CompetitionLargeDetector(WRN50定位特征+监督F1阈值)。
对比升级前基线(EAD残差+正常分位阈值:校准IoU 0.09-0.40)。全程IoU。
注:定位走WRN50与EAD训练无关,train_steps低速跑不影响IoU结论。
用法:python scripts/run_prod_iou.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def iou_metrics(S, L, cal_thr):
    s = np.concatenate(S); l = np.concatenate(L)
    order = np.argsort(-s); ls = l[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(np.argmax(f1))
    best = float(tp[bi] / max(tp[bi] + fp[bi] + (P - tp[bi]), 1))
    pred = s >= cal_thr
    TP = int((pred & (l == 1)).sum()); FP = int((pred & (l == 0)).sum()); FN = int((~pred & (l == 1)).sum())
    return best, TP / max(TP + FP + FN, 1)


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "test/good/*.png")))]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit_i = [_load_img(p, 320) for p, _ in df[:k]]
    fit_m = [_read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW) for p, fo in df[:k]]
    tests = [(_load_img(p, 320), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    tests += [(g, np.zeros(HW, np.uint8)) for g in goods[:len(df) - k]]
    return normals, fit_i, fit_m, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 320) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit_i = [_load_img(R / x["image_path"], 320) for x in ng[:30]]
    fit_m = [_read(R / x["mask_path"], HW) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 320), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    tests += [(_load_img(R / x["image_path"], 320), np.zeros(HW, np.uint8))
              for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, fit_i, fit_m, tests


def main():
    torch.manual_seed(0)
    print("=== 生产链路严格IoU(WRN50定位 + 监督F1阈值)===")
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("电子 pcb", lambda: prep_realiad("pcb")),
        ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    B, C = [], []
    for name, prep in jobs:
        normals, fit_i, fit_m, tests = prep()
        det = CompetitionLargeDetector(train_steps=200)     # EAD步数不影响WRN50定位
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        S = [det.segment(im).ravel() for im, _ in tests]
        L = [m.ravel() for _, m in tests]
        best, cal = iou_metrics(S, L, det.pix_thr)
        B.append(best); C.append(cal)
        print(f"{name:20s} best-IoU={best:.3f}  校准IoU={cal:.3f}(阈值={det.pix_thr:.2f})", flush=True)
    print(f"\n均值: best-IoU={np.mean(B):.3f}  校准IoU={np.mean(C):.3f}")
    print("(升级前基线: best均值0.263 / 校准均值~0.170)")


if __name__ == "__main__":
    main()

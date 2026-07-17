"""2500²大图一致性验收(回归竞赛题目):
① 延时:locate()(=predict+segment+boxes,评分要定位→按这个计时)在真实大图上是否<200ms@2060
② 严格IoU:AD2真实大图(2100×1520,有GT掩膜)走生产locate路径的校准IoU
③ 框输出:数量/合理性
用法:python scripts/run_large_consistency.py
"""
import glob
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = (256, 256)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def iou_metrics(S, L, thr):
    s = np.concatenate(S); l = np.concatenate(L)
    order = np.argsort(-s); ls = l[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(np.argmax(f1))
    best = float(tp[bi] / max(tp[bi] + fp[bi] + (P - tp[bi]), 1))
    pred = s >= thr
    TP = int((pred & (l == 1)).sum()); FP = int((pred & (l == 0)).sum()); FN = int((~pred & (l == 1)).sum())
    return best, TP / max(TP + FP + FN, 1)


def bench(fn, reps=8):
    torch.cuda.synchronize(); fn(); fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps


def part1_latency():
    print("=== ① 大图延时:predict vs locate(评分口径含定位)===")
    imgs = sorted(glob.glob("data/_dl/pku_pcb/images/*.jpg"))
    det = CompetitionLargeDetector(train_steps=200)
    seed = [load_fast(p) for p in imgs[:20]]
    det.fit_fewshot(seed[:15], seed[15:20])
    p = imgs[25]
    W, H = Image.open(p).size
    t_pred = bench(lambda: det.predict(load_fast(p)))
    t_loc = bench(lambda: det.locate(load_fast(p)))
    d_seg = t_loc - t_pred
    # CPU解码不×1.7,GPU部分×1.7:load≈29ms(此前实测)
    t_load = 29.0
    est_pred = t_load + (t_pred - t_load) * 1.7
    est_loc = t_load + (t_loc - t_load) * 1.7
    print(f"真实PCB {W}x{H}: predict={t_pred:.0f}ms  locate={t_loc:.0f}ms(定位开销+{d_seg:.0f}ms)")
    print(f"  →2060估: predict~{est_pred:.0f}ms  locate~{est_loc:.0f}ms {'✅<200' if est_loc < 200 else '⚠️超'}")


def part2_ad2_iou(cat="sheet_metal"):
    print(f"\n=== ② AD2真实大图({cat})生产locate路径严格IoU ===")
    root = Path(f"data/mvtec_ad_2/{cat}")
    gn = sorted(glob.glob(str(root / "train/good/*.png")))
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png")))
    good_t = sorted(glob.glob(str(root / "test_public/good/*.png")))
    random.Random(0).shuffle(bad)
    normals = [load_fast(p) for p in gn[:100]]
    fit_b = bad[:30]; test_b = bad[30:]
    def mpath(p):
        return str(root / "test_public/ground_truth/bad" / (Path(p).stem + "_mask.png"))
    fit_i = [load_fast(p) for p in fit_b]
    fit_m = [_read(mpath(p), HW) for p in fit_b]
    det = CompetitionLargeDetector(train_steps=200)
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    S, L, nb = [], [], []
    for p in test_b:
        img = load_fast(p)
        o = det.locate(img)
        S.append(det.segment(img).ravel()); L.append(_read(mpath(p), HW).ravel())
        nb.append(len(o["boxes"]))
    for p in good_t[:40]:
        img = load_fast(p)
        o = det.locate(img)
        S.append(det.segment(img).ravel()); L.append(np.zeros(HW, np.uint8).ravel())
        nb.append(len(o["boxes"]))
    best, cal = iou_metrics(S, L, det.pix_thr)
    print(f"{cat}: best-IoU={best:.3f}  校准IoU={cal:.3f}  框数/图 中位={int(np.median(nb))} 最大={max(nb)}")


def main():
    torch.manual_seed(0)
    part1_latency()
    part2_ad2_iou()


if __name__ == "__main__":
    main()

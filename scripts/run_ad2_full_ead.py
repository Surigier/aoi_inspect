"""AD2 全8类 EfficientAD 统一检测器成绩单:python scripts/run_ad2_full_ead.py
真100张fit,报每类 EAD检测AUROC + 二值准确率 + 延时,并给均值。"""
import glob
import time
import random
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.fusion import auroc
from eval.mvtec import _load_img_native

ROOT = "data/mvtec_ad_2"
CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
CAP = 80


def acc(scores, labels, thr):
    return sum((s >= thr) == bool(l) for s, l in zip(scores, labels)) / len(labels)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aus, acs = [], []
    for cat in CATS:
        g = lambda s: sorted(glob.glob(f"{ROOT}/{cat}/{s}"))
        train, bad, good = g("train/good/*.png"), g("test_public/bad/*.png"), g("test_public/good/*.png")
        if not train or len(bad) < 6:
            print(f"{cat}: 数据不足"); continue
        rng = random.Random(0)
        rng.shuffle(train); rng.shuffle(bad); rng.shuffle(good)
        fn = [_load_img_native(p) for p in train[:100]]
        fd = [_load_img_native(p) for p in bad[:30]]
        tp = good + bad[30:]
        tl = [0] * len(good) + [1] * len(bad[30:])
        idx = list(range(len(tp))); rng.shuffle(idx); idx = idx[:CAP]
        tp = [tp[i] for i in idx]; tl = [tl[i] for i in idx]
        test = [_load_img_native(p) for p in tp]
        t = time.time()
        det = CompetitionLargeDetector(device=dev, train_steps=10000)
        det.fit_fewshot(fn, fd)
        ds = [det.branches[0].score(im) for im in test]
        det.predict(test[0])                              # 预热
        t0 = time.perf_counter()
        for i in range(4):
            det.predict(test[i])
        lat = (time.perf_counter() - t0) / 4 * 1000
        au = auroc(ds, tl); ac = acc(ds, tl, det.threshold)
        aus.append(au); acs.append(ac)
        print(f"{cat:12s} AUROC={au:.3f} acc={ac:.3f} 延时={lat:.0f}ms ({time.time()-t:.0f}s)", flush=True)
    if aus:
        print(f"\n均值: AUROC={sum(aus)/len(aus):.3f} acc={sum(acs)/len(acs):.3f} (n={len(aus)})", flush=True)


if __name__ == "__main__":
    main()

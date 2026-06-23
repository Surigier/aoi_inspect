"""赛场大图统一检测器验证:python scripts/run_competition_bench.py [类别]
真100张 fit,报融合AUROC + 各分支AUROC + 权重 + 延时,确认融合不掉EAD精度且补类型覆盖。"""
import sys
import glob
import time
import random
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.fusion import auroc
from eval.mvtec import _load_img_native

ROOT = "data/mvtec_ad_2"


def main():
    cat = sys.argv[1] if len(sys.argv) > 1 else "sheet_metal"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = lambda s: sorted(glob.glob(f"{ROOT}/{cat}/{s}"))
    train, bad, good = g("train/good/*.png"), g("test_public/bad/*.png"), g("test_public/good/*.png")
    rng = random.Random(0)
    rng.shuffle(train); rng.shuffle(bad); rng.shuffle(good)
    fn = [_load_img_native(p) for p in train[:100]]
    fd = [_load_img_native(p) for p in bad[:30]]
    tp = good + bad[30:]
    tl = [0] * len(good) + [1] * len(bad[30:])
    idx = list(range(len(tp))); rng.shuffle(idx); idx = idx[:80]
    tp = [tp[i] for i in idx]; tl = [tl[i] for i in idx]
    test = [_load_img_native(p) for p in tp]

    t = time.time()
    det = CompetitionLargeDetector(device=dev, train_steps=10000)
    det.fit_fewshot(fn, fd)
    names = ["EAD核心", "色彩", "尺寸", "结构"]
    print(f"{cat} 训练完成 ({time.time()-t:.0f}s)", flush=True)
    print("权重:", {n: round(w, 3) for n, w in zip(names, det.weights)}, flush=True)
    # 各分支单独 AUROC
    for j, b in enumerate(det.branches):
        bs = [b.score(im) for im in test]
        print(f"  {names[j]:6s} 单独AUROC={auroc(bs, tl):.3f}", flush=True)
    fused = [det._fuse([b.score(im) for b in det.branches]) for im in test]
    det.predict(test[0])
    t0 = time.perf_counter()
    for i in range(4):
        det.predict(test[i])
    lat = (time.perf_counter() - t0) / 4 * 1000
    print(f"融合AUROC={auroc(fused, tl):.3f} 延时={lat:.0f}ms (2060估~{lat*1.7:.0f}ms) | EAD单独基线0.840", flush=True)


if __name__ == "__main__":
    main()

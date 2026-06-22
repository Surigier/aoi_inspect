"""混合检测器 AD2 验证:python scripts/run_hybrid_bench.py [类别...]
一次 fit,同时报 全局(整图512)/ 局部(分块)/ 融合 三者 AUROC + 融合acc + 延时。
目标:融合稳过 baseline(纯全局 0.690),逼近 oracle 0.731。"""
import sys
import glob
import time
import random
import torch
from aoi.backbone import Backbone
from aoi.hybrid import HybridDetector
from aoi.fusion import auroc
from eval.mvtec import _load_img_native

ROOT = "data/mvtec_ad_2"
CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
CAP_TEST = 120


def acc(scores, labels, thr):
    return sum((s >= thr) == bool(l) for s, l in zip(scores, labels)) / len(labels)


def run_cat(cat, bb_big, bb_small, agg):
    g = lambda sub: sorted(glob.glob(f"{ROOT}/{cat}/{sub}"))
    train, bad, good = g("train/good/*.png"), g("test_public/bad/*.png"), g("test_public/good/*.png")
    if not train or len(bad) < 6 or not good:
        print(f"{cat}: 数据不足"); return
    rng = random.Random(0)
    rng.shuffle(train); rng.shuffle(bad); rng.shuffle(good)
    fn = [_load_img_native(p) for p in train[:100]]
    fd = [_load_img_native(p) for p in bad[:30]]
    test_paths = good + bad[30:]
    test_lab = [0] * len(good) + [1] * len(bad[30:])
    idx = list(range(len(test_paths))); rng.shuffle(idx); idx = idx[:CAP_TEST]
    test_paths = [test_paths[i] for i in idx]; test_lab = [test_lab[i] for i in idx]
    test = [_load_img_native(p) for p in test_paths]

    t = time.time()
    det = HybridDetector(bb_big, bb_small,
                         local_kw=dict(tile=512, stride=512, coreset_ratio=0.01,
                                       feat_grid=32, position_aware=True))
    det.fit_fewshot(fn, fd)
    gs = [det.g.score(im) for im in test]
    ls = [det.l.score(im) for im in test]
    fs = [det._fuse_vec([g_, l_]) for g_, l_ in zip(gs, ls)]
    det._fused(test[0])                       # 预热
    t0 = time.perf_counter()
    for i in range(3):
        det._fused(test[i])
    lat = (time.perf_counter() - t0) / 3 * 1000
    gau, lau, fau = auroc(gs, test_lab), auroc(ls, test_lab), auroc(fs, test_lab)
    fac = acc(fs, test_lab, det.threshold)
    w = det.weights
    print(f"{cat:12s} 全局={gau:.3f} 局部={lau:.3f} 融合={fau:.3f} acc={fac:.3f} "
          f"lat={lat:.0f}ms w=[{w[0]:.2f},{w[1]:.2f}]  ({time.time()-t:.0f}s)", flush=True)
    agg.append((gau, lau, fau))


def main():
    cats = sys.argv[1:] or CATS
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb_big = Backbone(pretrained=True, device=dev)
    bb_small = Backbone(name="resnet18", layers=(2, 3), pretrained=True, device=dev)
    agg = []
    for c in cats:
        run_cat(c, bb_big, bb_small, agg)
    if agg:
        n = len(agg)
        print(f"\n平均: 全局={sum(a[0] for a in agg)/n:.3f} "
              f"局部={sum(a[1] for a in agg)/n:.3f} 融合={sum(a[2] for a in agg)/n:.3f}")


if __name__ == "__main__":
    main()

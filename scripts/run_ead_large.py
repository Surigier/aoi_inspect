"""大图 EfficientAD 验证:python scripts/run_ead_large.py [类别...]
AD2 真实高分辨率(2232×1024)上跑 TiledEfficientAD(真100张正常 fit),
报 AUROC + 单图延时,对比之前 PatchCore-分块的数。看换地基后大图效果与延时。"""
import sys
import glob
import time
import random
import torch
from aoi.tiled_efficientad import TiledEfficientAD
from aoi.fusion import auroc
from eval.mvtec import _load_img_native

ROOT = "data/mvtec_ad_2"
# 之前 PatchCore-分块(位置感知)的数,供对比
PC = {"can": 0.599, "sheet_metal": 0.772, "walnuts": 0.769, "fabric": 0.530}
CATS = ["can", "sheet_metal"]
CAP = 80


def acc(scores, labels, thr):
    return sum((s >= thr) == bool(l) for s, l in zip(scores, labels)) / len(labels)


def run(cat, dev):
    g = lambda s: sorted(glob.glob(f"{ROOT}/{cat}/{s}"))
    train, bad, good = g("train/good/*.png"), g("test_public/bad/*.png"), g("test_public/good/*.png")
    if not train or len(bad) < 6:
        print(f"{cat}: 数据不足"); return
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
    det = TiledEfficientAD(model_size="small", device=dev, train_steps=6000,
                           tile=512, stride=512)
    det.fit_fewshot(fn, fd)
    fit_s = time.time() - t
    scores = [det._image_score(im) for im in test]
    det.predict(test[0])
    lat = sum(det.predict(test[i])["latency_ms"] for i in range(min(4, len(test)))) / min(4, len(test))
    au = auroc(scores, tl); ac = acc(scores, tl, det.threshold)
    print(f"{cat:12s} EfficientAD-分块 AUROC={au:.3f} acc={ac:.3f} 延时={lat:.0f}ms "
          f"fit={fit_s:.0f}s | PatchCore-分块={PC.get(cat,'?')}", flush=True)


def main():
    cats = sys.argv[1:] or CATS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备={dev};AD2真实2232×1024,真100张fit", flush=True)
    for c in cats:
        run(c, dev)


if __name__ == "__main__":
    main()

"""MVTec AD 2 真实高分辨率验证:python scripts/run_ad2_bench.py [类别...]
官方协议:train/good 取100正常 + test_public/bad 取30缺陷 → fit → 测 test_public 剩余。
对比三法在真实 2232×1024 高分辨率小缺陷上的 AUROC/准确率/延时:
  baseline 整图resize512  |  tiled 共享库  |  tiled 位置感知库
证明分块在真实数据上也救回小缺陷,并定 position_aware 取舍。"""
import sys
import glob
import time
import random
import torch
from aoi.backbone import Backbone
from aoi.tiled import TiledFewShotDetector
from aoi.branches.texture_ad import TextureADBranch
from aoi.fusion import auroc
from eval.mvtec import _load_img, _load_img_native

ROOT = "data/mvtec_ad_2"
CATS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
CAP_TEST = 120          # 每类测试上限(估指标够用、加速)


def acc(scores, labels, thr):
    return sum((s >= thr) == bool(l) for s, l in zip(scores, labels)) / len(labels)


def run_cat(cat, bb_big, bb_small):
    g = lambda sub: sorted(glob.glob(f"{ROOT}/{cat}/{sub}"))
    train = g("train/good/*.png")
    bad = g("test_public/bad/*.png")
    good = g("test_public/good/*.png")
    if not train or len(bad) < 6 or not good:
        print(f"{cat}: 数据不足 (train={len(train)} bad={len(bad)} good={len(good)})")
        return
    rng = random.Random(0)
    rng.shuffle(train); rng.shuffle(bad); rng.shuffle(good)
    fit_n_paths = train[:100]
    fit_d_paths = bad[:30]
    test_paths = good + bad[30:]
    test_lab = [0] * len(good) + [1] * len(bad[30:])
    idx = list(range(len(test_paths))); rng.shuffle(idx); idx = idx[:CAP_TEST]
    test_paths = [test_paths[i] for i in idx]; test_lab = [test_lab[i] for i in idx]

    # --- baseline: 整图 resize 512(老做法,强制正方形)---
    t = time.time()
    fn512 = [_load_img(p, 512) for p in fit_n_paths]
    bad_ad = TextureADBranch(backbone=bb_big, coreset_ratio=0.05)
    bad_ad.fit(torch.stack(fn512))
    bs = [bad_ad.infer(_load_img(p, 512).unsqueeze(0)).score for p in test_paths]
    base_au = auroc(bs, test_lab)
    print(f"{cat:12s} baseline(resize512)  AU={base_au:.3f}  ({time.time()-t:.0f}s)", flush=True)

    # --- tiled: 原生分辨率 ---
    fn = [_load_img_native(p) for p in fit_n_paths]
    fd = [_load_img_native(p) for p in fit_d_paths]
    test_imgs = [_load_img_native(p) for p in test_paths]
    for posaware in (False, True):
        t = time.time()
        det = TiledFewShotDetector(bb_small, tile=512, stride=512,
                                   coreset_ratio=0.01, feat_grid=32, position_aware=posaware)
        det.fit_fewshot(fn, fd)
        scores = [det._image_score(im)[0] for im in test_imgs]
        det.predict(test_imgs[0])
        lat = sum(det.predict(test_imgs[i])["latency_ms"] for i in range(3)) / 3
        au = auroc(scores, test_lab)
        ac = acc(scores, test_lab, det.threshold)
        tag = "位置感知库" if posaware else "共享库   "
        print(f"{cat:12s} tiled-{tag}  AU={au:.3f} acc={ac:.3f} lat={lat:.0f}ms  ({time.time()-t:.0f}s)", flush=True)


def main():
    cats = sys.argv[1:] or CATS
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb_big = Backbone(pretrained=True, device=dev)                     # baseline 用原 WRN50
    bb_small = Backbone(name="resnet18", layers=(2, 3), pretrained=True, device=dev)
    for c in cats:
        run_cat(c, bb_big, bb_small)


if __name__ == "__main__":
    main()

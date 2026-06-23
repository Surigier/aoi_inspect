"""max_size 扫描定 2060 安全配置:python scripts/run_ead_sweep.py [类别]
训一次(256细块),推理扫多个 max_size 看 AUROC/延时权衡,锁定 2060<200ms 配置。"""
import sys
import glob
import time
import random
import torch
from aoi.tiled_efficientad import TiledEfficientAD
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
    det = TiledEfficientAD(model_size="small", device=dev, train_steps=10000,
                           tile=256, stride=256, whole_infer=True)
    det.fit_fewshot(fn, fd)
    print(f"{cat} 训练完成 ({time.time()-t:.0f}s)。扫 max_size:", flush=True)
    print(f"{'max_size':10s} {'AUROC':>7} {'延时(4060)':>10} {'2060估':>8}", flush=True)
    for ms in [1024, 1280, 1536, 2048]:
        det.max_size = ms
        scores = [det._image_score(im) for im in test]
        det._image_score(test[0])                       # 预热
        t0 = time.perf_counter()
        for i in range(5):
            det.det.score_large(test[i], max_size=ms)
        lat = (time.perf_counter() - t0) / 5 * 1000
        au = auroc(scores, tl)
        print(f"{ms:10d} {au:7.3f} {lat:9.0f}ms {lat*1.7:7.0f}ms", flush=True)


if __name__ == "__main__":
    main()

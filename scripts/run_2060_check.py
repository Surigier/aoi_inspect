"""2060 可行性检查:python scripts/run_2060_check.py
合成 2500²,测大图混合路径的【峰值显存】(对照 2060 6GB)与【单图延时】
(在本机 GPU 测;2060 约慢 1.5~2x,据此估)。再试更轻配置看延时裕量。"""
import time
import glob
import random
import torch
from aoi.backbone import Backbone
from aoi.hybrid import HybridDetector
from aoi.tiled import TiledFewShotDetector
from eval.mvtec import _load_img

SIZE = 2500


def syn(n, defect=False, rng=None):
    base = _load_img(sorted(glob.glob("data/mvtec/*/train/good/*.png"))[0], SIZE)
    out = []
    for _ in range(n):
        x = (base + torch.randn_like(base) * 0.02).clamp(0, 1)
        if defect:
            y, xx = rng.randint(0, SIZE - 80), rng.randint(0, SIZE - 80)
            x[:, y:y + 3, xx:xx + 80] = 0.0
        out.append(x)
    return out


def measure(name, build_fn, fit_n, fit_d, test_imgs, dev):
    if dev == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    det = build_fn()
    det.fit_fewshot(fit_n, fit_d)
    fit_s = time.time() - t
    pred = (lambda im: det.predict(im)) if hasattr(det, "predict") else None
    pred(test_imgs[0])                                   # 预热
    t0 = time.perf_counter()
    for im in test_imgs[:4]:
        pred(im)
    lat = (time.perf_counter() - t0) / min(4, len(test_imgs)) * 1000
    mem = torch.cuda.max_memory_reserved() / 1024**3 if dev == "cuda" else 0
    print(f"{name:28s} 峰值显存={mem:.2f}GB 延时={lat:.0f}ms "
          f"(2060估~{lat*1.7:.0f}ms) fit={fit_s:.0f}s", flush=True)
    return mem, lat


def main():
    torch.manual_seed(0); rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备={dev};对照:2060=6GB/<200ms", flush=True)
    fit_n = syn(100, rng=rng)
    fit_d = syn(30, defect=True, rng=rng)
    test = syn(6, defect=True, rng=rng)
    gbb = Backbone(pretrained=True, device=dev)
    rbb = Backbone(name="resnet18", layers=(2, 3), pretrained=True, device=dev)

    # 当前混合配置(submit 大图路径)
    measure("混合(当前feat32/stride512)",
            lambda: HybridDetector(gbb, rbb, local_kw=dict(tile=512, stride=512, coreset_ratio=0.01, feat_grid=32)),
            fit_n, fit_d, test, dev)
    # 更轻:更大 stride(更少块)+ 更小 feat_grid
    measure("混合(轻:feat24/stride640)",
            lambda: HybridDetector(gbb, rbb, local_kw=dict(tile=512, stride=640, coreset_ratio=0.01, feat_grid=24)),
            fit_n, fit_d, test, dev)
    # 纯分块(无全局,最轻,看分块本身延时)
    measure("纯分块(feat24/stride640)",
            lambda: TiledFewShotDetector(rbb, tile=512, stride=640, coreset_ratio=0.01, feat_grid=24),
            fit_n, fit_d, test, dev)


if __name__ == "__main__":
    main()

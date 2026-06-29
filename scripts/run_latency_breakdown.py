"""延时拆解:CompetitionLargeDetector 在 AD2 大图上,逐分支计时 + 总延时。
找出 200ms+ 的瓶颈到底在 EAD 还是辅助分支(WideResNet50 结构分支)。
快速 fit(train_steps 小,只为populate,不影响延时),warm 后计时。
用法:python scripts/run_latency_breakdown.py [max_size]
"""
import sys
import glob
import time
import random
import torch
from aoi.competition import CompetitionLargeDetector
from eval.mvtec import _load_img_native

ROOT = "data/mvtec_ad_2"
CATS = ["sheet_metal", "rice", "fabric", "can"]   # 含最快/最慢
MAX_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
MAX_PIXELS = int(sys.argv[2]) if len(sys.argv) > 2 else 800_000


def t_ms(fn, img, reps=5):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fn(img)  # warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main():
    print(f"=== max_size={MAX_SIZE} max_pixels={MAX_PIXELS} ===")
    for cat in CATS:
        train = sorted(glob.glob(f"{ROOT}/{cat}/train/good/*.png"))
        test = sorted(glob.glob(f"{ROOT}/{cat}/test_public/bad/*.png"))
        if not train or not test:
            print(f"{cat}: 数据不足"); continue
        rng = random.Random(0); rng.shuffle(train)
        normals = [_load_img_native(p) for p in train[:20]]
        det = CompetitionLargeDetector(train_steps=200)
        det.branches[0].det.max_size = MAX_SIZE
        det.branches[0].det.max_pixels = MAX_PIXELS
        det.fit_fewshot(normals[:15], normals[15:20])  # 用正常充当缺陷,只为populate
        img = _load_img_native(test[0])
        H, W = img.shape[-2:]
        names = ["EAD", "Color", "Dim", "Struct"]
        per = [t_ms(b.score, img) for b in det.branches]
        total = t_ms(det.predict, img)
        brk = "  ".join(f"{n}={t:.0f}" for n, t in zip(names, per))
        print(f"{cat:12s} 原图{W}x{H}  {brk}  | 总predict={total:.0f}ms")


if __name__ == "__main__":
    main()

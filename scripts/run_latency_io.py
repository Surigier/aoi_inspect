"""端到端延时(按赛题口径:含单图加载+预处理,不含模型加载/初始化)。
在 AD2 大图(2K-5K,最接近赛题2500²)上,warm 计时:磁盘读图+预处理(to tensor) + predict。
用法:python scripts/run_latency_io.py
"""
import glob
import time
import torch
from PIL import Image
import numpy as np
from aoi.competition import CompetitionLargeDetector

ROOT = "data/mvtec_ad_2"
CATS = ["sheet_metal", "rice", "fabric", "can"]


def load_preprocess(path):
    """生产路径:读盘 → RGB → float tensor [0,1] (3,H,W)。这部分按赛题计入耗时。"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_preprocess_fast(path, max_size=1152):
    """优化:读盘后立即在 uint8 上 resize(长边≤max_size)再转 float——
    float转换量从 2500²降到 ~1152²,IO+预处理大幅降。与检测器内部缩放等价。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        s = max_size / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def bench(fn, reps=8):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fn(); fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main():
    print("=== 端到端延时(读盘+预处理+predict),AD2大图,max_size=1152 FP16 ===")
    det = CompetitionLargeDetector(train_steps=200)
    # 用任意正常图 populate(延时与训练质量无关)
    seed = [load_preprocess(p) for p in sorted(glob.glob(f"{ROOT}/sheet_metal/train/good/*.png"))[:20]]
    det.fit_fewshot(seed[:15], seed[15:20])
    for cat in CATS:
        ps = sorted(glob.glob(f"{ROOT}/{cat}/test_public/bad/*.png"))
        if not ps:
            continue
        path = ps[0]
        W, H = Image.open(path).size
        # 原始 vs 早缩放
        t_e2e_old = bench(lambda: det.predict(load_preprocess(path)))
        t_load_f = bench(lambda: load_preprocess_fast(path))
        t_e2e_f = bench(lambda: det.predict(load_preprocess_fast(path)))
        s_old, s_new = t_e2e_old * 1.7, t_e2e_f * 1.7
        flag = "✅" if s_new < 200 else "⚠️超"
        print(f"{cat:12s} {W}x{H}  [原始端到端 {t_e2e_old:.0f}ms→~{s_old:.0f}@2060]  "
              f"[早缩放 读+预处理={t_load_f:.0f}ms 端到端={t_e2e_f:.0f}ms→~{s_new:.0f}@2060 {flag}]", flush=True)


if __name__ == "__main__":
    main()

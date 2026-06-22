"""CPU<2s 挑战验证:python scripts/run_cpu_latency.py
OpenVINO ResNet18 骨干 + 分块,实测 2500² 单图 CPU 延时(目标 <2s)。"""
import time
import glob
import random
import torch
from aoi.ov_backbone import OVBackbone
from aoi.tiled import TiledFewShotDetector
from eval.mvtec import _load_img

SIZE = 2500


def make_normal(base, rng):
    return (base + torch.randn_like(base) * 0.02).clamp(0, 1)


def make_defect(base, rng):
    img = make_normal(base, rng)
    y, x = rng.randint(0, SIZE - 80), rng.randint(0, SIZE - 80)
    img[:, y:y + 3, x:x + 80] = 0.0
    return img


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    t = time.time()
    bb = OVBackbone(name="resnet18", layers=(2, 3), tile=512)
    print(f"OpenVINO 编译完成 ({time.time()-t:.0f}s)", flush=True)
    base = _load_img(sorted(glob.glob("data/mvtec/leather/train/good/*.png"))[0], SIZE)
    fit_n = [make_normal(base, rng) for _ in range(4)]
    det = TiledFewShotDetector(bb, tile=512, stride=512, coreset_ratio=0.01, feat_grid=32)
    t = time.time()
    det.fit_fewshot(fit_n, [make_defect(base, rng) for _ in range(2)])
    print(f"fit 完成 ({time.time()-t:.0f}s)", flush=True)
    det.predict(make_defect(base, rng))          # 预热
    lats = [det.predict(make_defect(base, rng))["latency_ms"] for _ in range(3)]
    print(f"CPU 2500² 单图延时 = {sum(lats)/len(lats):.0f}ms (目标<2000ms) "
          f"{'✅达标' if sum(lats)/len(lats) < 2000 else '❌超时'}", flush=True)


if __name__ == "__main__":
    main()

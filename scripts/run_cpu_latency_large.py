"""CPU<2s 挑战目标——真实验证当前生产大图架构(CompetitionLargeDetector:EAD双学生+
WRN50浅层分割头+DINOv2 ViT-S/14+MobileSAM)在CPU上的真实延时,而不是
run_cpu_latency.py测的那套已过时的OpenVINO ResNet18+分块小架构(那套和submit.py
--mode large实际调用的CompetitionLargeDetector完全是两回事)。

用法:PYTHONPATH=. python scripts/run_cpu_latency_large.py
"""
import time
import glob
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast

SIZE = 2500


def main():
    torch.manual_seed(0)
    device = "cpu"
    print("构建CompetitionLargeDetector(device=cpu,生产默认配置)...", flush=True)
    t0 = time.time()
    det = CompetitionLargeDetector(device=device)
    print(f"构造耗时={time.time()-t0:.1f}s", flush=True)

    files = sorted(glob.glob("data/mvtec/leather/train/good/*.png"))
    normals = [load_fast(p) for p in files[:8]]
    defects = [load_fast(p) for p in files[8:11]]     # 无掩膜:只测检测层,不训监督分割头(先看下限)
    print(f"fit_fewshot(仅检测层,无掩膜)... normals={len(normals)} defects={len(defects)}", flush=True)
    t0 = time.time()
    det.fit_fewshot(normals, defects)
    print(f"fit_fewshot耗时={time.time()-t0:.1f}s", flush=True)

    test_img = load_fast(files[11])
    t0 = time.time()
    _ = det.predict(test_img)
    print(f"首次predict(含惰性初始化)耗时={time.time()-t0:.1f}s", flush=True)

    lats = []
    for _ in range(3):
        t0 = time.time()
        o = det.locate(test_img)
        lats.append((time.time() - t0) * 1000)
    avg = sum(lats) / len(lats)
    print(f"CPU locate()延时(小图,leather原生尺寸,3次均值)={avg:.0f}ms  is_defect={o['is_defect']}", flush=True)
    print(f"目标<2000ms: {'✅达标(注意:测的是小图,非2500²真实尺度,仅做架构下限参考)' if avg < 2000 else '❌超时'}", flush=True)


if __name__ == "__main__":
    main()

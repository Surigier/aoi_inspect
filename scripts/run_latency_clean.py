"""干净延时复核:冷卡空显存下,fit一次→预热→计时 locate(赛题打分口径=定位全链)。
排除脏环境(热降频/显存94%抖动)干扰,给出真实单图 locate 延时。
用法:PYTHONPATH=. python scripts/run_latency_clean.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_realiad, prep_mvtec


def bench(name, prep, n_warm=5, n_timed=25):
    normals, fit_i, fit_m, test_defs, _ = prep()
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    imgs = [im for im, _ in test_defs]
    for i in range(min(n_warm, len(imgs))):                  # 预热(触发cudnn autotune/惰性init)
        det.locate(imgs[i])
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    lats = []
    for i in range(len(imgs)):
        if len(lats) >= n_timed:
            break
        t0 = time.perf_counter()
        det.locate(imgs[i])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    print(f"{name:16s} locate 均值={lats.mean():.0f}ms  中位={np.median(lats):.0f}ms  "
          f"p90={np.percentile(lats,90):.0f}ms  min={lats.min():.0f}ms  (n={len(lats)}, DINO门={det._dino is not None})",
          flush=True)


def main():
    torch.manual_seed(0)
    print("=== 干净延时复核(冷卡空显存,locate全链,含DINO门)===", flush=True)
    bench("电子 pcb", lambda: prep_realiad("pcb"))
    bench("电池 battery", lambda: prep_realiad("phone_battery"))
    bench("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]))


if __name__ == "__main__":
    main()

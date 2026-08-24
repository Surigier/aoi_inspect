"""类型头上生产后的**干净延时归因**。

eval_type_head.py 报出 locate p90=906ms,和记录基线148~180ms差一个量级,但那批
数据有三个混淆没排除:①全是缺陷图(每张都走满分割+SAM,正常图的早退路径一次没吃到)
②compile_infer=False(submit.py生产入口是True)③GPU已连续满负载100分钟。

这里一次fit,然后在**同一个检测器、同一段GPU状态**下背靠背测4组,唯一变量清晰:
  正常图/缺陷图 × 类型头开/关。先跑10张预热(SAM/DINO懒加载+cuda图预热)再计时。

用法:PYTHONPATH=. python scripts/lat_typehead.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from global_context.eval_global_branch import prep_mvtec


def timeit(det, imgs, tag):
    ms = []
    for im in imgs:
        t = time.time(); det.locate(im); ms.append((time.time() - t) * 1000)
    a = np.array(ms)
    print(f"  {tag:22s} n={len(a):3d}  中位={np.median(a):6.0f}ms  p90={np.percentile(a,90):6.0f}ms  "
          f"最大={a.max():6.0f}ms", flush=True)
    return float(np.percentile(a, 90))


def main():
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs = prep_mvtec("cable", ["missing_cable", "missing_wire"])
    t0 = time.time()
    det = CompetitionLargeDetector(compile_infer=True)      # 生产入口的真实设置
    det.fit_fewshot(normals[:-25], fit_i, defect_masks=fit_m)
    print(f"fit完成 {time.time()-t0:.0f}s  type_head={'就绪' if det.type_head else '未启用'}", flush=True)
    if det._dino is None:
        det._calibrate_dino_gate(normals[:-25], fit_i)
    goods = normals[-25:]
    defs = [i for i, _ in test_defs[:25]]

    print("预热(SAM/DINO懒加载 + cuda图预热,不计入)...", flush=True)
    for im in (goods[:5] + defs[:5]):
        det.locate(im)

    th = det.type_head
    print("\n=== 类型头 开 ===", flush=True)
    timeit(det, goods, "正常图(热路径早退)")
    timeit(det, defs, "缺陷图(走满管线)")
    det.type_head = None
    print("=== 类型头 关 ===", flush=True)
    timeit(det, goods, "正常图(热路径早退)")
    timeit(det, defs, "缺陷图(走满管线)")
    det.type_head = th

    print("\n注:赛题现场1000+张里正常图占绝大多数,评分看的是整体延时;"
          "缺陷图走满管线本来就更贵,不能拿它单独代表平均延时。", flush=True)
    print("LAT OK", flush=True)


if __name__ == "__main__":
    main()

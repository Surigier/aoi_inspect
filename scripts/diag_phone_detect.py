"""手机屏图级检出率只有46%——查根因:是"分数卡在阈值下面"(调阈值能救)
还是"缺陷图和正常图分数完全重叠"(特征层面抓不到,调阈值没用)。

三个候选原因,本脚本一次分开:
  ①正常图只有20张(协议1/5)→ 阈值标不准
  ②缺陷极小(640×360上几个像素)→ 图级异常分几乎不动
  ③20张正常图混了多种机型 → 正常方差大 → 阈值被迫抬高

做法:把20张正常图与100张测试缺陷图的**图级分数分布**全部打出来,再算"若阈值取
最优(oracle)能到多少召回",与当前召回对比:
  - oracle召回 ≫ 当前召回 → 阈值问题,可救
  - oracle召回 也很低      → 分数本身分不开,特征层面问题,调阈值没用

用法:PYTHONPATH=. python scripts/diag_phone_detect.py
"""
import glob
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard_phone import _load, _boxes_mask, _defects, GOOD


def main():
    torch.manual_seed(0)
    normals = [_load(f) for f in sorted(glob.glob(f"{GOOD}/*.png"))]
    fit_pairs = _defects("train", 30)
    test_pairs = _defects("val", 100)
    fit_i = [_load(f) for f, _ in fit_pairs]
    fit_m = [_boxes_mask(l) for _, l in fit_pairs]

    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)

    thr = det.decision_threshold()
    ns = np.array([det.frame_score(x) for x in normals])       # fit内样本,乐观
    ds, areas = [], []
    for f, l in test_pairs:
        ds.append(det.frame_score(_load(f)))
        areas.append(_boxes_mask(l).mean())
    ds = np.array(ds); areas = np.array(areas)

    print(f"\n=== 图级分数分布(判决阈值={thr:.4f})===", flush=True)
    print(f"正常图(20,fit内): 中位={np.median(ns):.4f} p90={np.percentile(ns,90):.4f} max={ns.max():.4f}", flush=True)
    print(f"缺陷图(100)     : 中位={np.median(ds):.4f} p10={np.percentile(ds,10):.4f} "
          f"min={ds.min():.4f} max={ds.max():.4f}", flush=True)
    print(f"当前召回        : {(ds>=thr).sum()}/100 = {(ds>=thr).mean():.1%}", flush=True)

    print(f"\n=== 阈值能救多少 ===", flush=True)
    t0 = ns.max()
    print(f"阈值=正常图最大分({t0:.4f}) → 零误报下召回 {(ds>=t0).sum()}/100 = {(ds>=t0).mean():.1%}", flush=True)
    for q in (75, 50, 25):
        t = np.percentile(ns, q)
        print(f"  阈值=正常图p{q}({t:.4f}) → 召回 {(ds>=t).mean():.1%}(代价:约{100-q}%正常图被误判)", flush=True)
    print(f"落在正常图分数区间内的缺陷图占比: {((ds>=ns.min())&(ds<=ns.max())).mean():.1%}  ← 越高越分不开", flush=True)

    print(f"\n=== 按缺陷面积分档看召回(验证'缺陷太小')===", flush=True)
    qs = np.percentile(areas, [25, 50, 75])
    for lo, hi, name in [(0, qs[0], "最小25%"), (qs[0], qs[1], "25~50%"),
                         (qs[1], qs[2], "50~75%"), (qs[2], 1.0, "最大25%")]:
        m = (areas > lo) & (areas <= hi)
        if m.sum():
            print(f"  {name:8s} 面积{lo:.5f}~{hi:.5f} n={m.sum():3d} 召回={(ds[m]>=thr).mean():.1%} "
                  f"分数中位={np.median(ds[m]):.4f}", flush=True)

    print(f"\n=== 正常图自身方差(验证'混了多种机型')===", flush=True)
    print(f"  20张正常图: min={ns.min():.4f} max={ns.max():.4f} 极差/中位={(ns.max()-ns.min())/max(np.median(ns),1e-9):.2f}", flush=True)
    print("  判读:极差/中位≫1 → 正常图之间的差异比'正常vs缺陷'还大,阈值被迫抬高", flush=True)
    print("DIAG_PHONE OK", flush=True)


if __name__ == "__main__":
    main()

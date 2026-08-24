"""类型归属:整图打分 vs **掩膜内打分** 的判别力对比(验证假设,不改生产)。

现状问题:类型归属拿整张图问每个辅助分支"像不像你这类",小缺陷被整图统计稀释——
实测pill的真实色彩缺陷里,色彩分支的z分有7/17次只排第3(ColorADBranch把图下采样到
320²后按16×16格取每格色度均值,一格约20px,小色斑被均值平摊掉)。

假设:locate()已经产出缺陷掩膜,**位置是已知的**。把打分限制在掩膜覆盖的格子上
(而不是整图取max),稀释问题就不存在了。

本脚本对每张检出的缺陷图同时算两种分,看哪种的类型判别更准。不改任何生产代码。
用法:PYTHONPATH=. python scripts/diag_type_masked.py
"""
import collections
import numpy as np
import torch
import torch.nn.functional as F
from aoi.competition import CompetitionLargeDetector, _down
from aoi.fusion import znorm
from global_context.eval_global_branch import prep_mvtec
from scripts.run_scorecard_5types import prep_mvtec_color

JOBS = [("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]), "外观缺陷"),
        ("pill", lambda: prep_mvtec("pill", ["color"]), "色彩变化"),
        ("carpet", lambda: prep_mvtec_color("carpet")[:4], "色彩变化"),
        ("metal_nut", lambda: prep_mvtec_color("metal_nut")[:4], "色彩变化")]


def aux_results(det, img):
    """3路辅助分支的BranchResult。注意**尺寸分支是全局标量**(量整图前景面积,
    anomaly_map=None)——尺寸偏差本来就是全局属性,掩膜化对它没意义,保持原样。
    只有色彩/结构分支有(grid,grid)空间图,可以做掩膜内取max。"""
    pre = _down(img, det.aux_size).cpu()
    return [b.branch.infer(pre) for b in det.branches[1:]]


def masked_max(amap, mask):
    """只在掩膜覆盖到的格子里取max;掩膜下采样到amap的网格尺寸。"""
    g = amap.shape[0]
    m = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                      size=(g, g), mode="area")[0, 0].numpy() > 0.02
    return float(amap[m].max()) if m.any() else float(amap.max())


def main():
    torch.manual_seed(0)
    tot = collections.Counter()
    for cat, prep, truth in JOBS:
        normals, fit_i, fit_m, test_defs = prep()
        det = CompetitionLargeDetector(train_steps=100)
        det.fit_fewshot(normals[:40], fit_i, defect_masks=fit_m)
        names = [b.defect_type for b in det.branches]
        # 掩膜内打分需要自己的正常统计量(尺度和整图max不同)——用留出正常图估
        cal = normals[40:70]
        cal_res = [aux_results(det, x) for x in cal]
        mstats = []
        for j in range(3):
            v = [(float(r[j].anomaly_map.max()) if r[j].anomaly_map is not None else r[j].score)
                 for r in cal_res]                          # 正常图无缺陷区,用整图max当基线
            mu = sum(v) / len(v); sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
            mstats.append((mu, sd))
        c_whole, c_mask, n = collections.Counter(), collections.Counter(), 0
        for img, _ in test_defs[:20]:
            o = det.locate(img)
            if not o["is_defect"] or o.get("mask") is None:
                continue
            n += 1
            ead_raw = det.branches[0].score(img)
            res = aux_results(det, img)
            # ①现状:整图打分
            zs_w = [znorm(ead_raw, *det.stats[0])] + \
                   [znorm(r.score, *det.stats[i + 1]) for i, r in enumerate(res)]
            c_whole[names[int(np.argmax(zs_w))]] += 1
            # ②掩膜内打分(EAD不变;尺寸分支是全局标量也不变;只有色彩/结构改成掩膜内max)
            zs_m = [znorm(ead_raw, *det.stats[0])]
            for i, r in enumerate(res):
                v = masked_max(r.anomaly_map, o["mask"]) if r.anomaly_map is not None else r.score
                zs_m.append(znorm(v, *mstats[i]))
            c_mask[names[int(np.argmax(zs_m))]] += 1
        ok_w, ok_m = c_whole.get(truth, 0), c_mask.get(truth, 0)
        tot["n"] += n; tot["w"] += ok_w; tot["m"] += ok_m
        print(f"{cat}(真实={truth}, 检出{n}): 整图={ok_w}/{n} {dict(c_whole)} | "
              f"掩膜内={ok_m}/{n} {dict(c_mask)}", flush=True)
    print(f"\n合计: 整图 {tot['w']}/{tot['n']}={tot['w']/max(tot['n'],1):.0%} | "
          f"掩膜内 {tot['m']}/{tot['n']}={tot['m']/max(tot['n'],1):.0%}", flush=True)
    print("MASKED OK", flush=True)


if __name__ == "__main__":
    main()

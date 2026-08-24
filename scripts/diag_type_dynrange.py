"""类型归属:z分动态范围归一(用分支自己的缺陷响应做尺度) vs 现状。

已排除的两条错误路线(留档,别重开):
  ①"把EAD降级成兜底、只在3个专家分支里选"——hazelnut会永远判不出"外观缺陷"(0/18)
  ②"掩膜内打分"(以为小缺陷被整图稀释)——pill的判定分布**一模一样**(色彩6/外观11),
    说明错判的赢家始终是EAD,抬高专家分支的分根本不影响EAD自己的分,稀释不是主因

真症结:"外观缺陷"没有专属检测器,被映射给了EAD;EAD是**通用**异常检测器,对任何
缺陷都强响应,z分天然碾压专家分支。现状 z=(分-正常均值)/正常标准差 只反映"比正常
高多少"——分支越强、z越大,跨分支根本不可比。

本方案:用**分支自己的缺陷响应**当尺度(fit的30张缺陷图上的均值),
  z' = (分 - 正常均值) / max(缺陷均值 - 正常均值, eps)
这样每个分支的典型缺陷响应都归一到≈1.0,问的是"这个分支的反应是否超出**它自己**的
常态",而不是"哪个分支绝对值大"。fit缺陷图是多类型混合的,不影响该分支动态范围的估计。

用法:PYTHONPATH=. python scripts/diag_type_dynrange.py
"""
import collections
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.fusion import znorm
from global_context.eval_global_branch import prep_mvtec
from scripts.run_scorecard_5types import prep_mvtec_color

JOBS = [("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]), "外观缺陷"),
        ("pill", lambda: prep_mvtec("pill", ["color"]), "色彩变化"),
        ("carpet", lambda: prep_mvtec_color("carpet")[:4], "色彩变化"),
        ("metal_nut", lambda: prep_mvtec_color("metal_nut")[:4], "色彩变化")]


def main():
    torch.manual_seed(0)
    tot = collections.Counter()
    for cat, prep, truth in JOBS:
        normals, fit_i, fit_m, test_defs = prep()
        det = CompetitionLargeDetector(train_steps=100)
        det.fit_fewshot(normals[:40], fit_i, defect_masks=fit_m)
        names = [b.defect_type for b in det.branches]

        # 每个分支在fit缺陷图上的均值 → 它自己的"典型缺陷响应"尺度
        dmeans = []
        for i, b in enumerate(det.branches):
            v = [b.score(d) if i == 0 else b.score(d) for d in fit_i]
            dmeans.append(sum(v) / len(v))
        spans = [max(dm - m, 1e-9) for dm, (m, s) in zip(dmeans, det.stats)]
        print(f"--- {cat} 各分支 正常均值/缺陷均值/动态范围 ---", flush=True)
        for nm, (m, s), dm, sp in zip(names, det.stats, dmeans, spans):
            print(f"    {nm:10s} 正常={m:.4g} 缺陷={dm:.4g} 范围={sp:.4g} 正常std={s:.4g}", flush=True)

        c_old, c_new, n = collections.Counter(), collections.Counter(), 0
        for img, _ in test_defs[:20]:
            o = det.locate(img)
            if not o["is_defect"]:
                continue
            n += 1
            raws = [det.branches[0].score(img)] + det._aux_raws(img)
            z_old = [znorm(r, m, s) for r, (m, s) in zip(raws, det.stats)]
            z_new = [(r - m) / sp for r, (m, s), sp in zip(raws, det.stats, spans)]
            c_old[names[int(np.argmax(z_old))]] += 1
            c_new[names[int(np.argmax(z_new))]] += 1
        ok_o, ok_n = c_old.get(truth, 0), c_new.get(truth, 0)
        tot["n"] += n; tot["o"] += ok_o; tot["w"] += ok_n
        print(f"  {cat}(真实={truth},检出{n}): 现状={ok_o}/{n} {dict(c_old)} | "
              f"动态范围归一={ok_n}/{n} {dict(c_new)}", flush=True)
    print(f"\n合计: 现状 {tot['o']}/{tot['n']}={tot['o']/max(tot['n'],1):.0%} | "
          f"动态范围归一 {tot['w']}/{tot['n']}={tot['w']/max(tot['n'],1):.0%}", flush=True)
    print("DYNRANGE OK", flush=True)


if __name__ == "__main__":
    main()

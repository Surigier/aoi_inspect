"""VLM类型头的**端到端**真实数字:走完整locate()、用预测掩膜(不是GT掩膜)。

之前87%那个数是用GT掩膜的离线组件测试,是上界。这里测生产状态:掩膜由检测器
自己预测,框歪了VLM学到的判据也就看错了地方,数字必然低于87%——这个数才是能
写进汇报的。同时归因类型头带来的延时增量(它进了推理热路径)。

用法:PYTHONPATH=. python scripts/eval_type_head.py
"""
import collections
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from global_context.eval_global_branch import prep_mvtec
from scripts.run_scorecard_5types import prep_mvtec_color

# 启发式的4类名 → 赛题5类名。"缺件/逻辑"一个顶两类,判对任一都算它对(对基线宽容)
HEUR = {"外观缺陷": {"常见外观缺陷"}, "色彩变化": {"色彩变化"},
        "尺寸偏差": {"尺寸偏差"}, "缺件/逻辑": {"缺件少件", "逻辑错误"}}

JOBS = [("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]), "常见外观缺陷"),
        ("pill", lambda: prep_mvtec("pill", ["color"]), "色彩变化"),
        ("carpet", lambda: prep_mvtec_color("carpet")[:4], "色彩变化"),
        ("metal_nut", lambda: prep_mvtec_color("metal_nut")[:4], "色彩变化"),
        ("cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"]), "缺件少件")]


def main():
    torch.manual_seed(0)
    tot = collections.Counter()
    head_ms, loc_ms = [], []
    for cat, prep, truth in JOBS:
        normals, fit_i, fit_m, test_defs = prep()
        t0 = time.time()
        det = CompetitionLargeDetector()
        det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
        print(f"  [{cat}] fit完成 {time.time()-t0:.0f}s  type_head={'就绪' if det.type_head else '未启用(降级)'}",
              flush=True)
        if det._dino is None:
            det._calibrate_dino_gate(normals, fit_i)
        cv, ch, n = 0, 0, 0
        cnt = collections.Counter()
        for img, _ in test_defs[:12]:
            t1 = time.time()
            o = det.locate(img)
            loc_ms.append((time.time() - t1) * 1000)
            if not o["is_defect"] or o.get("mask") is None:
                continue
            n += 1
            cnt[o["defect_type"]] += 1
            cv += o["defect_type"] == truth
            # 同一张图上的启发式对照(不重跑检测,直接用缓存的raws)
            if o.get("_raws") is not None:
                ch += truth in HEUR.get(det._ztype(o["_raws"]), set())
            # 类型头单独计时(归因它在热路径上的增量)
            if det.type_head is not None:
                t2 = time.time()
                det.type_head.predict(det, img, o["mask"], o.get("_raws"))
                head_ms.append((time.time() - t2) * 1000)
        tot["n"] += n; tot["vlm"] += cv; tot["heur"] += ch
        print(f"{cat}(真实={truth},检出{n}): VLM头={cv}/{n}  启发式={ch}/{n}  {dict(cnt)}", flush=True)
    n = max(tot["n"], 1)
    print(f"\n=== 端到端(预测掩膜) === VLM类型头 {tot['vlm']}/{n}={tot['vlm']/n:.0%}  |  "
          f"启发式 {tot['heur']}/{n}={tot['heur']/n:.0%}  |  GT掩膜离线上界=87%", flush=True)
    if head_ms:
        print(f"类型头延时:均值{np.mean(head_ms):.1f}ms p90={np.percentile(head_ms,90):.1f}ms  "
              f"| 整条locate p90={np.percentile(loc_ms,90):.0f}ms", flush=True)
    print("TYPEHEAD EVAL OK", flush=True)


if __name__ == "__main__":
    main()

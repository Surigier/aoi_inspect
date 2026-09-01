"""ead_smooth_k(检测头score_large的噪声平滑)A/B:CLAHE(±双边滤波)在grid/screw上
验证出"个别类目受益、个别类目受损"(screw误报率27.5%→52.5%)之后,把杠杆从"改输入
图像"换成"改检测头聚合方式"——score_large原来直接取教师-学生差异图的**全局最大值**
当图级分数,对孤立单像素噪声极度敏感,smooth_k>1时先做smooth_k×smooth_k均值池化
再取最大值,理论上能抑制这类噪声、同时基本不伤有空间连续性的真实缺陷。

验证范围(比CLAHE那次更广,不只测2个类目):
①5类生产成绩单(hazelnut/cable/pill/pcb/phone_battery,run_scorecard.py同款,官方口径)
②screw(原图,不掺CLAHE)——历史上误报率就偏高(27.5%),是天然的压力测试类目

对比smooth_k=1(baseline,代码默认值,行为与改动前逐位一致)/3/5三档,同一批fit/test
切分种子完全一致,唯一变量是smooth_k。

用法:PYTHONPATH=. python scripts/run_ead_smooth_ab.py
"""
import os
import numpy as np
import torch

from scripts.run_scorecard import evaluate, prep_mvtec, prep_realiad

JOBS = [
    ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
    ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
    ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
    ("电子 pcb", lambda: prep_realiad("pcb")),
    ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
    ("压力 screw", lambda: prep_mvtec("screw", ["manipulated_front", "scratch_head",
                                                "scratch_neck", "thread_side", "thread_top"])),
]


def run_one(smooth_k):
    os.environ["EAD_SMOOTH_K"] = str(smooth_k)
    print(f"\n########## smooth_k={smooth_k} ##########", flush=True)
    rows = []
    for name, prep in JOBS:
        rows.append((name, evaluate(name, *prep())))
    a, g, p, h = (np.mean([r[1][i] for r in rows]) for i in range(4))
    print(f"smooth_k={smooth_k} 均值: 图级acc={a:.3f} 含漏检IoU={g:.3f} 纯定位IoU={p:.3f} 框命中@0.5={h:.3f}",
          flush=True)
    return rows


def main():
    torch.manual_seed(0)
    print("=== ead_smooth_k(检测头噪声平滑)A/B:5类生产成绩单 + screw压力测试 ===")
    all_results = {k: run_one(k) for k in (1, 3, 5)}
    print("\n\n=== 汇总对比(逐类目) ===")
    for name, _ in JOBS:
        line = f"{name:20s}"
        for k in (1, 3, 5):
            res = next(r for n, r in all_results[k] if n == name)
            line += f"  k={k}: acc={res[0]:.3f} IoU={res[1]:.3f} hit={res[3]:.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()

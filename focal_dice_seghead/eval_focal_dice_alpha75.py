"""FocalDice原判负结论里留的一个疑点:alpha=0.25是RetinaNet论文对付"背景远多于
前景"场景的经验设定,数学上是在**降权**正样本(缺陷像素)、加权负样本——但我们
pcb/phone_battery这类"正样本像素少到个位数"的极端场景,直觉上该反过来加权正样本
才对。原判负实验(pcb/phone_battery跌得最狠)可能不是"FocalLoss这个机制没用",
是"alpha方向选反了"。这里把alpha从0.25换成0.75,同一批类目重测,同gamma=2.0/
同steps/lr/batch,唯一变量是alpha方向。

用法:PYTHONPATH=. python focal_dice_seghead/eval_focal_dice_alpha75.py
"""
import numpy as np
import torch
from focal_dice_seghead.eval_focal_dice import run_one
from focal_dice_seghead.losses import FocalDiceLoss
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad


def main():
    torch.manual_seed(0)
    lossf = FocalDiceLoss(alpha=0.75, gamma=2.0)
    jobs = [
        ("生产:pcb(微小缺陷,最该受益)", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("生产:cable(回归检查)", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        row = run_one(name, *prep(), lossf=lossf)
        if row is None:
            continue
        names.append(name)
        deltas.append(row["fd"][0] - row["base"][0])
    if deltas:
        d = np.array(deltas)
        passed = (np.median(d) >= 0.005 and np.mean(d) > 0
                 and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
        print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
              f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

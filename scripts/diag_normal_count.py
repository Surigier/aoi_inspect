"""隔离"正常图数量"这一个变量:手机屏成绩单只有20张正常图(协议要100),
检出率46%。必须先回答——这46%里有多少是数据饥饿造成的,多少是真实弱项?

做法:拿 phone_battery(有100张正常图、基线已知),**只改正常图数量**,
100 → 50 → 20,其余全部不变,看检出率(召回)怎么变。

  掉幅相当 → 手机屏的46%主要是数据饥饿,评委给100张时不会这么差,别过度投入
  掉幅很小 → 46%是真实弱项,必须修检测门

同时打印每档的阈值,看阈值是不是随正常图变少而被迫抬高(那是漏检的直接机制)。

用法:PYTHONPATH=. python scripts/diag_normal_count.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_realiad


def main(cat="phone_battery"):
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs, goods = prep_realiad(cat)
    print(f"{cat}: 可用正常图{len(normals)} fit缺陷{len(fit_i)} 测试缺陷{len(test_defs)} 测试正常{len(goods)}",
          flush=True)
    rows = []
    for n_norm in (100, 50, 20):
        torch.manual_seed(0)
        det = CompetitionLargeDetector()
        det.fit_fewshot(normals[:n_norm], fit_i, defect_masks=fit_m)
        if det._dino is None:
            det._calibrate_dino_gate(normals[:n_norm], fit_i)
        thr = det.decision_threshold()
        n_det = sum(1 for img, _ in test_defs if det.locate(img)["is_defect"])
        n_fp = sum(1 for img, _ in goods if det.locate(img)["is_defect"])
        rec = n_det / max(len(test_defs), 1)
        fpr = n_fp / max(len(goods), 1)
        rows.append((n_norm, rec, fpr, thr))
        print(f"  正常图{n_norm:3d}张 → 检出率(召回)={rec:.1%}  误报率={fpr:.1%}  阈值={thr:.4f}", flush=True)
    print("\n=== 结论判读 ===", flush=True)
    if len(rows) >= 3:
        d = rows[0][1] - rows[2][1]
        print(f"  100张→20张 召回变化 {rows[0][1]:.1%} → {rows[2][1]:.1%}  (掉 {d:+.1%})", flush=True)
        print(f"  阈值变化 {rows[0][3]:.4f} → {rows[2][3]:.4f}", flush=True)
        print("  掉幅≫20个百分点 → 手机屏46%主因是数据饥饿(评委给100张不会这么差)", flush=True)
        print("  掉幅很小        → 46%是真实弱项,必须修检测门", flush=True)
    print("DIAG_NCOUNT OK", flush=True)


if __name__ == "__main__":
    main()

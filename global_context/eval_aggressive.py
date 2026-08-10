"""GCAD判负结论里没排除的一个疑点:PixelAE/EmbedAE的训练配方(steps=300/lr=1e-3)
有没有太保守——原结论排除的是"模型容量不够"(EmbedAE用DINOv2语义嵌入没比PixelAE
纯像素强多少),但没测过"练得够不够狠"这个独立维度,和WRN-LoRA/seg_head同一个模式
的疑点还没验过。这里把steps从300拉到900、lr从1e-3拉到3e-3(倍数比照seg_head那次
300→900/5e-3→1e-2的验证),同一批9类目重测。

用法:PYTHONPATH=. python global_context/eval_aggressive.py
"""
import numpy as np
import torch
from global_context.eval_global_branch import run_one, prep_loco, prep_mvtec, prep_realiad


def main():
    torch.manual_seed(0)
    print("=== GCAD(PixelAE/EmbedAE)激进训练配方(900步/3e-3)重验 ===", flush=True)
    jobs = [
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:juice_bottle", lambda: prep_loco("juice_bottle", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
        ("logical:screw_bag", lambda: prep_loco("screw_bag", "logical_anomalies")),
        ("logical:splicing_connectors", lambda: prep_loco("splicing_connectors", "logical_anomalies")),
        ("structural:breakfast_box(回归检查)", lambda: prep_loco("breakfast_box", "structural_anomalies")),
        ("structural:juice_bottle(回归检查)", lambda: prep_loco("juice_bottle", "structural_anomalies")),
        ("生产:cable(回归检查)", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("生产:pcb(回归检查)", lambda: prep_realiad("pcb")),
    ]
    names, pix_deltas, emb_deltas = [], [], []
    for name, prep in jobs:
        row = run_one(name, *prep(), ae_steps=900, ae_lr=3e-3)
        if row is None:
            continue
        names.append(name)
        pix_deltas.append(row["pix"][0] - row["base"][0])
        emb_deltas.append(row["emb"][0] - row["base"][0])

    for tag, deltas in [("PixelAE", pix_deltas), ("EmbedAE", emb_deltas)]:
        d = np.array(deltas)
        passed = (np.median(d) >= 0.005 and np.mean(d) > 0
                  and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
        print(f"\n=== {tag}汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
              f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

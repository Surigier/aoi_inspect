"""把diagnose_aggressive.py里验证过的"保守配置太弱,激进配置(lr=1e-3/steps=300)才
是WRN-LoRA该测的量级"这个结论,从AD2数据集(只该测延时/形状,不做精度叙事)搬到真正
的生产类目上重新验证——这次的正负结果才能算数,能用在竞赛材料里。

用法:PYTHONPATH=. python wrn_lora/validate_production.py
"""
from wrn_lora.diagnose import train_and_collect, weight_delta_ratio, logit_delta_stats
from global_context.eval_global_branch import prep_mvtec
from scripts.run_scorecard_5types import prep_mvtec_color


def _hazelnut(_): return prep_mvtec("hazelnut", ["crack", "cut", "hole"])
def _cable(_): return prep_mvtec("cable", ["missing_cable", "missing_wire"])
def _pill(_): return prep_mvtec("pill", ["color"])
def _color(cat): return lambda _: prep_mvtec_color(cat)[:4]


def main():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print("=== 生产类目上重验激进LoRA配置(保守vs激进,margin判据以test IoU的Δ为准) ===", flush=True)
    jobs = [
        ("hazelnut", 0, _hazelnut),
        ("cable", 0, _cable),
        ("pill", 0, _pill),
        ("carpet", 0, _color("carpet")),
        ("leather", 0, _color("leather")),
        ("metal_nut", 0, _color("metal_nut")),
        ("wood", 0, _color("wood")),
    ]
    names, deltas = [], []
    for name, seed, prep_fn in jobs:
        h_base, e_base, _, fit_b, test_b, _, test_defs, _ = train_and_collect(
            name, seed, 0, 2, device=device, prep_fn=prep_fn)
        h_lora, e_lora, lms, fit_l, test_l, _, _, _ = train_and_collect(
            name, seed, 2, 4, lora_lr=1e-3, steps=300, device=device, prep_fn=prep_fn)
        ratios = weight_delta_ratio(lms)
        mean_d, p95_d = logit_delta_stats(h_base, e_base, h_lora, e_lora, test_defs, device)
        d = test_l - test_b
        names.append(name); deltas.append(d)
        print(f"{name} (激进配置)", flush=True)
        print(f"  ①各层||ΔW||/||W_base||: {[f'{r:.2e}' for r in ratios]}", flush=True)
        print(f"  ②logit差异 mean={mean_d:.4f} p95={p95_d:.4f}", flush=True)
        print(f"  ③fit IoU: base={fit_b:.3f} lora={fit_l:.3f} (Δ={fit_l-fit_b:+.3f})  |  "
              f"test IoU: base={test_b:.3f} lora={test_l:.3f} (Δ={d:+.3f})", flush=True)

    import numpy as np
    d = np.array(deltas)
    passed = (np.median(d) >= 0.005 and np.mean(d) > 0
              and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
    print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
          f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
    print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

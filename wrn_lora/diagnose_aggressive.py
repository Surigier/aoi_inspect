"""拆分"WRN-LoRA判负"里悬而未决的两个假说:diagnose.py里唯一的压力测试
(lr=1e-3/steps=300)只在fruit_jelly上跑过,sheet_metal/walnuts从没用这套激进配置
测过——不知道当时2/3类别打平,是"类别天生学不出东西"还是"保守配置(lr=2e-4/150步)
本身太弱,激进配置下可能也会涨"。这里把同一套激进配置搬到sheet_metal/walnuts上,
和fruit_jelly放在一起对比。

判读:如果sheet_metal/walnuts在激进配置下依然平/负,而fruit_jelly依然大涨→类别
天生假说成立(某些类目有LoRA能学的规律,某些没有);如果sheet_metal/walnuts也涨了
→说明是超参数问题,之前的保守配置判负结论需要重新评估,不是"该不该用LoRA"是
"该用多猛的LoRA"。

用法:PYTHONPATH=. python wrn_lora/diagnose_aggressive.py
"""
from wrn_lora.diagnose import train_and_collect, weight_delta_ratio, logit_delta_stats


def main():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print("=== 激进配置(lr=1e-3/steps=300)搬到全部3类,拆分类别假说vs超参数假说 ===", flush=True)
    for cat, seed in [("sheet_metal", 0), ("walnuts", 0), ("fruit_jelly", 1)]:
        h_base, e_base, _, fit_b, test_b, _, test_defs, _ = train_and_collect(cat, seed, 0, 2, device=device)
        h_lora, e_lora, lms, fit_l, test_l, _, _, _ = train_and_collect(
            cat, seed, 2, 4, lora_lr=1e-3, steps=300, device=device)
        ratios = weight_delta_ratio(lms)
        mean_d, p95_d = logit_delta_stats(h_base, e_base, h_lora, e_lora, test_defs, device)
        print(f"{cat} seed={seed} (激进配置)", flush=True)
        print(f"  ①各层||ΔW||/||W_base||: {[f'{r:.2e}' for r in ratios]}", flush=True)
        print(f"  ②logit差异 mean={mean_d:.4f} p95={p95_d:.4f}", flush=True)
        print(f"  ③fit IoU: base={fit_b:.3f} lora={fit_l:.3f} (Δ={fit_l-fit_b:+.3f})  |  "
              f"test IoU: base={test_b:.3f} lora={test_l:.3f} (Δ={test_l-test_b:+.3f})", flush=True)


if __name__ == "__main__":
    main()

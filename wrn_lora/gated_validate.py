"""给激进配置LoRA加OOF门控:30张缺陷fit图内部切一份留出集(val_frac),LoRA只在
训练切分上训,在内部留出集上自检——留出集上真的比base强才启用LoRA,否则退回base
(等于没做LoRA)。这样leather那种"fit涨test跌"的过拟合情况,理论上会被内部留出集
提前发现并自动跳过,不会拖累生产分数。

复用wrn_lora/diagnose.py的train_and_collect(不改它的训练逻辑),用一个"分裂版
prep_fn"把fit集切成train_sub(用来训)和val_sub(当train_and_collect的"test_defs"
参数,借它内部机制算出的test_iou就是我们要的内部留出集IoU,不用改函数本身)。
训完再用返回的head/extractor/thr,在真正的held-out test_defs上分别评估base/lora,
按门控结果选一个当"gated"结果上报。

⚠️已知风险:fit集本来就小(prep_mvtec等函数给的fit_i经常只有5~30张),内部再切一刀
留出集可能小到不可靠(和AHL那次"越拆越没监督信息"是同一类风险),这里如实跑出来看
是不是真的可靠,不预设结论。

用法:PYTHONPATH=. python wrn_lora/gated_validate.py
"""
import random
import numpy as np
import torch
from wrn_lora.diagnose import train_and_collect, _mask_to
from wrn_lora.experiment import per_image_iou
from global_context.eval_global_branch import prep_mvtec
from scripts.run_scorecard_5types import prep_mvtec_color


def _hazelnut(_): return prep_mvtec("hazelnut", ["crack", "cut", "hole"])
def _cable(_): return prep_mvtec("cable", ["missing_cable", "missing_wire"])
def _pill(_): return prep_mvtec("pill", ["color"])
def _color(cat): return lambda _: prep_mvtec_color(cat)[:4]


def run_gated(cat, seed, prep_fn, val_frac=0.3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    normals, fit_i, fit_m, test_defs = prep_fn(cat)
    n = len(fit_i)
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_fit_i = [fit_i[i] for i in train_idx]; train_fit_m = [fit_m[i] for i in train_idx]
    val_defs = [(fit_i[i], fit_m[i]) for i in val_idx]
    print(f"  [{cat}] fit={n} -> train_sub={len(train_idx)} val_sub={len(val_idx)}", flush=True)

    def split_prep(_):
        return normals, train_fit_i, train_fit_m, val_defs

    h_base, e_base, _, _, val_b, thr_b, _, _ = train_and_collect(
        cat, seed, 0, 2, device=device, prep_fn=split_prep)
    h_lora, e_lora, lms, _, val_l, thr_l, _, _ = train_and_collect(
        cat, seed, 2, 4, lora_lr=1e-3, steps=300, device=device, prep_fn=split_prep)

    gate_pass = val_l > val_b

    def eval_on(head, extractor, thr, defs):
        ious = []
        with torch.no_grad():
            for img, mk in defs:
                feat = extractor(img.to(device))[None]
                logit = head(feat)[0, 0].cpu().numpy()
                gh, gw = logit.shape
                pred = (logit >= thr).astype(np.uint8)
                ious.append(per_image_iou(pred, _mask_to(mk, gh, gw)))
        return float(np.mean(ious))

    test_base = eval_on(h_base, e_base, thr_b, test_defs)
    test_lora = eval_on(h_lora, e_lora, thr_l, test_defs)
    test_gated = test_lora if gate_pass else test_base
    return dict(val_base=val_b, val_lora=val_l, gate_pass=gate_pass,
                test_base=test_base, test_lora=test_lora, test_gated=test_gated)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== OOF门控LoRA:留出集自检,过拟合自动退回base ===", flush=True)
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
        r = run_gated(name, seed, prep_fn, device=device)
        d = r["test_gated"] - r["test_base"]
        names.append(name); deltas.append(d)
        print(f"{name}: val(base={r['val_base']:.3f} lora={r['val_lora']:.3f}) "
              f"gate={'开' if r['gate_pass'] else '关'}  "
              f"test(base={r['test_base']:.3f} lora={r['test_lora']:.3f} gated={r['test_gated']:.3f}) "
              f"Δ(gated-base)={d:+.3f}", flush=True)

    d = np.array(deltas)
    passed = (np.median(d) >= 0.005 and np.mean(d) > 0
              and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
    print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
          f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
    print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

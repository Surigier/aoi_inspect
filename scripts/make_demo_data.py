"""整理一批演示/验收用数据集,按赛题协议的目录结构组织。

产出 demo_data/<产品>/{normal,defect,mask,test}/,可直接喂给:
    python submit.py --normal ... --defect ... --defect-mask ... --test ... --out result.csv
测试集**混合缺陷与正常**(赛题就是混合流),并另存一份 answer.csv 作为对照答案,
方便工作人员核对检出/漏检/误报。

用法:PYTHONPATH=. python scripts/make_demo_data.py [--n-test 40]
"""
import argparse
import csv
import glob
import random
import shutil
from pathlib import Path
from PIL import Image

GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
# 覆盖赛题5类缺陷里能对应上的4类(尺寸偏差用metal_nut的bent;逻辑错误用cable_swap/flip)
PRODUCTS = {
    "手机中框代理_hazelnut": ("hazelnut", None),
    "线束_cable": ("cable", None),
    "表面_carpet": ("carpet", None),
    "药片_pill": ("pill", None),
    "金属件_metal_nut": ("metal_nut", None),
}


def build(name, cat, out_root, n_norm=100, n_def=30, n_test=40, seed=0):
    root = Path(f"data/mvtec/{cat}")
    if not root.exists():
        print(f"跳过 {name}:无数据"); return None
    ns = sorted(glob.glob(str(root / "train/good/*.png")))
    gs = sorted(glob.glob(str(root / "test/good/*.png")))
    df = []
    for sub in sorted((root / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            for f in sorted(sub.glob("*.png")):
                m = GT / cat / "ground_truth" / sub.name / (f.stem + "_mask.png")
                if m.exists():
                    df.append((str(f), str(m), sub.name))
    rng = random.Random(seed)
    rng.shuffle(ns); rng.shuffle(gs); rng.shuffle(df)
    d = out_root / name
    for sub in ("normal", "defect", "mask", "test"):
        (d / sub).mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(ns[:n_norm]):
        shutil.copy(p, d / "normal" / f"n{i:03d}.png")
    fit_d = df[:n_def]
    for i, (p, m, _t) in enumerate(fit_d):
        shutil.copy(p, d / "defect" / f"d{i:03d}.png")
        shutil.copy(m, d / "mask" / f"d{i:03d}.png")

    # 测试流:缺陷与正常混合(赛题就是混合的),另存答案
    n_td = min(n_test // 2, len(df) - n_def)
    picks = [(p, "缺陷", t) for p, _m, t in df[n_def:n_def + n_td]] + \
            [(p, "正常", "") for p in gs[:n_test - n_td]]
    rng.shuffle(picks)
    rows = []
    for i, (p, lab, t) in enumerate(picks):
        fn = f"t{i:03d}.png"
        shutil.copy(p, d / "test" / fn)
        rows.append([fn, lab, t])
    with open(d / "answer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "真实标签", "缺陷类型(MVTec原始)"]); w.writerows(rows)
    sz = Image.open(ns[0]).size
    print(f"  {name:24s} 原生{sz}  正常{min(n_norm,len(ns))} 缺陷{len(fit_d)} "
          f"测试{len(picks)}(缺陷{n_td}+正常{len(picks)-n_td})", flush=True)
    return d


def main(n_test=40):
    out = Path("demo_data")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    print("按赛题协议组织演示数据集(100正常 + 30缺陷带掩膜 + 混合测试流):", flush=True)
    made = [build(n, c, out, n_test=n_test) for n, (c, _) in PRODUCTS.items()]
    made = [m for m in made if m]
    print(f"\n共 {len(made)} 个产品,输出在 {out}/", flush=True)
    print("每个产品可直接跑:", flush=True)
    print("  python submit.py --normal demo_data/<产品>/normal --defect demo_data/<产品>/defect \\", flush=True)
    print("      --defect-mask demo_data/<产品>/mask --test demo_data/<产品>/test --out result.csv", flush=True)
    print("  对照答案:demo_data/<产品>/answer.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-test", type=int, default=40)
    main(ap.parse_args().n_test)

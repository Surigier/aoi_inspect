"""补齐单域成绩单:MVTec AD剩余8类(此前5类外的bottle/capsule/grid/screw/tile/
toothbrush/transistor/zipper)+ DAGM 2007 Class1(赛题原文点名的参考数据集,此前从
未进过任何成绩单)。口径与run_scorecard.py完全一致(同一个evaluate函数),
只是数据换成这9个此前没测过的单域,目的是在"混域拖累"之外,把系统单域真实能力的
证据面铺开——leon要的"拍多点单域"。

用法:PYTHONPATH=. python scripts/run_scorecard_extra.py
"""
import glob
import random
from pathlib import Path

import numpy as np
import torch

from aoi.imageio import load_fast
from scripts.run_scorecard import evaluate, _read, GT, HW

DAGM = Path("data/dagm/Class1")

MVTEC_EXTRA = [
    ("bottle", ["broken_large", "broken_small", "contamination"]),
    ("capsule", ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]),
    ("grid", ["bent", "broken", "glue", "metal_contamination", "thread"]),
    ("screw", ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]),
    ("tile", ["crack", "glue_strip", "gray_stroke", "oil", "rough"]),
    ("toothbrush", ["defective"]),
    ("transistor", ["bent_lead", "cut_lead", "damaged_case", "misplaced"]),
    ("zipper", ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"]),
]


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [(load_fast(p), None) for p in sorted(glob.glob(str(root / "test/good/*.png")))[:40]]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit_i = [load_fast(p) for p, _ in df[:k]]
    fit_m = [_read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW) for p, fo in df[:k]]
    test_defs = [(load_fast(p), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    return normals, fit_i, fit_m, test_defs, goods


def prep_dagm():
    """DAGM原生Train/Test各自混着正常+缺陷,按有无Label拆开(同make_exam_data.dagm_pick
    的逻辑,这里独立实现以直接产出run_scorecard.evaluate要的5元组)。"""
    def split(sub):
        normals, defs = [], []
        for f in sorted((DAGM / sub).glob("*.PNG")):
            m = DAGM / sub / "Label" / f"{f.stem}_label.PNG"
            (defs if m.exists() else normals).append((f, m if m.exists() else None))
        return normals, defs
    tr_n, tr_d = split("Train"); te_n, te_d = split("Test")
    normals, defs = tr_n + te_n, tr_d + te_d
    rng = random.Random(0); rng.shuffle(normals); rng.shuffle(defs)

    fit_normals = [load_fast(p) for p, _ in normals[:100]]
    goods = [(load_fast(p), None) for p, _ in normals[100:140]]
    fit_i = [load_fast(p) for p, _ in defs[:30]]
    fit_m = [_read(m, HW) for _, m in defs[:30]]
    test_defs = [(load_fast(p), _read(m, HW)) for p, m in defs[30:]]
    return fit_normals, fit_i, fit_m, test_defs, goods


def main():
    torch.manual_seed(0)
    print("=== 单域补齐成绩单(MVTec剩余8类 + DAGM 2007 Class1) ===")
    jobs = [(f"MVTec {cat}", lambda cat=cat, fo=fo: prep_mvtec(cat, fo)) for cat, fo in MVTEC_EXTRA]
    jobs.append(("DAGM Class1", prep_dagm))
    rows = []
    for name, prep in jobs:
        rows.append(evaluate(name, *prep()))
    a, g, p, h = (np.mean([r[i] for r in rows]) for i in range(4))
    print(f"\n均值(n={len(rows)}): 图级acc={a:.3f}  含漏检IoU={g:.3f}  纯定位IoU={p:.3f}  框命中@0.5={h:.3f}")


if __name__ == "__main__":
    main()

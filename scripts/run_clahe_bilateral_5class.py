"""CLAHE+双边滤波在5类生产成绩单(官方口径)上的验证。之前只在grid/screw(MVTec
补充类目)上测过,已经因screw误报率暴涨(27.5%→52.5%)判负;这里补齐在真正官方
评测类目上的完整数据,确认判负结论是否在生产5类上同样成立,供写文档时引用。

用法:PYTHONPATH=. python scripts/run_clahe_bilateral_5class.py
"""
import cv2
import numpy as np
import torch

from scripts.run_scorecard import evaluate, prep_mvtec, prep_realiad

JOBS = [
    ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
    ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
    ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
    ("电子 pcb", lambda: prep_realiad("pcb")),
    ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
]


def apply_clahe_bilateral(img_t):
    """同run_clahe_bilateral_ab.py:LAB空间CLAHE拉伸对比度→RGB空间双边滤波去噪。"""
    arr = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    rgb2 = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
    rgb3 = cv2.bilateralFilter(rgb2, d=5, sigmaColor=50, sigmaSpace=50)
    return torch.from_numpy(rgb3.astype(np.float32) / 255.0).permute(2, 0, 1)


def prep(name, raw_prep):
    normals, fit_i, fit_m, test_defs, goods = raw_prep()
    normals = [apply_clahe_bilateral(x) for x in normals]
    fit_i = [apply_clahe_bilateral(x) for x in fit_i]
    test_defs = [(apply_clahe_bilateral(img), gt) for img, gt in test_defs]
    goods = [(apply_clahe_bilateral(img), None) for img, _ in goods]
    return normals, fit_i, fit_m, test_defs, goods


def main():
    torch.manual_seed(0)
    print("=== CLAHE+双边滤波:5类生产成绩单(官方口径) ===")
    rows = []
    for name, raw_prep in JOBS:
        rows.append((name, evaluate(f"{name}(CLAHE+滤波)", *prep(name, raw_prep))))
    a, g, p, h = (np.mean([r[1][i] for r in rows]) for i in range(4))
    print(f"\n均值: 图级acc={a:.3f}  含漏检IoU={g:.3f}  纯定位IoU={p:.3f}  框命中@0.5={h:.3f}")


if __name__ == "__main__":
    main()

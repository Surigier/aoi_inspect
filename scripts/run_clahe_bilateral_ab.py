"""CLAHE+双边滤波A/B:leon看了纯CLAHE结果后问"能加滤波吗""太细微的可以剔除掉"——
纯CLAHE对metal_contamination/thread这两个原本表现好的类目框命中掉了0.12~0.14,
猜测是CLAHE拉伸对比度时把细粒度噪声也放大了。双边滤波(bilateralFilter)保边缘
去噪点,在CLAHE**之后**串一步,试试能不能把纯CLAHE的副作用滤掉、同时保住它对
glue这类低对比度缺陷的增益。

baseline仍是run_scorecard_by_defect_type.py的原图结果,对照组是run_clahe_ab.py
的纯CLAHE结果,这里是第三个变体:CLAHE→双边滤波。三者种子/切分完全一致
(prep_mvtec_typed内部random.Random(0)),唯一变量是预处理链路。

用法:PYTHONPATH=. python scripts/run_clahe_bilateral_ab.py
"""
import cv2
import numpy as np
import torch

from scripts.run_scorecard_by_defect_type import evaluate_typed, prep_mvtec_typed

CATS = [
    ("grid", ["bent", "broken", "glue", "metal_contamination", "thread"]),
    ("screw", ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]),
]


def apply_clahe_bilateral(img_t):
    """(3,H,W) float[0,1] tensor -> 同形状。LAB空间CLAHE拉伸对比度后,
    RGB空间双边滤波(d=5,sigmaColor=50,sigmaSpace=50)去掉被放大的细粒度噪声,
    同时保边缘(缺陷边界不会被抹平)。"""
    arr = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    rgb2 = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
    rgb3 = cv2.bilateralFilter(rgb2, d=5, sigmaColor=50, sigmaSpace=50)
    return torch.from_numpy(rgb3.astype(np.float32) / 255.0).permute(2, 0, 1)


def prep(cat, folders):
    normals, fit_i, fit_m, test_defs, goods = prep_mvtec_typed(cat, folders)
    normals = [apply_clahe_bilateral(x) for x in normals]
    fit_i = [apply_clahe_bilateral(x) for x in fit_i]
    test_defs = [(apply_clahe_bilateral(img), gt, st) for img, gt, st in test_defs]
    goods = [(apply_clahe_bilateral(img), None) for img, _ in goods]
    return normals, fit_i, fit_m, test_defs, goods


def main():
    torch.manual_seed(0)
    print("=== CLAHE+双边滤波A/B(grid/screw,vs run_scorecard_by_defect_type.py原图基线 / run_clahe_ab.py纯CLAHE) ===")
    for cat, folders in CATS:
        evaluate_typed(f"{cat}(CLAHE+双边滤波)", *prep(cat, folders))


if __name__ == "__main__":
    main()

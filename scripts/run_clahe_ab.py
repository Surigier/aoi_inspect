"""CLAHE对比度增强预处理的A/B验证:leon要求针对"低对比度/微小缺陷"失效模式
(grid的glue检出率0.333、screw的thread_side IoU0.152/框命中0.133)试一版CLAHE。

CLAHE只在图像进EAD/WRN之前做一次对比度拉伸(几毫秒开销,不改骨干/损失/训练配方),
fit和test两端必须统一处理(否则训练-测试分布不一致)。不生成任何新数据,只是把
已存在但弱的真实信号放大,风险比合成增广(CutPaste/TF-IDG)小。

baseline数字来自scripts/run_scorecard_by_defect_type.py已经跑出的grid/screw结果
(原图,无CLAHE),这里只跑CLAHE版本做背靠背对比,种子固定(prep_mvtec_typed内部
random.Random(0))与baseline的fit/test切分逐位一致,唯一变量是有无CLAHE。

用法:PYTHONPATH=. python scripts/run_clahe_ab.py
"""
import cv2
import numpy as np
import torch

from scripts.run_scorecard_by_defect_type import evaluate_typed, prep_mvtec_typed

CATS = [
    ("grid", ["bent", "broken", "glue", "metal_contamination", "thread"]),
    ("screw", ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]),
]


def apply_clahe(img_t):
    """(3,H,W) float[0,1] tensor -> 同形状,LAB空间对亮度通道做CLAHE。"""
    arr = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    rgb2 = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
    return torch.from_numpy(rgb2.astype(np.float32) / 255.0).permute(2, 0, 1)


def prep_clahe(cat, folders):
    normals, fit_i, fit_m, test_defs, goods = prep_mvtec_typed(cat, folders)
    normals = [apply_clahe(x) for x in normals]
    fit_i = [apply_clahe(x) for x in fit_i]
    test_defs = [(apply_clahe(img), gt, st) for img, gt, st in test_defs]
    goods = [(apply_clahe(img), None) for img, _ in goods]
    return normals, fit_i, fit_m, test_defs, goods


def main():
    torch.manual_seed(0)
    print("=== CLAHE对比度增强A/B(grid/screw,vs run_scorecard_by_defect_type.py的原图基线) ===")
    for cat, folders in CATS:
        evaluate_typed(f"{cat}(CLAHE)", *prep_clahe(cat, folders))


if __name__ == "__main__":
    main()

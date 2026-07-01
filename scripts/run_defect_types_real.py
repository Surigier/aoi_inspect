"""赛题5类缺陷 × 真实公开数据逐类验证(把"5类覆盖"从合成变真实证据)。
每类映射到 MVTec 真实缺陷文件夹,少样本迁移后报 检测image-AUROC + 定位pixel-AUROC。
尺寸偏差用几何代理(flip/misplaced)+诚实标注。用法:python scripts/run_defect_types_real.py
"""
import glob
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.efficientad import EfficientADDetector
from aoi.seg_head import SupervisedSegHead
from eval.protocol import image_auroc
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
EVAL_HW = (256, 256)

# 赛题5类 → (MVTec类别, [该类型缺陷文件夹], 是否真实/代理)
MAP = [
    ("外观缺陷", "hazelnut",   ["crack", "cut", "hole"],           "真实"),
    ("缺件少件", "cable",      ["missing_cable", "missing_wire"],  "真实"),
    ("逻辑顺序", "cable",      ["cable_swap"],                     "真实"),
    ("色彩变化", "pill",       ["color"],                          "真实"),
    ("尺寸偏差", "metal_nut",  ["flip"],                           "几何代理"),
]


def _mask(cat, folder, stem, hw):
    p = GT / cat / "ground_truth" / folder / (stem + "_mask.png")
    if not p.exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def run_type(name, cat, folders, kind):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "test/good/*.png")))]
    dfiles = []
    for fo in folders:
        dfiles += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(dfiles)
    if len(dfiles) < 8:
        print(f"{name}({cat}/{folders}): 缺陷不足({len(dfiles)})"); return None
    k = max(5, len(dfiles) // 3)
    fit_d = dfiles[:k]; test_d = dfiles[k:]

    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(normals, None)
    d_imgs = [_load_img(p, 320) for p, _ in fit_d]
    d_masks = [_mask(cat, fo, Path(p).stem, EVAL_HW) for p, fo in fit_d]
    head = SupervisedSegHead(device=DEV); head.fit(det, d_imgs, d_masks, normals[:30])

    # 检测 image-AUROC:good vs 该型缺陷(图级=像素图max)
    def img_score(img):
        return float(head.map(det, img, EVAL_HW).max())
    scores = [img_score(g) for g in goods] + [img_score(_load_img(p, 320)) for p, _ in test_d]
    labels = [0] * len(goods) + [1] * len(test_d)
    det_au = image_auroc(scores, labels)
    # 定位 pixel-AUROC
    S, L = [], []
    for p, fo in test_d:
        amap = head.map(det, _load_img(p, 320), EVAL_HW)
        S.append(amap.ravel()); L.append(_mask(cat, fo, Path(p).stem, EVAL_HW).ravel())
    for g in goods[:len(test_d)]:
        amap = head.map(det, g, EVAL_HW)
        S.append(amap.ravel()); L.append(np.zeros(EVAL_HW, np.uint8).ravel())
    pix_au = image_auroc(np.concatenate(S), np.concatenate(L))
    tag = "" if kind == "真实" else f" [{kind}]"
    print(f"{name:8s}{tag:8s} {cat}/{'+'.join(folders):20s} 检测AUROC={det_au:.3f} 定位AUROC={pix_au:.3f} "
          f"(n缺陷={len(test_d)})", flush=True)
    return det_au, pix_au


def main():
    torch.manual_seed(0)
    print("=== 赛题5类缺陷 × 真实公开数据逐类验证 ===")
    res = []
    for name, cat, folders, kind in MAP:
        r = run_type(name, cat, folders, kind)
        if r:
            res.append(r)
    if res:
        d = np.mean([x[0] for x in res]); p = np.mean([x[1] for x in res])
        print(f"\n5类均值: 检测AUROC={d:.3f}  定位AUROC={p:.3f}")


if __name__ == "__main__":
    main()

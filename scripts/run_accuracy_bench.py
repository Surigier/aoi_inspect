"""精度基准:python scripts/run_accuracy_bench.py [size] [coreset_ratio] [top_k]
在固定跨类别集合上测 纹理/结构/融合 的 image-AUROC,打印每类 + 均值。
固定随机种子以便公平对比各杠杆。"""
import sys
import random
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.fewshot import FewShotAdapter
from aoi.multibranch import MultiBranchAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category

# 跨表面/结构/逻辑/跨域的代表性类别(均为 MVTec 布局,load_category 通用)
CATEGORIES = [
    ("mvtec/bottle", "data/mvtec/bottle"),
    ("mvtec/metal_nut", "data/mvtec/metal_nut"),
    ("mvtec/screw", "data/mvtec/screw"),
    ("mvtec/transistor", "data/mvtec/transistor"),
    ("mpdd/metal_plate", "data/mpdd/metal_plate"),
    ("mpdd/tubes", "data/mpdd/tubes"),
    ("loco/breakfast_box", "data/_dl/mvtec_loco/breakfast_box"),
]


def _texture(bb, coreset, top_k):
    b = TextureADBranch(backbone=bb, coreset_ratio=coreset)
    if top_k is not None:
        b.top_k = top_k          # 若分支支持 top_k 打分
    return b


def main(size=256, coreset=0.25, top_k=None):
    torch.manual_seed(0)
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    sums = {"texture": 0.0, "structural": 0.0, "fused": 0.0}
    n = 0
    print(f"size={size} coreset={coreset} top_k={top_k}")
    for name, root in CATEGORIES:
        try:
            data = load_category(root, size=size)
        except Exception as e:
            print(f"{name:22s} 跳过({e})")
            continue
        if len(data["train_normal"]) == 0 or len(data["test_defect"]) < 6:
            print(f"{name:22s} 跳过(样本不足)")
            continue
        rng = random.Random(0)
        normals = data["train_normal"][:]
        defects = data["test_defect"][:]
        rng.shuffle(normals)
        rng.shuffle(defects)
        nfit = min(100, len(normals))
        dfit = min(30, len(defects) // 2)
        fn, fd = normals[:nfit], defects[:dfit]
        ti = data["test_normal"] + defects[dfit:]
        tl = [0] * len(data["test_normal"]) + [1] * len(defects[dfit:])
        res = {
            "texture": run_protocol(FewShotAdapter(_texture(bb, coreset, top_k)), fn, fd, ti, tl),
            "structural": run_protocol(FewShotAdapter(StructuralADBranch(backbone=bb, grid_size=16)), fn, fd, ti, tl),
            "fused": run_protocol(MultiBranchAdapter([_texture(bb, coreset, top_k), StructuralADBranch(backbone=bb, grid_size=16)]), fn, fd, ti, tl),
        }
        for k in sums:
            sums[k] += res[k]["auroc"]
        n += 1
        print(f"{name:22s} tex={res['texture']['auroc']:.3f} struct={res['structural']['auroc']:.3f} fused={res['fused']['auroc']:.3f}")
    print(f"{'MEAN':22s} tex={sums['texture']/n:.3f} struct={sums['structural']/n:.3f} fused={sums['fused']/n:.3f}  (n={n})")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    coreset = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(size, coreset, top_k)

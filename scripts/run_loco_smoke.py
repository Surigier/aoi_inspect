"""LOCO 逻辑异常冒烟:python scripts/run_loco_smoke.py data/_dl/mvtec_loco/breakfast_box
对比 纹理 / 结构 / 融合 在逻辑异常上的 AUROC。"""
import sys
import random
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.fewshot import FewShotAdapter
from aoi.multibranch import MultiBranchAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category


def main(root):
    data = load_category(root)
    rng = random.Random(0)
    normals = data["train_normal"][:]
    defects = data["test_defect"][:]
    rng.shuffle(normals)
    rng.shuffle(defects)
    fn = normals[:100]
    fd = defects[:30]
    ti = data["test_normal"] + defects[30:]
    tl = [0] * len(data["test_normal"]) + [1] * len(defects[30:])
    bb = Backbone(pretrained=True, device="cuda")
    configs = {
        "texture": FewShotAdapter(TextureADBranch(backbone=bb, coreset_ratio=0.25)),
        "structural": FewShotAdapter(StructuralADBranch(backbone=bb, grid_size=16)),
        "fused": MultiBranchAdapter([
            TextureADBranch(backbone=bb, coreset_ratio=0.25),
            StructuralADBranch(backbone=bb, grid_size=16),
        ]),
    }
    for name, ad in configs.items():
        m = run_protocol(ad, fn, fd, ti, tl)
        print(f"{name:11s} AUROC={m['auroc']:.3f} acc={m['accuracy']:.3f} lat={m['latency_ms_mean']:.0f}ms")


if __name__ == "__main__":
    main(sys.argv[1])

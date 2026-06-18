"""DAGM 纹理表面缺陷冒烟:python scripts/run_dagm_smoke.py data/dagm/Class1
DAGM 用 Test/Label/<id>_label.PNG 标记缺陷图。对比 纹理/结构 在该域的 AUROC。"""
import sys
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.fewshot import FewShotAdapter
from eval.protocol import run_protocol


def _load(p, size=256):
    img = Image.open(p).convert("RGB").resize((size, size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_dagm(root):
    """返回 (train_normal, test_normal, test_defect)。缺陷=Test 中有 _label 的图。"""
    root = Path(root)
    train_normal = [_load(p) for p in sorted((root / "Train").glob("*.PNG"))]
    defect_ids = {p.name.split("_")[0] for p in (root / "Test" / "Label").glob("*_label.PNG")}
    test_normal, test_defect = [], []
    for p in sorted((root / "Test").glob("*.PNG")):
        (test_defect if p.stem in defect_ids else test_normal).append(_load(p))
    return train_normal, test_normal, test_defect


def main(root):
    tn, te_n, te_d = load_dagm(root)
    rng = random.Random(0)
    rng.shuffle(tn)
    rng.shuffle(te_d)
    fn, fd = tn[:100], te_d[:30]
    ti = te_n + te_d[30:]
    tl = [0] * len(te_n) + [1] * len(te_d[30:])
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    for name, b in [("texture", TextureADBranch(backbone=bb, coreset_ratio=0.25)),
                    ("structural", StructuralADBranch(backbone=bb, grid_size=16))]:
        m = run_protocol(FewShotAdapter(b), fn, fd, ti, tl)
        print(f"{name:11s} AUROC={m['auroc']:.3f} acc={m['accuracy']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])

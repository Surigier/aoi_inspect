"""全量基准:python scripts/run_full_bench.py
用最佳配置(默认 320 / coreset0.25 / top_k10)跑 MVTec(15)+MPDD(6)+LOCO(5)+DAGM(10)
全部类别的 纹理 / 融合 image-AUROC,按数据集与总体出均值。种子固定可复现。"""
import os
import random
from collections import defaultdict
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.fewshot import FewShotAdapter
from aoi.multibranch import MultiBranchAdapter
from eval.protocol import run_protocol
from eval.mvtec import load_category, _load_img

SIZE = 320


def load_dagm(root):
    root = Path(root)
    tn = [_load_img(p, SIZE) for p in sorted((root / "Train").glob("*.PNG"))]
    did = {p.name.split("_")[0] for p in (root / "Test" / "Label").glob("*_label.PNG")}
    te_n, te_d = [], []
    for p in sorted((root / "Test").glob("*.PNG")):
        (te_d if p.stem in did else te_n).append(_load_img(p, SIZE))
    return {"train_normal": tn, "test_normal": te_n, "test_defect": te_d}


def collect():
    items = []
    for c in sorted(os.listdir("data/mvtec")):
        items.append(("mvtec", c, f"data/mvtec/{c}", load_category))
    for c in sorted(os.listdir("data/mpdd")):
        items.append(("mpdd", c, f"data/mpdd/{c}", load_category))
    loco = "data/_dl/mvtec_loco"
    for c in sorted(os.listdir(loco)):
        if os.path.isdir(f"{loco}/{c}") and not c.startswith("."):
            items.append(("loco", c, f"{loco}/{c}", load_category))
    for c in sorted(os.listdir("data/dagm")):
        items.append(("dagm", c, f"data/dagm/{c}", load_dagm))
    return items


def main():
    torch.manual_seed(0)
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for ds, c, root, loader in collect():
        try:
            data = loader(root)
        except Exception as e:
            print(f"{ds}/{c} skip ({e})")
            continue
        if len(data["train_normal"]) == 0 or len(data["test_defect"]) < 6:
            print(f"{ds}/{c} skip (insufficient)")
            continue
        rng = random.Random(0)
        nm, df = data["train_normal"][:], data["test_defect"][:]
        rng.shuffle(nm)
        rng.shuffle(df)
        nfit, dfit = min(100, len(nm)), min(30, len(df) // 2)
        fn, fd = nm[:nfit], df[:dfit]
        ti = data["test_normal"] + df[dfit:]
        tl = [0] * len(data["test_normal"]) + [1] * len(df[dfit:])
        try:
            tex = run_protocol(FewShotAdapter(TextureADBranch(backbone=bb)), fn, fd, ti, tl)["auroc"]
            fus = run_protocol(MultiBranchAdapter([TextureADBranch(backbone=bb), StructuralADBranch(backbone=bb, grid_size=16)]), fn, fd, ti, tl)["auroc"]
        except Exception as e:
            print(f"{ds}/{c} err ({e})")
            continue
        a = agg[ds]
        a[0] += tex
        a[1] += fus
        a[2] += 1
        print(f"{ds}/{c:18s} tex={tex:.3f} fused={fus:.3f}", flush=True)
    print("--- per-dataset mean ---")
    gt = gf = gn = 0
    for ds, (st, sf, n) in agg.items():
        print(f"{ds:8s} tex={st / n:.3f} fused={sf / n:.3f} (n={n})")
        gt += st
        gf += sf
        gn += n
    print(f"{'OVERALL':8s} tex={gt / gn:.3f} fused={gf / gn:.3f} (n={gn})")


if __name__ == "__main__":
    main()

"""全量基准(按竞赛规范):python scripts/run_full_bench.py
5 大族 ~48 类(MVTec15+MPDD6+LOCO5+DAGM10+VisA12),官方协议 100 正常+30 缺陷现场迁移→测剩余,
报融合的 AUROC / 准确率(平衡阈值)/ 单图延时,按数据族与总体汇总。种子固定可复现。"""
import os
import random
from collections import defaultdict
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.fewshot import FewShotAdapter
from aoi.ensemble import default_adapter
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


def load_visa(root):
    base = Path(root) / "Data" / "Images"
    normal = [_load_img(p, SIZE) for p in sorted((base / "Normal").glob("*.JPG"))]
    anomaly = [_load_img(p, SIZE) for p in sorted((base / "Anomaly").glob("*.JPG"))]
    random.Random(0).shuffle(normal)
    half = len(normal) // 2
    return {"train_normal": normal[:half], "test_normal": normal[half:], "test_defect": anomaly}


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
    visa = "data/visa"
    for c in sorted(os.listdir(visa)):
        if (Path(visa) / c / "Data" / "Images" / "Normal").is_dir():
            items.append(("visa", c, f"{visa}/{c}", load_visa))
    return items


def main():
    torch.manual_seed(0)
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])   # tex_au, fus_au, fus_acc, fus_lat, n
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
            m = run_protocol(default_adapter(bb), fn, fd, ti, tl)
        except Exception as e:
            print(f"{ds}/{c} err ({e})")
            continue
        a = agg[ds]
        a[0] += tex; a[1] += m["auroc"]; a[2] += m["accuracy"]; a[3] += m["latency_ms_mean"]; a[4] += 1
        print(f"{ds}/{c:18s} texAU={tex:.3f} fusAU={m['auroc']:.3f} acc={m['accuracy']:.3f} lat={m['latency_ms_mean']:.0f}ms", flush=True)
    print("--- 数据族均值(融合)---")
    g = [0.0, 0.0, 0.0, 0.0, 0]
    for ds, (ta, fa, ac, la, n) in agg.items():
        print(f"{ds:8s} texAU={ta/n:.3f} fusAU={fa/n:.3f} acc={ac/n:.3f} lat={la/n:.0f}ms (n={n})")
        for i in range(5):
            g[i] += [ta, fa, ac, la, n][i]
    print(f"{'OVERALL':8s} texAU={g[0]/g[4]:.3f} fusAU={g[1]/g[4]:.3f} acc={g[2]/g[4]:.3f} lat={g[3]/g[4]:.0f}ms (n={g[4]})")


if __name__ == "__main__":
    main()

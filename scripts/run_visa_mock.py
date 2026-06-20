"""VisA(真实电子件:PCB 等)模拟考:python scripts/run_visa_mock.py [cat ...]
默认跑 pcb1-4(最接近手机电子组件)。VisA 布局:<cat>/Data/Images/{Normal,Anomaly}/*.JPG。
按官方协议:100 正常 + 30 缺陷现场迁移 → 测剩余。"""
import sys
import random
from pathlib import Path
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from aoi.multibranch import MultiBranchAdapter
from eval.protocol import run_protocol
from eval.mvtec import _load_img

ROOT = "data/visa"
SIZE = 320


def load_visa(cat):
    base = Path(ROOT) / cat / "Data" / "Images"
    normal = [_load_img(p, SIZE) for p in sorted((base / "Normal").glob("*.JPG"))]
    anomaly = [_load_img(p, SIZE) for p in sorted((base / "Anomaly").glob("*.JPG"))]
    return normal, anomaly


def main(cats):
    bb = Backbone(pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
    sa = sacc = sl = 0.0
    n = 0
    for cat in cats:
        normal, anomaly = load_visa(cat)
        if len(normal) < 30 or len(anomaly) < 6:
            print(f"visa/{cat} skip (normal={len(normal)} anomaly={len(anomaly)})")
            continue
        rng = random.Random(0)
        rng.shuffle(normal)
        rng.shuffle(anomaly)
        nfit = min(100, len(normal) - 10)
        fn, fd = normal[:nfit], anomaly[:30]
        ti = normal[nfit:] + anomaly[30:]
        tl = [0] * len(normal[nfit:]) + [1] * len(anomaly[30:])
        ad = MultiBranchAdapter([TextureADBranch(backbone=bb), StructuralADBranch(backbone=bb, grid_size=16)])
        m = run_protocol(ad, fn, fd, ti, tl)
        print(f"visa/{cat:8s} AUROC={m['auroc']:.3f} acc={m['accuracy']:.3f} lat={m['latency_ms_mean']:.0f}ms (test {len(ti)})", flush=True)
        sa += m["auroc"]
        sacc += m["accuracy"]
        sl += m["latency_ms_mean"]
        n += 1
    if n:
        print(f"MEAN AUROC={sa / n:.3f} acc={sacc / n:.3f} lat={sl / n:.0f}ms (n={n})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pcb1", "pcb2", "pcb3", "pcb4"])

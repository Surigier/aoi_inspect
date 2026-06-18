"""延时基准:python scripts/run_benchmark.py data/mvtec/bottle
对各分支测单图端到端延时。"""
import sys
import torch
from aoi.backbone import Backbone
from aoi.branches.texture_ad import TextureADBranch
from aoi.branches.structural_ad import StructuralADBranch
from deploy.benchmark import benchmark_latency
from eval.mvtec import load_category


def main(root):
    data = load_category(root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(pretrained=True, device=device)
    normals = torch.stack(data["train_normal"][:50])
    probe = data["test_normal"][0].unsqueeze(0)
    branches = {
        "texture": TextureADBranch(backbone=bb, coreset_ratio=0.25),
        "structural": StructuralADBranch(backbone=bb, grid_size=16),
    }
    print(f"device={device}")
    for name, b in branches.items():
        b.fit(normals)
        m = benchmark_latency(b.infer, probe, runs=20, warmup=3)
        print(f"{name:11s} mean={m['mean_ms']:.1f}ms  p95={m['p95_ms']:.1f}ms")


if __name__ == "__main__":
    main(sys.argv[1])

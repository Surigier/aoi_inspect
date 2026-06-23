"""色彩分支分通道验证:python scripts/run_color_smoke.py
同一批正常图,分别施加 亮度变化 vs 颜色偏移,看 ColorADBranch 是否
只对颜色报警、对亮度鲁棒(坐实"用 Lab 色度而非亮度"的设计)。
对比 TextureADBranch(会把亮度也当异常)。"""
import glob
import random
import torch
from aoi.backbone import Backbone
from aoi.branches.color_ad import ColorADBranch
from aoi.branches.texture_ad import TextureADBranch
from eval.mvtec import _load_img


def perturb(img, kind):
    x = img.clone()
    if kind == "亮度×0.6":
        x = (x * 0.6).clamp(0, 1)
    elif kind == "亮度+0.2":
        x = (x + 0.2).clamp(0, 1)
    elif kind == "偏红+0.15":
        x[0] = (x[0] + 0.15).clamp(0, 1)
    elif kind == "偏黄(R,G+0.12)":
        x[0] = (x[0] + 0.12).clamp(0, 1)
        x[1] = (x[1] + 0.12).clamp(0, 1)
    return x


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cands = sorted(glob.glob("data/mvtec/pill/train/good/*.png")) or \
        sorted(glob.glob("data/mvtec/*/train/good/*.png"))
    rng.shuffle(cands)
    imgs = [_load_img(p, 320) for p in cands[:40]]
    fit, test = imgs[:30], imgs[30:38]

    color = ColorADBranch(grid_size=16)
    color.fit(torch.stack(fit))
    bb = Backbone(pretrained=True, device=dev)
    tex = TextureADBranch(backbone=bb, coreset_ratio=0.25)
    tex.fit(torch.stack(fit))

    def avg(branch, kind):
        ss = [branch.infer(perturb(t, kind).unsqueeze(0)).score for t in test]
        return sum(ss) / len(ss)

    print(f"base={cands[0].split('/')[-3]}")
    print(f"{'扰动':16s} {'色彩分支':>10} {'纹理分支':>10}")
    base_c = avg(color, "无"); base_t = avg(tex, "无")
    print(f"{'正常(基线)':16s} {base_c:10.3f} {base_t:10.3f}")
    for k in ["亮度×0.6", "亮度+0.2", "偏红+0.15", "偏黄(R,G+0.12)"]:
        c, t = avg(color, k), avg(tex, k)
        print(f"{k:16s} {c:10.3f}({c/base_c:.1f}x) {t:10.3f}({t/base_t:.1f}x)")
    print("\n期望:色彩分支 偏色↑↑ 亮度≈基线;纹理分支 亮度也会↑(分不清光照)")


if __name__ == "__main__":
    main()

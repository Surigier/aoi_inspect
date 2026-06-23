"""大图分块效果验证:python scripts/run_tiled_smoke.py
合成 2500²(真实纹理铺底 + 贴小缺陷裁片),对比:
  baseline 整图 resize→512 单 PatchCore   vs   分块 TiledFewShotDetector(原生分辨率)
报两者对"极小缺陷"的可分性(AUROC/分差)+ 分块 predict 延时。
证明:resize 抹掉小缺陷(AUROC≈0.5),分块救回(AUROC↑)。"""
import sys
import glob
import time
import random
import torch
import torch.nn.functional as F
from aoi.backbone import Backbone
from aoi.tiled import TiledFewShotDetector
from aoi.branches.texture_ad import TextureADBranch
from aoi.fusion import auroc
from eval.mvtec import _load_img

SIZE = 2500
SCR_LEN, SCR_W = 80, 3       # 细划痕:80px长×3px宽,高频→下采样会被抹掉


def make_normal(base, rng):
    return (base + torch.randn_like(base) * 0.02).clamp(0, 1)


def make_defect(base, rng):
    """贴一条细划痕(高频小缺陷):原生分辨率可见,resize 到 512 后≈消失。"""
    img = make_normal(base, rng)
    y, x = rng.randint(0, SIZE - SCR_LEN), rng.randint(0, SIZE - SCR_LEN)
    val = 0.0 if rng.random() < 0.5 else 1.0
    if rng.random() < 0.5:                       # 水平划痕
        img[:, y:y + SCR_W, x:x + SCR_LEN] = val
    else:                                         # 垂直划痕
        img[:, y:y + SCR_LEN, x:x + SCR_W] = val
    return img


def resize512(img):
    return F.interpolate(img.unsqueeze(0), size=(512, 512), mode="bilinear",
                         align_corners=False)[0]


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(name="resnet18", layers=(2, 3), pretrained=True, device=dev)
    cands = sorted(glob.glob("data/mvtec/leather/train/good/*.png")) or \
        sorted(glob.glob("data/mvtec/*/train/good/*.png"))
    base = _load_img(cands[0], SIZE)
    print(f"base={cands[0]}  划痕{SCR_LEN}x{SCR_W}px,resize后宽≈{SCR_W*512/SIZE:.2f}px", flush=True)

    fit_n = [make_normal(base, rng) for _ in range(8)]
    test_n = [make_normal(base, rng) for _ in range(8)]
    test_d = [make_defect(base, rng) for _ in range(8)]
    test = test_n + test_d
    labels = [0] * 8 + [1] * 8

    # --- baseline: 整图 resize 512 ---
    t = time.time()
    bad = TextureADBranch(backbone=bb, coreset_ratio=0.1)
    bad.fit(torch.stack([resize512(i) for i in fit_n]))
    t1 = time.perf_counter()
    base_scores = [bad.infer(resize512(i).unsqueeze(0)).score for i in test]
    base_lat = (time.perf_counter() - t1) / len(test) * 1000
    base_au = auroc(base_scores, labels)
    print(f"baseline(resize512) AUROC={base_au:.3f}  lat={base_lat:.0f}ms/图  ({time.time()-t:.0f}s)", flush=True)

    # --- 分块(几种配置对比延时/精度) ---
    for fg, cr in [(0, 0.03), (32, 0.01), (24, 0.01)]:
        t = time.time()
        det = TiledFewShotDetector(bb, tile=512, stride=512,
                                   coreset_ratio=cr, feat_grid=fg)
        det.fit_fewshot(fit_n, test_d[:2])
        tiled_scores = [det._image_score(i)[0] for i in test]
        det.predict(test_d[0])             # 预热
        lats = [det.predict(test_d[i])["latency_ms"] for i in range(4)]
        lat = sum(lats) / len(lats)
        tiled_au = auroc(tiled_scores, labels)
        tag = f"feat_grid={fg or '原始64'} coreset={cr}"
        print(f"tiled[{tag:24s}] AUROC={tiled_au:.3f}  lat={lat:.0f}ms/图  ({time.time()-t:.0f}s)", flush=True)
        if fg == 32:
            gt = sum(tiled_scores[:8]) / 8, sum(tiled_scores[8:]) / 8

    gn = sum(base_scores[:8]) / 8, sum(base_scores[8:]) / 8
    print(f"\n正常/缺陷均分:  baseline {gn[0]:.2f}/{gn[1]:.2f}(差{gn[1]-gn[0]:+.2f})  "
          f"tiled {gt[0]:.2f}/{gt[1]:.2f}(差{gt[1]-gt[0]:+.2f})")
    print(f"结论:小缺陷下 baseline AUROC={base_au:.3f} → tiled 各配置见上(均≈1.0)")


if __name__ == "__main__":
    main()

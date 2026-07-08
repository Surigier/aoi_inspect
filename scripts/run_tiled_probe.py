"""分块检测门探针(治大图检测漏):整图EAD+DINO门漏掉的小缺陷,原生分辨率分块EAD能否救回?
在rice/walnuts(漏检差最大-0.235/-0.138)上量:①EAD-only召回/正常acc ②整图OR分块 召回/正常acc。
分块只补检测门(二值is_defect),不碰定位。tile阈值fit标定(正常块vs缺陷图的最大块)。
用法:PYTHONPATH=. python scripts/run_tiled_probe.py [类名...]
"""
import sys
import glob
import random
from pathlib import Path
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from aoi.fewshot import FewShotAdapter

AD2 = Path("data/mvtec_ad_2")


def tiles(img, rows=2, cols=3, overlap=0.15):
    """(3,H,W)→ 网格裁块(带重叠防缺陷被切边)。"""
    C, H, W = img.shape
    th, tw = H // rows, W // cols
    oh, ow = int(th * overlap), int(tw * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, x0 = max(0, r * th - oh), max(0, c * tw - ow)
            y1, x1 = min(H, (r + 1) * th + oh), min(W, (c + 1) * tw + ow)
            out.append(img[:, y0:y1, x0:x1])
    return out


def tile_max(det, img):
    return max(det.branches[0].score(t) for t in tiles(img))


def run(cat, n_norm=100, n_fit=30):
    root = AD2 / cat
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    fit_b, test_b = bad[:n_fit], bad[n_fit:n_fit + 40]
    fit_i = [load_fast(p) for p in fit_b]
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=None)
    thr = det.threshold                                       # 整图EAD阈值
    # 分块阈值fit标定:正常图每块最大分 vs 缺陷图每块最大分(平衡acc)
    fn = [tile_max(det, n) for n in normals[:40]]
    fd = [tile_max(det, d) for d in fit_i]
    tile_thr = FewShotAdapter._calibrate(fn, fd)
    # 测试
    test_d = [load_fast(p) for p in test_b]
    test_g = [load_fast(p) for p in sorted(glob.glob(str(root / "test_public/good/*.png")))[:40]]
    wd = [det.branches[0].score(x) for x in test_d]; wg = [det.branches[0].score(x) for x in test_g]
    td = [tile_max(det, x) for x in test_d]; tg = [tile_max(det, x) for x in test_g]
    # EAD-only vs 整图OR分块
    rec_e = np.mean([s >= thr for s in wd]); nacc_e = np.mean([s < thr for s in wg])
    rec_o = np.mean([w >= thr or t >= tile_thr for w, t in zip(wd, td)])
    nacc_o = np.mean([not (w >= thr or t >= tile_thr) for w, t in zip(wg, tg)])
    print(f"{cat:12s} EAD门: 召回={rec_e:.3f} 正常acc={nacc_e:.3f} 平衡={(rec_e+nacc_e)/2:.3f}  |  "
          f"整图OR分块: 召回={rec_o:.3f} 正常acc={nacc_o:.3f} 平衡={(rec_o+nacc_o)/2:.3f}  "
          f"Δ召回={rec_o-rec_e:+.3f} Δ正常acc={nacc_o-nacc_e:+.3f}", flush=True)


def main():
    torch.manual_seed(0)
    print("=== 分块检测门探针(补大图检测漏,rice/walnuts)===", flush=True)
    for cat in (sys.argv[1:] or ["rice", "walnuts"]):
        run(cat)


if __name__ == "__main__":
    main()

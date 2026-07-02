"""大图定位特征输入自适应实验(AD2真大图):
zoom贴回负面后的正解假设:大图应提高特征输入(512→1024,格128²→256²),而非裁块。
变体:A load1152+seg512 / B load4096+seg512 / C load4096+seg1024。量纯定位逐图IoU+特征延时。
用法:python scripts/run_ad2_segres.py [cat]
"""
import sys
import glob
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aoi.backbone import Backbone
from aoi.seg_head import SupervisedSegHead
from aoi.efficientad import EfficientADDetector
from aoi.imageio import load_fast

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = (256, 256)


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def iou(pred, gt):
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def make_ex(bb, seg_in):
    @torch.no_grad()
    def ex(img):
        x = img.unsqueeze(0).to(bb.device) if img.dim() == 3 else img.to(bb.device)
        x = F.interpolate(x, size=(seg_in, seg_in), mode="bilinear", align_corners=False)
        return bb.extract(x)[0]
    return ex


def run(cat, load_sz, seg_in, bb):
    root = Path(f"data/mvtec_ad_2/{cat}")
    gn = sorted(glob.glob(str(root / "train/good/*.png")))
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png")))
    random.Random(0).shuffle(bad)
    def mpath(p):
        return str(root / "test_public/ground_truth/bad" / (Path(p).stem + "_mask.png"))
    normals = [load_fast(p, max_size=load_sz) for p in gn[:60]]
    fit_i = [load_fast(p, max_size=load_sz) for p in bad[:30]]
    fit_m = [_read(mpath(p), HW) for p in bad[:30]]
    ex = make_ex(bb, seg_in)
    head = SupervisedSegHead(device=DEV, extractor=ex)
    class _Dummy: pass
    head.fit(_Dummy(), fit_i, fit_m, normals[:30])
    thr = head.thr
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        ex(fit_i[0])
    torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 200
    ious = []
    for p in bad[30:]:
        img = load_fast(p, max_size=load_sz)
        amap = head.map(_Dummy(), img, HW)
        ious.append(iou(amap >= thr, _read(mpath(p), HW)))
    return np.mean(ious), lat


def main():
    torch.manual_seed(0)
    cat = sys.argv[1] if len(sys.argv) > 1 else "sheet_metal"
    bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)
    print(f"=== 大图特征输入自适应({cat},纯定位IoU,无门控)===")
    for tag, load_sz, seg_in in [("A load1152/seg512", 1152, 512),
                                 ("B load4096/seg512", 4096, 512),
                                 ("C load4096/seg1024", 4096, 1024),
                                 ("D load4096/seg1536", 4096, 1536)]:
        i, lat = run(cat, load_sz, seg_in, bb)
        print(f"{tag:20s} IoU={i:.3f}  特征延时={lat:.0f}ms", flush=True)


if __name__ == "__main__":
    main()

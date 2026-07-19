"""优先级5(TF-IDG生成增广)严格门控评审:用户规定三条全过才准入生产——
①3个不同切分的OOF平均IoU必须提升 ②最差类回退不超过0.01 ③真实样本占比≥50%(硬编码在此)。
CutPaste式简单合成已有负对照(-0.02~-0.07,seg_head.n_synth默认0),生成增广必须过同样严的门。

生成侧(租卡做,本机8GB跑不动ViT-G+SD2.1):
  git clone https://github.com/rubymiaomiao/TF-IDG   # ICCV2025官方代码,py3.8/torch2.0
  # 权重:AnyDoor epoch=1-step=8687.ckpt + dinov2_vitg14_pretrain.pth → ./checkpoint/
  # 每类用1张fit缺陷做参考,生成50-100对(不要上千),存 <pairs_dir>/<cat>/{imgs,masks}/
本机侧(此脚本):
  PYTHONPATH=. python scripts/run_tfidg_gate.py --pairs-dir data/_gen/tfidg
无--pairs-dir时对AD2三类跑"空增广自检"(real-only vs real-only,应零差异,验证套件本身无泄漏)。
"""
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from aoi.backbone import Backbone
from aoi.seg_head import SupervisedSegHead
from aoi.imageio import load_fast
from scripts.run_seg_head_ab import prep_ad2, _read, HW, SEG_IN

MAX_REAL_RATIO = 1.0        # aug数量≤real数量 → 真实占比≥50%(规格③)
WORST_REG = 0.01            # 最差类回退上限(规格②)


def load_pairs(pairs_dir, cat, hw):
    d = Path(pairs_dir) / cat
    imgs = sorted(glob.glob(str(d / "imgs" / "*")))
    out_i, out_m = [], []
    for p in imgs:
        mp = d / "masks" / (Path(p).stem + ".png")
        if not mp.exists():
            continue
        out_i.append(load_fast(p))
        out_m.append(_read(str(mp), hw))
    return out_i, out_m


def split_iou(extractor, dev, fit_i, fit_m, aug_i, aug_m, seed):
    """一个切分:real按seed分2/3训+1/3验;aug只进训练集(测试永远纯real)。"""
    n = len(fit_i)
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    k = max(2, n // 3)
    hold, tr = idx[:k], idx[k:]
    tr_i = [fit_i[i] for i in tr] + aug_i
    tr_m = [fit_m[i] for i in tr] + aug_m
    h = SupervisedSegHead(device=dev, extractor=extractor, seed=seed)
    ok = h.fit(None, tr_i, tr_m, [])
    if not ok or h.thr is None:
        return 0.0
    ious = []
    for i in hold:
        amap = h.map(None, fit_i[i], HW)
        pred = (amap >= h.thr)
        gt = fit_m[i]
        TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
        ious.append(TP / max(TP + FP + FN, 1))
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default=None, help="TF-IDG生成对根目录(<cat>/imgs,<cat>/masks)")
    ap.add_argument("--cats", default="sheet_metal,walnuts,fruit_jelly")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(layers=(1, 2), device=dev)

    @torch.no_grad()
    def extractor(img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(dev)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        return bb.extract(x)[0]

    deltas = {}
    for cat in args.cats.split(","):
        _, fit_i, fit_m, _ = prep_ad2(cat)
        if args.pairs_dir:
            aug_i, aug_m = load_pairs(args.pairs_dir, cat, HW)
            cap = int(len(fit_i) * MAX_REAL_RATIO)
            aug_i, aug_m = aug_i[:cap], aug_m[:cap]          # 规格③:真实占比≥50%
        else:
            aug_i, aug_m = [], []                            # 自检模式
        base, aug = [], []
        for seed in (0, 1, 2):                               # 规格①:3个不同切分
            base.append(split_iou(extractor, dev, fit_i, fit_m, [], [], seed))
            aug.append(split_iou(extractor, dev, fit_i, fit_m, aug_i, aug_m, seed))
        d = float(np.mean(aug) - np.mean(base))
        deltas[cat] = d
        print(f"{cat:14s} aug对数={len(aug_i)}  base(3切分均值)={np.mean(base):.3f}  "
              f"+aug={np.mean(aug):.3f}  Δ={d:+.3f}", flush=True)
    mean_d = float(np.mean(list(deltas.values())))
    worst = min(deltas.values())
    verdict = mean_d > 0 and worst >= -WORST_REG
    print(f"\n=== 门控判定 === Δ均值={mean_d:+.3f}(须>0) 最差类Δ={worst:+.3f}(须≥-{WORST_REG}) "
          f"→ {'✅准入' if verdict else '❌拒绝(维持real-only)'}", flush=True)


if __name__ == "__main__":
    main()

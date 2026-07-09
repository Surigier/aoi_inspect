"""多尺度特征融合探针(ETFA2026论文:layer2+3融合是AD2大图定位杠杆,非分辨率)。
隔离测监督seg-head的WRN层组合(1,2)/(2,3)/(1,2,3)对纯定位IoU+框的影响(不跑EAD,快)。
论文recipe=layer2+3上采样到layer2网格拼接;我们头是监督训练(无PatchCore延时)。
用法:PYTHONPATH=. python scripts/run_ad2_multiscale.py [类名...]
"""
import sys
import glob
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from aoi.backbone import Backbone
from aoi.imageio import load_fast
from scripts.run_scorecard import _read
from scripts.run_dino_locfuse import Lin, Cnv, gtb, hit_rate, f1_thr, iou

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = (256, 256); SEG_IN = 512
AD2 = Path("data/mvtec_ad_2")
CONFIGS = [(1, 2), (2, 3), (1, 2, 3)]


class Ex:
    def __init__(self, layers):
        self.bb = Backbone(layers=layers, pretrained=True, device=DEV)

    @torch.no_grad()
    def __call__(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        return self.bb.extract(x)                                       # (1,Cconcat,h,w)


def train_dual(feat, fit_i, fit_m, normals, steps=300):
    feats, gts = [], []
    with torch.no_grad():
        g0 = feat(fit_i[0]); h, w = g0.shape[-2:]
        for img, mk in zip(fit_i, fit_m):
            feats.append(feat(img))
            gts.append(torch.from_numpy(np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST))).float())
        for img in normals[:20]:
            feats.append(feat(img)); gts.append(torch.zeros(h, w))
    Fa = torch.cat(feats).to(DEV); Ga = torch.stack(gts).to(DEV)
    mu = Fa.mean(dim=(0, 2, 3), keepdim=True); sd = Fa.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    heads = []; N = Fa.shape[0]
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    for cls in (Lin, Cnv):
        head = cls(Fa.shape[1]).to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw); gg = torch.Generator().manual_seed(0)
        for _ in range(steps):
            sel = torch.randperm(N, generator=gg)[:8]
            opt.zero_grad(); lossf(head(((Fa[sel] - mu) / sd)).squeeze(1), Ga[sel]).backward(); opt.step()
        head.eval(); heads.append(head)
    return heads, mu, sd


def seg(feat, heads, mu, sd, img):
    f = feat(img); acc = None
    for head in heads:
        with torch.no_grad():
            lo = head((f - mu) / sd)
        acc = lo if acc is None else acc + lo
    return F.interpolate(acc / len(heads), size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def prep(cat, n_norm=60, n_fit=30):
    root = AD2 / cat
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    gt = root / "test_public/ground_truth/bad"
    def m(p):
        return _read(str(gt / (Path(p).stem + "_mask.png")), HW)
    fit_i = [load_fast(p) for p in bad[:n_fit]]; fit_m = [m(p) for p in bad[:n_fit]]
    test = [(load_fast(p), m(p)) for p in bad[n_fit:n_fit + 40]]
    return normals, fit_i, fit_m, test


def evaluate(cat):
    normals, fit_i, fit_m, test = prep(cat)
    fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for mk in fit_m]
    tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in test]
    tst_gt = [gtb(l) for l in tstL]
    out = {}
    for cfg in CONFIGS:
        feat = Ex(cfg)
        heads, mu, sd = train_dual(feat, fit_i, fit_m, normals)
        fitS = [seg(feat, heads, mu, sd, im) for im in fit_i]
        thr = f1_thr(fitS, fitL)
        tstS = [seg(feat, heads, mu, sd, im) for im, _ in test]
        ious = [iou(s >= thr, l) for s, l in zip(tstS, tstL)]
        bh = hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(tstS, tst_gt)])
        out[cfg] = (float(np.mean(ious)), bh)
    print(f"{cat:12s} " + "  ".join(f"{''.join(map(str,c))}:IoU={out[c][0]:.3f}/框{out[c][1]:.3f}" for c in CONFIGS), flush=True)
    return out


def main():
    torch.manual_seed(0)
    print("=== AD2大图 多尺度seg-head:layers(1,2) vs (2,3) vs (1,2,3)(纯定位IoU/框)===", flush=True)
    cats = sys.argv[1:] or ["sheet_metal", "wallplugs", "walnuts", "can"]
    agg = {c: [] for c in CONFIGS}
    for cat in cats:
        o = evaluate(cat)
        for c in CONFIGS:
            agg[c].append(o[c])
    print("\n均值:", flush=True)
    for c in CONFIGS:
        print(f"  {''.join(map(str,c)):6s} IoU={np.mean([x[0] for x in agg[c]]):.3f} 框={np.mean([x[1] for x in agg[c]]):.3f}", flush=True)


if __name__ == "__main__":
    main()

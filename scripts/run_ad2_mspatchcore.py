"""真正没试过的方向:多尺度PatchCore定位(ETFA2026,training-free最近邻,不过拟合)
   vs 我们监督(1,2)头。绕开"30掩膜过拟合"死结——PatchCore无训练头,多尺度对它直接受益。
冻结WRN50 layer2+3特征→coreset记忆库→per-patch最近邻距离图。掩膜只用于F1阈值(公平)。
用法:PYTHONPATH=. python scripts/run_ad2_mspatchcore.py [类名...]
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
HW = (256, 256); SEG_IN = 512; BANK = 2000
AD2 = Path("data/mvtec_ad_2")


class MSBank:
    """多尺度PatchCore:WRN50 layer2+3特征→coreset库→per-patch最近邻距离图(training-free)。"""
    def __init__(self, layers=(2, 3)):
        self.bb = Backbone(layers=layers, pretrained=True, device=DEV)
        self.bank = None; self.g = None

    @torch.no_grad()
    def _feat(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        f = self.bb.extract(x)[0]                              # (C,h,w)
        self.g = f.shape[-2:]
        return f.reshape(f.shape[0], -1).t()                  # (h*w, C)

    def build(self, normals):
        vs = [self._feat(n).cpu() for n in normals]
        V = torch.cat(vs); gg = torch.Generator().manual_seed(0)
        if V.shape[0] > BANK:
            V = V[torch.randperm(V.shape[0], generator=gg)[:BANK]]
        self.bank = F.normalize(V, dim=1).to(DEV)

    @torch.no_grad()
    def dmap(self, img):
        q = F.normalize(self._feat(img), dim=1); g = self.g
        d = []
        for i in range(0, q.shape[0], 4096):
            sim = q[i:i + 4096] @ self.bank.t()
            d.append(1 - sim.max(dim=1).values)
        dm = torch.cat(d).reshape(1, 1, g[0], g[1])
        return F.interpolate(dm, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


# ---- 监督(1,2)基线 ----
class BaseFeat:
    def __init__(self):
        self.bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)

    @torch.no_grad()
    def __call__(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        return self.bb.extract(x)


def train_base(feat, fit_i, fit_m, normals, steps=400):
    feats, gts = [], []
    with torch.no_grad():
        for img, mk in zip(fit_i, fit_m):
            f = feat(img); h, w = f.shape[-2:]; feats.append(f)
            gts.append(torch.from_numpy(np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST))).float())
        for img in normals[:20]:
            feats.append(feat(img)); gts.append(torch.zeros(feats[-1].shape[-2:]))
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


def seg_base(feat, heads, mu, sd, img):
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
    m = lambda p: _read(str(gt / (Path(p).stem + "_mask.png")), HW)
    fit_i = [load_fast(p) for p in bad[:n_fit]]; fit_m = [m(p) for p in bad[:n_fit]]
    test = [(load_fast(p), m(p)) for p in bad[n_fit:n_fit + 40]]
    return normals, fit_i, fit_m, test


def ev(cat, bf):
    normals, fit_i, fit_m, test = prep(cat)
    fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for mk in fit_m]
    tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in test]
    tgt = [gtb(l) for l in tstL]
    out = {}
    # 监督(1,2)
    h, mu, sd = train_base(bf, fit_i, fit_m, normals)
    fs = [seg_base(bf, h, mu, sd, im) for im in fit_i]; thr = f1_thr(fs, fitL)
    ts = [seg_base(bf, h, mu, sd, im) for im, _ in test]
    out["监督12"] = (float(np.mean([iou(s >= thr, l) for s, l in zip(ts, tstL)])),
                    hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(ts, tgt)]))
    # 多尺度PatchCore(掩膜仅标阈值)
    mb = MSBank(layers=(2, 3)); mb.build(normals)
    fs = [mb.dmap(im) for im in fit_i]; thr = f1_thr(fs, fitL)
    ts = [mb.dmap(im) for im, _ in test]
    out["MS-PC"] = (float(np.mean([iou(s >= thr, l) for s, l in zip(ts, tstL)])),
                    hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(ts, tgt)]))
    print(f"{cat:12s} 监督(1,2):IoU={out['监督12'][0]:.3f}/框{out['监督12'][1]:.3f}  "
          f"MS-PatchCore:IoU={out['MS-PC'][0]:.3f}/框{out['MS-PC'][1]:.3f}  "
          f"ΔIoU={out['MS-PC'][0]-out['监督12'][0]:+.3f} Δ框={out['MS-PC'][1]-out['监督12'][1]:+.3f}", flush=True)
    return out


def main():
    torch.manual_seed(0)
    print("=== 多尺度PatchCore(training-free) vs 监督(1,2)头 —— AD2大图严格IoU ===", flush=True)
    bf = BaseFeat()
    cats = sys.argv[1:] or ["sheet_metal", "wallplugs", "walnuts", "can"]
    agg = {"监督12": [], "MS-PC": []}
    for c in cats:
        o = ev(c, bf)
        for k in agg:
            agg[k].append(o[k])
    print("\n均值:", flush=True)
    for k in agg:
        print(f"  {k:8s} IoU={np.mean([x[0] for x in agg[k]]):.3f} 框={np.mean([x[1] for x in agg[k]]):.3f}", flush=True)


if __name__ == "__main__":
    main()

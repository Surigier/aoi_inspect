"""自创新:残差注意力多尺度融合头 RAMS-Head vs 当前(1,2)双头(AD2大图纯定位)。
上一枪粗暴concat多尺度判负=原始高维在30掩膜过拟合。RAMS-Head正解:
①逐尺度1×1降维(治过拟合,论文PCA的可学版)②残差空间注意力(逐像素逐尺度softmax权重,
网络自学"每处该信哪个尺度")③融合→双头logit,端到端训。治固定层组合"这类赢那类输"。
用法:PYTHONPATH=. python scripts/run_ad2_msattn.py [类名...]
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
SPLITS = (256, 512, 1024)                                    # WRN50 layer1/2/3 通道数


class MSFeat:
    """WRN50 layer1/2/3 原始特征(concat@128²),score时按通道切回三尺度。"""
    def __init__(self):
        self.bb = Backbone(layers=(1, 2, 3), pretrained=True, device=DEV)

    @torch.no_grad()
    def __call__(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        f = self.bb.extract(x)                                # (1, 1792, 128,128)
        return torch.split(f, SPLITS, dim=1)                  # (l1,l2,l3) 各@128²


class RAMSHead(nn.Module):
    """残差注意力多尺度融合:逐尺度降维d → 逐像素逐尺度softmax注意力 → 加权融合 → 卷积头。"""
    def __init__(self, d=48):
        super().__init__()
        self.red = nn.ModuleList([nn.Conv2d(c, d, 1) for c in SPLITS])   # 逐尺度1×1降维
        self.att = nn.Conv2d(d * 3, 3, 1)                                # 逐像素3尺度注意力logit
        self.head = nn.Sequential(nn.Conv2d(d, 64, 1), nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1))

    def forward(self, scales):
        r = [red(s) for red, s in zip(self.red, scales)]      # 3×(B,d,H,W)
        w = torch.softmax(self.att(torch.cat(r, 1)), dim=1)   # (B,3,H,W) 逐像素尺度权重
        fused = sum(w[:, i:i + 1] * r[i] for i in range(3))   # 残差注意力加权融合
        return self.head(fused)


def train_rams(feat, fit_i, fit_m, normals, steps=400):
    S, gts = [], []
    with torch.no_grad():
        for img, mk in zip(fit_i, fit_m):
            sc = feat(img); h, w = sc[0].shape[-2:]
            S.append([s.cpu() for s in sc])
            gts.append(torch.from_numpy(np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST))).float())
        for img in normals[:20]:
            sc = feat(img); h, w = sc[0].shape[-2:]
            S.append([s.cpu() for s in sc]); gts.append(torch.zeros(h, w))
    # 逐尺度标准化统计
    mus = [torch.cat([s[i] for s in S]).mean(dim=(0, 2, 3), keepdim=True).to(DEV) for i in range(3)]
    sds = [torch.cat([s[i] for s in S]).std(dim=(0, 2, 3), keepdim=True).add(1e-6).to(DEV) for i in range(3)]
    Ga = torch.stack(gts).to(DEV); N = len(S)
    head = RAMSHead().to(DEV)
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); gg = torch.Generator().manual_seed(0)
    for _ in range(steps):
        sel = torch.randperm(N, generator=gg)[:8].tolist()
        scales = [torch.cat([S[j][i] for j in sel]).to(DEV) for i in range(3)]
        scales = [(scales[i] - mus[i]) / sds[i] for i in range(3)]
        opt.zero_grad(); lossf(head(scales).squeeze(1), Ga[sel]).backward(); opt.step()
    head.eval()
    return head, mus, sds


def seg_rams(feat, head, mus, sds, img):
    sc = feat(img)
    scales = [(sc[i] - mus[i]) / sds[i] for i in range(3)]
    with torch.no_grad():
        lo = head(scales)
    return F.interpolate(lo, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


# ---- (1,2) 基线双头(对照)----
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


def ev(cat, bf, mf):
    normals, fit_i, fit_m, test = prep(cat)
    fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for mk in fit_m]
    tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in test]
    tgt = [gtb(l) for l in tstL]
    out = {}
    # 基线(1,2)
    h, mu, sd = train_base(bf, fit_i, fit_m, normals)
    fs = [seg_base(bf, h, mu, sd, im) for im in fit_i]; thr = f1_thr(fs, fitL)
    ts = [seg_base(bf, h, mu, sd, im) for im, _ in test]
    out["12"] = (float(np.mean([iou(s >= thr, l) for s, l in zip(ts, tstL)])),
                 hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(ts, tgt)]))
    # RAMS
    hh, mus, sds = train_rams(mf, fit_i, fit_m, normals)
    fs = [seg_rams(mf, hh, mus, sds, im) for im in fit_i]; thr = f1_thr(fs, fitL)
    ts = [seg_rams(mf, hh, mus, sds, im) for im, _ in test]
    out["RAMS"] = (float(np.mean([iou(s >= thr, l) for s, l in zip(ts, tstL)])),
                   hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(ts, tgt)]))
    print(f"{cat:12s} (1,2):IoU={out['12'][0]:.3f}/框{out['12'][1]:.3f}  "
          f"RAMS:IoU={out['RAMS'][0]:.3f}/框{out['RAMS'][1]:.3f}  "
          f"ΔIoU={out['RAMS'][0]-out['12'][0]:+.3f} Δ框={out['RAMS'][1]-out['12'][1]:+.3f}", flush=True)
    return out


def main():
    torch.manual_seed(0)
    print("=== 自创新 RAMS-Head(残差注意力多尺度) vs (1,2)双头 —— AD2大图纯定位 ===", flush=True)
    bf, mf = BaseFeat(), MSFeat()
    cats = sys.argv[1:] or ["sheet_metal", "wallplugs", "walnuts", "can"]
    agg = {"12": [], "RAMS": []}
    for c in cats:
        o = ev(c, bf, mf)
        for k in agg:
            agg[k].append(o[k])
    print("\n均值:", flush=True)
    for k in agg:
        print(f"  {k:6s} IoU={np.mean([x[0] for x in agg[k]]):.3f} 框={np.mean([x[1] for x in agg[k]]):.3f}", flush=True)


if __name__ == "__main__":
    main()

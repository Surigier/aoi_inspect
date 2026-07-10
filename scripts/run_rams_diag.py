"""诊断 RAMS 判负真因(回应质疑:"过拟合只是简单判断吧")。
三证据定案:①fit-IoU vs test-IoU gap(过拟合=fit高test低;优化失败=fit也低,可修)
②RAMS注意力权重分布(塌缩到单尺度=注意力没学起来)③3种子±std(EAD方差教训,单种子不下结论)。
外加真残差 RAMS-R:基线双头冻结 + 零初始化修正支(zero-conv式)→出发点≡基线,构造性不劣;
若它有增益=上枪是优化问题;若无=注意力确无信号可加。
用法:PYTHONPATH=. python scripts/run_rams_diag.py [类名...]
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
from scripts.run_dino_locfuse import Lin, Cnv, f1_thr, iou

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = (256, 256); SEG_IN = 512
SPLITS = (256, 512, 1024)
AD2 = Path("data/mvtec_ad_2")
SEEDS = [0, 1, 2]
STEPS = 400


class RAMSHead(nn.Module):
    def __init__(self, d=48, zero_last=False):
        super().__init__()
        self.red = nn.ModuleList([nn.Conv2d(c, d, 1) for c in SPLITS])
        self.att = nn.Conv2d(d * 3, 3, 1)
        self.head = nn.Sequential(nn.Conv2d(d, 64, 1), nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        if zero_last:                                          # zero-conv:修正支出发点=0
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, scales, ret_w=False):
        r = [red(s) for red, s in zip(self.red, scales)]
        w = torch.softmax(self.att(torch.cat(r, 1)), dim=1)
        fused = sum(w[:, i:i + 1] * r[i] for i in range(3))
        return (self.head(fused), w) if ret_w else self.head(fused)


def prep(cat, n_norm=60, n_fit=30):
    root = AD2 / cat
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png"))); random.Random(0).shuffle(bad)
    gt = root / "test_public/ground_truth/bad"
    m = lambda p: _read(str(gt / (Path(p).stem + "_mask.png")), HW)
    fit_i = [load_fast(p) for p in bad[:n_fit]]; fit_m = [m(p) for p in bad[:n_fit]]
    test = [(load_fast(p), m(p)) for p in bad[n_fit:n_fit + 30]]
    return normals, fit_i, fit_m, test


@torch.no_grad()
def extract(bb, imgs, to_cpu=False):
    """→ 3 个 (N,C,128,128) half 张量(GPU 或 CPU)。"""
    out = [[], [], []]
    for im in imgs:
        x = (im.unsqueeze(0) if im.dim() == 3 else im).to(DEV)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        s = torch.split(bb.extract(x), SPLITS, dim=1)
        for i in range(3):
            out[i].append(s[i].half().cpu() if to_cpu else s[i].half())
    return [torch.cat(o) for o in out]


def norm_stats(S):
    return ([s.float().mean(dim=(0, 2, 3), keepdim=True).to(DEV) for s in S],
            [s.float().std(dim=(0, 2, 3), keepdim=True).add(1e-6).to(DEV) for s in S])


def batch(S, idx, mus, sds):
    return [((S[i][idx].to(DEV).float() - mus[i]) / sds[i]) for i in range(3)]


def train_base(S, G, mus, sds, seed):
    """基线:concat(l1,l2) 双头(与生产同款)。"""
    torch.manual_seed(seed)
    heads = []
    N = G.shape[0]
    pw = torch.tensor([(G == 0).sum() / max(1, (G == 1).sum())], device=DEV)
    for cls in (Lin, Cnv):
        head = cls(SPLITS[0] + SPLITS[1]).to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw); gg = torch.Generator().manual_seed(seed)
        for _ in range(STEPS):
            sel = torch.randperm(N, generator=gg)[:8]
            xs = batch(S, sel, mus, sds)
            x12 = torch.cat(xs[:2], 1)
            opt.zero_grad(); lossf(head(x12).squeeze(1), G[sel]).backward(); opt.step()
        head.eval(); heads.append(head)
    return heads


def base_logit(heads, xs):
    x12 = torch.cat(xs[:2], 1)
    with torch.no_grad():
        return sum(h(x12) for h in heads) / len(heads)


def train_rams(S, G, mus, sds, seed):
    torch.manual_seed(seed)
    m = RAMSHead().to(DEV)
    N = G.shape[0]
    pw = torch.tensor([(G == 0).sum() / max(1, (G == 1).sum())], device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); gg = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        sel = torch.randperm(N, generator=gg)[:8]
        opt.zero_grad(); lossf(m(batch(S, sel, mus, sds)).squeeze(1), G[sel]).backward(); opt.step()
    m.eval()
    return m


def train_resid(S, G, mus, sds, seed, base_heads):
    """RAMS-R:基线头冻结,零初始化修正支,logit = base + corr。出发点≡基线。"""
    torch.manual_seed(seed)
    corr = RAMSHead(zero_last=True).to(DEV)
    N = G.shape[0]
    pw = torch.tensor([(G == 0).sum() / max(1, (G == 1).sum())], device=DEV)
    opt = torch.optim.Adam(corr.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); gg = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        sel = torch.randperm(N, generator=gg)[:8]
        xs = batch(S, sel, mus, sds)
        lo = base_logit(base_heads, xs) + corr(xs)
        opt.zero_grad(); lossf(lo.squeeze(1), G[sel]).backward(); opt.step()
    corr.eval()
    return corr


def eval_iou(fwd, S_fit, fitL, S_tst, tstL, mus, sds):
    """fwd(xs)->logit(1,1,h,w)。返回 fit_iou, test_iou(阈值F1标定于fit)。"""
    def maps(S, n):
        out = []
        for i in range(n):
            xs = batch(S, torch.tensor([i]), mus, sds)
            with torch.no_grad():
                lo = fwd(xs)
            out.append(F.interpolate(lo, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy())
        return out
    fm = maps(S_fit, len(fitL)); thr = f1_thr(fm, fitL)
    fit_iou = float(np.mean([iou(s >= thr, l) for s, l in zip(fm, fitL)]))
    tm = maps(S_tst, len(tstL))
    tst_iou = float(np.mean([iou(s >= thr, l) for s, l in zip(tm, tstL)]))
    return fit_iou, tst_iou


def run(cat):
    normals, fit_i, fit_m, test = prep(cat)
    bb = Backbone(layers=(1, 2, 3), pretrained=True, device=DEV)
    S_tr = extract(bb, fit_i + normals[:20])                   # 训练缓存(GPU half)
    S_fit = [s[:len(fit_i)] for s in S_tr]                     # 前30=fit缺陷(fit-IoU用)
    S_tst = extract(bb, [im for im, _ in test], to_cpu=True)   # 测试缓存(CPU half,流式)
    del bb; torch.cuda.empty_cache()
    fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for mk in fit_m]
    tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in test]
    g0 = S_tr[0].shape[-2:]
    G = torch.stack([torch.from_numpy(np.array(Image.fromarray(mk).resize((g0[1], g0[0]), Image.NEAREST))).float()
                     for mk in fit_m] + [torch.zeros(g0) for _ in range(20)]).to(DEV)
    mus, sds = norm_stats(S_tr)
    agg = {k: [] for k in ("base", "RAMS", "RAMS-R")}
    for seed in SEEDS:
        bh = train_base(S_tr, G, mus, sds, seed)
        rm = train_rams(S_tr, G, mus, sds, seed)
        rr = train_resid(S_tr, G, mus, sds, seed, bh)
        r_base = eval_iou(lambda xs: base_logit(bh, xs), S_fit, fitL, S_tst, tstL, mus, sds)
        r_rams = eval_iou(lambda xs: rm(xs), S_fit, fitL, S_tst, tstL, mus, sds)
        r_rr = eval_iou(lambda xs: base_logit(bh, xs) + rr(xs), S_fit, fitL, S_tst, tstL, mus, sds)
        # 注意力权重分布(test前10图均值)
        ws = []
        for i in range(min(10, len(tstL))):
            xs = batch(S_tst, torch.tensor([i]), mus, sds)
            with torch.no_grad():
                _, w = rm(xs, ret_w=True)
            ws.append(w.mean(dim=(0, 2, 3)).cpu().numpy())
        wbar = np.mean(ws, axis=0)
        agg["base"].append(r_base); agg["RAMS"].append(r_rams); agg["RAMS-R"].append(r_rr)
        print(f"{cat:12s} s{seed}  base fit/test={r_base[0]:.3f}/{r_base[1]:.3f}  "
              f"RAMS fit/test={r_rams[0]:.3f}/{r_rams[1]:.3f}  "
              f"RAMS-R fit/test={r_rr[0]:.3f}/{r_rr[1]:.3f}  "
              f"attnW=[{wbar[0]:.2f},{wbar[1]:.2f},{wbar[2]:.2f}]", flush=True)
    for k, v in agg.items():
        f = np.array([x[0] for x in v]); t = np.array([x[1] for x in v])
        print(f"  {cat} {k:7s} fit={f.mean():.3f}±{f.std():.3f}  test={t.mean():.3f}±{t.std():.3f}  gap={f.mean()-t.mean():+.3f}", flush=True)
    del S_tr, S_fit, S_tst; torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    print("=== RAMS 判负诊断:fit/test gap ×3种子 × 注意力分布 × 真残差RAMS-R ===", flush=True)
    for cat in (sys.argv[1:] or ["sheet_metal", "wallplugs"]):
        run(cat)


if __name__ == "__main__":
    main()

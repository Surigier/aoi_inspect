"""决定性验证:原生分辨率下 WRN纹理 vs DINO语义(纠正上版GRID=64把WRN降采样的假象)。
加 DINO高分辨率档(@1036→74²)测"DINO输是否纯因37²太粗"。每源用自己原生网格,
seg map统一上采样到256²算IoU。300步/40测试图(稳健口径)。
用法:PYTHONPATH=. python scripts/run_logic_native.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from aoi.backbone import Backbone
from scripts.run_logic_scorecard import prep_logic
from scripts.run_dino_locfuse import Lin, Cnv, gtb, hit_rate, f1_thr, iou

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = (256, 256); WRN_IN = 512


class Dino:
    def __init__(self, sz):
        self.sz = sz
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(self.sz, self.sz), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        t = self.m.forward_features(x)[:, self.m.num_prefix_tokens:, :]
        g = int(t.shape[1] ** 0.5)
        return t[0].permute(1, 0).reshape(1, -1, g, g).float()          # (1,C,g,g) 原生g


class Wrn:
    def __init__(self):
        self.bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)

    @torch.no_grad()
    def __call__(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(WRN_IN, WRN_IN), mode="bilinear", align_corners=False)
        return self.bb.extract(x)                                       # (1,C,128,128) 原生


def train_dual(feat, fit_i, fit_m, normals, steps=300):
    feats, gts = [], []
    with torch.no_grad():
        g0 = feat(fit_i[0]); h, w = g0.shape[-2:]                       # 该源原生网格
        for img, mk in zip(fit_i, fit_m):
            feats.append(feat(img))
            gts.append(torch.from_numpy(np.array(
                Image.fromarray(mk).resize((w, h), Image.NEAREST))).float())
        for img in normals[:20]:
            feats.append(feat(img)); gts.append(torch.zeros(h, w))
    Fa = torch.cat(feats).to(DEV); Ga = torch.stack(gts).to(DEV)
    mu = Fa.mean(dim=(0, 2, 3), keepdim=True); sd = Fa.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    heads = []
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    N = Fa.shape[0]
    for cls in (Lin, Cnv):
        head = cls(Fa.shape[1]).to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        gg = torch.Generator().manual_seed(0)
        for _ in range(steps):
            sel = torch.randperm(N, generator=gg)[:8]                   # minibatch=8 提速
            Xb = ((Fa[sel] - mu) / sd); Yb = Ga[sel]
            opt.zero_grad(); lossf(head(Xb).squeeze(1), Yb).backward(); opt.step()
        head.eval(); heads.append(head)
    return heads, mu, sd


def seg(feat, heads, mu, sd, img):
    f = feat(img); acc = None
    for head in heads:
        with torch.no_grad():
            lo = head((f - mu) / sd)
        acc = lo if acc is None else acc + lo
    return F.interpolate(acc / len(heads), size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def evaluate(name, srcs, normals, fit_i, fit_m, test):
    fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for mk in fit_m]
    tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in test]
    tst_gt = [gtb(l) for l in tstL]
    out = {}
    for tag, feat in srcs.items():
        heads, mu, sd = train_dual(feat, fit_i, fit_m, normals)
        fitS = [seg(feat, heads, mu, sd, im) for im in fit_i]
        thr = f1_thr(fitS, fitL)
        tstS = [seg(feat, heads, mu, sd, im) for im, _ in test]
        ious = [iou(s >= thr, l) for s, l in zip(tstS, tstL)]
        bh = hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(tstS, tst_gt)])
        out[tag] = (float(np.mean(ious)), bh)
    print(f"{name:20s} " + "  ".join(f"{t}={out[t][0]:.3f}/框{out[t][1]:.3f}" for t in srcs), flush=True)
    return out


def main():
    torch.manual_seed(0)
    print("=== 原生分辨率:WRN128² vs DINO37²(@518) vs DINO50²(@700)(纯定位IoU/框)===", flush=True)
    srcs = {"WRN": Wrn(), "DINO518": Dino(518), "DINO700": Dino(700)}
    agg = {k: [] for k in srcs}
    for cat in ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]:
        normals, fit_i, fit_m, test, _ = prep_logic(cat)
        o = evaluate(cat, srcs, normals, fit_i, fit_m, test)
        for k in srcs:
            agg[k].append(o[k])
    print("\n均值:")
    for k in srcs:
        print(f"  {k:9s} IoU={np.mean([x[0] for x in agg[k]]):.3f} 框={np.mean([x[1] for x in agg[k]]):.3f}")


if __name__ == "__main__":
    main()

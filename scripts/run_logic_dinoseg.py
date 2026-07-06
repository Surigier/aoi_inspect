"""逻辑缺陷定位探针(方向#2核心杠杆验证):同一监督分割头,对比特征源
  WRN50纹理(现状,对缺件/错序盲) vs DINO组件语义 vs 两者拼接。
直打定位真口径(纯定位IoU/框命中@0.5),隔离"特征能否定位逻辑缺陷"这一个问题。
用法:PYTHONPATH=. python scripts/run_logic_dinoseg.py
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
HW = (256, 256); WRN_IN = 512; DINO_SZ = 518; GRID = 64


class DinoEx:
    def __init__(self):
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(DINO_SZ, DINO_SZ), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        t = self.m.forward_features(x)[:, self.m.num_prefix_tokens:, :]  # (1,N,C)
        g = int(t.shape[1] ** 0.5)
        return t[0].permute(1, 0).reshape(1, -1, g, g).float()          # (1,C,g,g)


class WrnEx:
    def __init__(self):
        self.bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)

    @torch.no_grad()
    def __call__(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(WRN_IN, WRN_IN), mode="bilinear", align_corners=False)
        return self.bb.extract(x)                                       # (1,C,128,128)


def _grid(f):
    return F.interpolate(f, size=(GRID, GRID), mode="bilinear", align_corners=False)


def make_feat(kind, wrn, dino):
    if kind == "WRN":
        return lambda im: _grid(wrn(im))
    if kind == "DINO":
        return lambda im: _grid(dino(im))
    return lambda im: torch.cat([_grid(wrn(im)), _grid(dino(im))], dim=1)  # CAT


def train_dual(feat, fit_i, fit_m, normals, steps=300):
    feats, gts = [], []
    with torch.no_grad():
        for img, mk in zip(fit_i, fit_m):
            f = feat(img); feats.append(f)
            gts.append(torch.from_numpy(np.array(
                Image.fromarray(mk).resize((GRID, GRID), Image.NEAREST))).float())
        for img in normals[:20]:
            feats.append(feat(img)); gts.append(torch.zeros(GRID, GRID))
    Fa = torch.cat(feats).to(DEV); Ga = torch.stack(gts).to(DEV)
    mu = Fa.mean(dim=(0, 2, 3), keepdim=True); sd = Fa.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    heads = []
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    for cls in (Lin, Cnv):
        head = cls(Fa.shape[1]).to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw); torch.manual_seed(0)
        for _ in range(steps):
            opt.zero_grad(); lossf(head((Fa - mu) / sd).squeeze(1), Ga).backward(); opt.step()
        head.eval(); heads.append(head)
    return heads, mu, sd


def seg(feat, heads, mu, sd, img):
    f = feat(img); acc = None
    for head in heads:
        with torch.no_grad():
            lo = head((f - mu) / sd)
        acc = lo if acc is None else acc + lo
    return F.interpolate(acc / len(heads), size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def evaluate(name, wrn, dino, normals, fit_i, fit_m, test):
    fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for mk in fit_m]
    tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in test]
    tst_gt = [gtb(l) for l in tstL]
    out = {}
    for kind in ("WRN", "DINO", "CAT"):
        feat = make_feat(kind, wrn, dino)
        heads, mu, sd = train_dual(feat, fit_i, fit_m, normals)
        fitS = [seg(feat, heads, mu, sd, im) for im in fit_i]
        thr = f1_thr(fitS, fitL)
        tstS = [seg(feat, heads, mu, sd, im) for im, _ in test]
        ious = [iou(s >= thr, l) for s, l in zip(tstS, tstL)]
        bh = hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(tstS, tst_gt)])
        out[kind] = (float(np.mean(ious)), bh)
    print(f"{name:20s} WRN IoU={out['WRN'][0]:.3f}/框={out['WRN'][1]:.3f}  "
          f"DINO IoU={out['DINO'][0]:.3f}/框={out['DINO'][1]:.3f}  "
          f"CAT IoU={out['CAT'][0]:.3f}/框={out['CAT'][1]:.3f}", flush=True)
    return out


def main():
    torch.manual_seed(0)
    print("=== 逻辑缺陷定位:WRN纹理 vs DINO组件语义 vs 拼接(纯定位IoU/框@0.5)===")
    wrn, dino = WrnEx(), DinoEx()
    agg = {"WRN": [], "DINO": [], "CAT": []}
    for cat in ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]:
        normals, fit_i, fit_m, test, _ = prep_logic(cat)
        o = evaluate(cat, wrn, dino, normals, fit_i, fit_m, test)
        for k in agg:
            agg[k].append(o[k])
    print("\n均值:")
    for k in agg:
        iou_m = np.mean([x[0] for x in agg[k]]); bh_m = np.mean([x[1] for x in agg[k]])
        print(f"  {k:5s} IoU={iou_m:.3f} 框={bh_m:.3f}")


if __name__ == "__main__":
    main()

"""DINOv2距离图 × WRN50监督头 定位层融合(正确用法):
补检门失败但DINOv2记忆AUROC 0.88-1.0说明信号强。DINOv2 patch距离图=training-free
定位信号,与WRN50不同空间(上次WRN50-kNN同空间α=0失败,此次异空间)。
fused = seg_logit + α·z(dino_distmap),α在fit按框命中标定。量逐图IoU+框命中。
用法:python scripts/run_dino_locfuse.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from aoi.backbone import Backbone
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256); SEG_IN = 512; DINO_SZ = 518; BANK_MAX = 40000


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def gtb(mask, min_a=4):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in (st[i] for i in range(1, n)) if a >= min_a]


def biou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1)


def hit_rate(pairs):
    tot = hit = 0
    for preds, gts in pairs:
        for g in gts:
            tot += 1
            hit += any(biou(p, g) >= 0.5 for p in preds)
    return hit / max(tot, 1)


def iou(pred, gt):
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


class Lin(nn.Module):
    def __init__(s, C): super().__init__(); s.f = nn.Conv2d(C, 1, 1)
    def forward(s, x): return s.f(x)


class Cnv(nn.Module):
    def __init__(s, C, m=64):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(C, m, 1), nn.ReLU(True),
                            nn.Conv2d(m, m, 3, padding=1), nn.ReLU(True), nn.Conv2d(m, 1, 1))
    def forward(s, x): return s.f(x)


def wrn_ex(bb, img):
    x = (img.unsqueeze(0) if img.dim() == 3 else img).to(bb.device)
    x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
    with torch.no_grad():
        return bb.extract(x)


def train_head(bb, cls, fit, normals, steps=300):
    feats, gts = [], []
    with torch.no_grad():
        for img, mk in fit:
            f = wrn_ex(bb, img); h, w = f.shape[-2:]
            feats.append(f); gts.append(torch.from_numpy(
                np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST))).float())
        for img in normals[:20]:
            feats.append(wrn_ex(bb, img)); gts.append(torch.zeros(feats[-1].shape[-2:]))
    Fa = torch.cat(feats).to(DEV); Ga = torch.stack(gts).to(DEV)
    mu = Fa.mean(dim=(0, 2, 3), keepdim=True); sd = Fa.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    head = cls(Fa.shape[1]).to(DEV)
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    torch.manual_seed(0)
    for _ in range(steps):
        opt.zero_grad(); lossf(head((Fa - mu) / sd).squeeze(1), Ga).backward(); opt.step()
    head.eval()
    return head, mu, sd


def seg_map(bb, heads, img):
    f = wrn_ex(bb, img)
    acc = None
    for head, mu, sd in heads:
        with torch.no_grad():
            lo = head((f - mu) / sd)
        acc = lo if acc is None else acc + lo
    return F.interpolate(acc / len(heads), size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


class DinoDist:
    """DINOv2 patch距离图(training-free定位)。"""
    def __init__(self, normals):
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        vs = [self._p(n)[0] for n in normals]
        V = torch.cat(vs); g = torch.Generator().manual_seed(0)
        if V.shape[0] > BANK_MAX:
            V = V[torch.randperm(V.shape[0], generator=g)[:BANK_MAX]]
        self.bank = F.normalize(V, dim=1).half().to(DEV)

    @torch.no_grad()
    def _p(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(DINO_SZ, DINO_SZ), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        t = self.m.forward_features(x)[:, self.m.num_prefix_tokens:, :]
        n = t.shape[1]; g = int(n ** 0.5)
        return t[0].float().cpu(), g

    @torch.no_grad()
    def distmap(self, img):
        p, g = self._p(img)
        q = F.normalize(p.to(DEV), dim=1).half()
        d = []
        for i in range(0, q.shape[0], 2048):
            sim = q[i:i + 2048] @ self.bank.t()
            d.append(1 - sim.max(dim=1).values.float())
        dm = torch.cat(d).reshape(1, 1, g, g)
        return F.interpolate(dm, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def f1_thr(S, L):
    s = np.concatenate([x.ravel() for x in S]); l = np.concatenate([x.ravel() for x in L])
    o = np.argsort(-s); ls = l[o]; ss = s[o]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    f1 = 2*(tp/np.maximum(tp+fp,1))*(tp/P)/np.maximum(tp/np.maximum(tp+fp,1)+tp/P,1e-9)
    return float(ss[int(np.argmax(f1))])


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:60]]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit = [(_load_img(p, 640), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[:k]]
    tests = [(_load_img(p, 640), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    return normals, fit, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 640) for x in tok[:60]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit = [(_load_img(R / x["image_path"], 640), _read(R / x["mask_path"], HW)) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 640), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    return normals, fit, tests


def main():
    torch.manual_seed(0)
    bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)
    print("=== DINOv2距离图 × WRN50监督头 定位融合(IoU/框)===")
    for name, prep in [("pcb", lambda: prep_realiad("pcb")),
                       ("battery", lambda: prep_realiad("phone_battery")),
                       ("pill", lambda: prep_mvtec("pill", ["color"])),
                       ("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]))]:
        normals, fit, tests = prep()
        heads = [train_head(bb, Lin, fit, normals), train_head(bb, Cnv, fit, normals)]
        dino = DinoDist(normals[:40])
        dn = [dino.distmap(n) for n in normals[40:55]]
        dmu = float(np.mean([m.mean() for m in dn])); dsd = float(np.mean([m.std() for m in dn])) + 1e-6
        fitS = [seg_map(bb, heads, im) for im, _ in fit]
        fitD = [(dino.distmap(im) - dmu) / dsd for im, _ in fit]
        fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in fit]
        fit_gt = [gtb(l) for l in fitL]
        best_a, best_h = 0.0, -1
        for a in [0.0, 0.5, 1.0, 2.0, 4.0]:
            fu = [s + a * d for s, d in zip(fitS, fitD)]
            thr = f1_thr(fu, fitL)
            h = hit_rate([(gtb((x >= thr).astype(np.uint8), 3), g) for x, g in zip(fu, fit_gt)])
            if h > best_h:
                best_h, best_a = h, a
        tstS = [seg_map(bb, heads, im) for im, _ in tests]
        tstD = [(dino.distmap(im) - dmu) / dsd for im, _ in tests]
        tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in tests]
        tst_gt = [gtb(l) for l in tstL]
        out = {}
        for tag, a in [("监督单路", 0.0), (f"+DINO(α={best_a})", best_a)]:
            fu_f = [s + a * d for s, d in zip(fitS, fitD)]
            thr = f1_thr(fu_f, fitL)
            fu = [s + a * d for s, d in zip(tstS, tstD)]
            ious = [iou(x >= thr, l) for x, l in zip(fu, tstL)]
            bh = hit_rate([(gtb((x >= thr).astype(np.uint8), 3), g) for x, g in zip(fu, tst_gt)])
            out[tag] = (np.mean(ious), bh)
        (t0, v0), (t1, v1) = list(out.items())
        print(f"{name:10s} {t0}:IoU={v0[0]:.3f}/框={v0[1]:.3f}  {t1}:IoU={v1[0]:.3f}/框={v1[1]:.3f}  "
              f"ΔIoU={v1[0]-v0[0]:+.3f} Δ框={v1[1]-v0[1]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

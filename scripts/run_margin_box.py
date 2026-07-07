"""对症提框(按诊断:过小38%+全漏29%):
B 加法边距(框各边外扩m像素,fit标定m;小框相对受益大→治'过小',乘法膨胀已证无效)
C B+次阈值峰值救援框(top-3峰+先验中位尺寸;治'全漏',额外框数受控)
用法:python scripts/run_margin_box.py
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
from PIL import Image
from aoi.backbone import Backbone
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256); SEG_IN = 512


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


class Lin(nn.Module):
    def __init__(s, C): super().__init__(); s.f = nn.Conv2d(C, 1, 1)
    def forward(s, x): return s.f(x)


class Cnv(nn.Module):
    def __init__(s, C, m=64):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(C, m, 1), nn.ReLU(True),
                            nn.Conv2d(m, m, 3, padding=1), nn.ReLU(True), nn.Conv2d(m, 1, 1))
    def forward(s, x): return s.f(x)


def extract(bb, img):
    x = (img.unsqueeze(0) if img.dim() == 3 else img).to(bb.device)
    x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
    with torch.no_grad():
        return bb.extract(x)


def train(bb, cls, fit, normals, steps=300):
    feats, gts = [], []
    with torch.no_grad():
        for img, mk in fit:
            f = extract(bb, img); h, w = f.shape[-2:]
            feats.append(f); gts.append(torch.from_numpy(
                np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST))).float())
        for img in normals[:20]:
            feats.append(extract(bb, img)); gts.append(torch.zeros(feats[-1].shape[-2:]))
    Fa = torch.cat(feats).to(DEV); Ga = torch.stack(gts).to(DEV)
    mu = Fa.mean(dim=(0, 2, 3), keepdim=True); sd = Fa.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    head = cls(Fa.shape[1]).to(DEV)
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    torch.manual_seed(0)
    N = Fa.shape[0]; gg = torch.Generator().manual_seed(0)
    for _ in range(steps):
        sel = torch.randperm(N, generator=gg)[:8]              # minibatch=8(避免大特征图全批训练卡死)
        opt.zero_grad(); lossf(head(((Fa[sel]-mu)/sd)).squeeze(1), Ga[sel]).backward(); opt.step()
    head.eval()
    return head, mu, sd


def amap_ens(bb, heads, img):
    f = extract(bb, img)
    acc = None
    for head, mu, sd in heads:
        with torch.no_grad():
            lo = head((f - mu) / sd)
        acc = lo if acc is None else acc + lo
    return F.interpolate(acc / len(heads), size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def f1_thr(S, L):
    s = np.concatenate([x.ravel() for x in S]); l = np.concatenate([x.ravel() for x in L])
    o = np.argsort(-s); ls = l[o]; ss = s[o]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    f1 = 2*(tp/np.maximum(tp+fp,1))*(tp/P)/np.maximum(tp/np.maximum(tp+fp,1)+tp/P,1e-9)
    return float(ss[int(np.argmax(f1))])


def boxes_mask(amap, thr):
    return gtb((amap >= thr).astype(np.uint8), min_a=3)


def add_margin(boxes, m, W=256, H=256):
    return [(max(0, x1-m), max(0, y1-m), min(W, x2+m), min(H, y2+m)) for x1, y1, x2, y2 in boxes]


def rescue_peaks(amap, thr, existing, prior, k=3, floor_off=2.0):
    """次阈值(thr-floor_off ~ thr)峰值→先验中位尺寸框(不与已有框重叠)。治'全漏'。"""
    if prior is None:
        return []
    mw, mh = prior
    dil = cv2.dilate(amap, np.ones((7, 7), np.float32))
    cand = np.argwhere((amap >= dil - 1e-6) & (amap >= thr - floor_off) & (amap < thr))
    if len(cand) == 0:
        return []
    vals = amap[cand[:, 0], cand[:, 1]]
    out = []
    for oi in np.argsort(-vals):
        y, x = cand[oi]
        b = (max(0, x - mw / 2), max(0, y - mh / 2), min(256, x + mw / 2), min(256, y + mh / 2))
        if any(biou(b, e) > 0.3 for e in existing + out):
            continue
        out.append(b)
        if len(out) >= k:
            break
    return out


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
    print("=== 加法边距 + 救援框(对症'过小'与'全漏')===")
    A, B, C = [], [], []
    for name, prep in [("pcb", lambda: prep_realiad("pcb")),
                       ("battery", lambda: prep_realiad("phone_battery")),
                       ("pill", lambda: prep_mvtec("pill", ["color"])),
                       ("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]))]:
        normals, fit, tests = prep()
        heads = [train(bb, Lin, fit, normals), train(bb, Cnv, fit, normals)]
        fitS = [amap_ens(bb, heads, im) for im, _ in fit]
        fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in fit]
        thr = f1_thr(fitS, fitL)
        fit_gt = [gtb(l) for l in fitL]
        all_gt = [b for g in fit_gt for b in g]
        prior = (float(np.median([b[2]-b[0] for b in all_gt])), float(np.median([b[3]-b[1] for b in all_gt]))) if all_gt else None
        # fit标定加法边距 m
        best_m, best_h = 0, -1
        for m in [0, 2, 4, 6, 8]:
            h = hit_rate([(add_margin(boxes_mask(s, thr), m), g) for s, g in zip(fitS, fit_gt)])
            if h > best_h:
                best_h, best_m = h, m
        tstS = [amap_ens(bb, heads, im) for im, _ in tests]
        tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in tests]
        tst_gt = [gtb(l) for l in tstL]
        a = hit_rate([(boxes_mask(s, thr), g) for s, g in zip(tstS, tst_gt)])
        bm = [add_margin(boxes_mask(s, thr), best_m) for s in tstS]
        b = hit_rate(list(zip(bm, tst_gt)))
        cm = [m0 + rescue_peaks(s, thr, m0, prior) for s, m0 in zip(tstS, bm)]
        c = hit_rate(list(zip(cm, tst_gt)))
        nb = np.mean([len(x) for x in cm])
        A.append(a); B.append(b); C.append(c)
        print(f"{name:10s} A现状={a:.3f}  B+边距(m={best_m})={b:.3f}  C+救援={c:.3f}  (Δ={c-a:+.3f}, 框/图={nb:.1f})", flush=True)
    print(f"\n均值: A={np.mean(A):.3f}  B={np.mean(B):.3f}  C={np.mean(C):.3f}  (总Δ={np.mean(C)-np.mean(A):+.3f})")


if __name__ == "__main__":
    main()

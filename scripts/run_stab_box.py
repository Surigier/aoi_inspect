"""稳定化实验(现有成熟方法):
A 单头自动选(现状) B 双头logit集成(WBF思想:组合优于选择,消选择方差)
C B+滞后阈值出框(Canny式:高阈找种子/低阈生长→框覆盖全) D C+TTA(翻转平均)。
量纯定位逐图IoU + 框命中@0.5。用法:python scripts/run_stab_box.py
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


def iou(pred, gt):
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


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
    Fn = (Fa - mu) / sd
    head = cls(Fa.shape[1]).to(DEV)
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    torch.manual_seed(0)
    for _ in range(steps):
        opt.zero_grad(); lossf(head(Fn).squeeze(1), Ga).backward(); opt.step()
    head.eval()
    return head, mu, sd


def amap_fn(bb, heads, img, tta=False):
    """heads: [(head,mu,sd)...] → 平均logit图(256²)。tta: +水平/垂直翻转平均。"""
    views = [img]
    if tta:
        views += [torch.flip(img, [-1]), torch.flip(img, [-2])]
    outs = []
    for vi, v in enumerate(views):
        f = extract(bb, v)
        acc = None
        for head, mu, sd in heads:
            with torch.no_grad():
                lo = head((f - mu) / sd)
            acc = lo if acc is None else acc + lo
        lo = acc / len(heads)
        lo = F.interpolate(lo, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        if vi == 1: lo = lo[:, ::-1]
        if vi == 2: lo = lo[::-1, :]
        outs.append(lo)
    return np.mean(outs, axis=0)


def f1_thr(S, L):
    s = np.concatenate([x.ravel() for x in S]); l = np.concatenate([x.ravel() for x in L])
    o = np.argsort(-s); ls = l[o]; ss = s[o]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    f1 = 2*(tp/np.maximum(tp+fp,1))*(tp/P)/np.maximum(tp/np.maximum(tp+fp,1)+tp/P,1e-9)
    return float(ss[int(np.argmax(f1))])


def boxes_plain(amap, thr):
    return gtb((amap >= thr).astype(np.uint8), min_a=3)


def boxes_hyst(amap, thr_hi, delta):
    """滞后:高阈种子+低阈生长,保留含种子的低阈连通域。"""
    seed = (amap >= thr_hi).astype(np.uint8)
    grow = (amap >= thr_hi - delta).astype(np.uint8)
    n, lab = cv2.connectedComponents(grow, connectivity=8)
    keep = np.zeros_like(grow)
    ids = np.unique(lab[seed > 0])
    for i in ids:
        if i != 0:
            keep[lab == i] = 1
    return gtb(keep, min_a=3)


def run_cat(name, normals, fit, tests, bb):
    lin = train(bb, Lin, fit, normals)
    cnv = train(bb, Cnv, fit, normals)
    ens = [lin, cnv]
    # fit上标定:各配置阈值(集成图)、滞后delta
    fitS = [amap_fn(bb, ens, im) for im, _ in fit]
    fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in fit]
    thr_e = f1_thr(fitS, fitL)
    fit_gt = [gtb(l) for l in fitL]
    best_d, best_h = 0.0, -1
    for d in [0.0, 0.5, 1.0, 2.0]:
        h = hit_rate([(boxes_hyst(s, thr_e, d), g) for s, g in zip(fitS, fit_gt)])
        if h > best_h:
            best_h, best_d = h, d
    # 单头(线性,近似现状主路径)
    linS = [amap_fn(bb, [lin], im) for im, _ in fit]
    thr_l = f1_thr(linS, fitL)
    res = {}
    tst_gt = [gtb(np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST))) for _, mk in tests]
    tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in tests]
    for tag, heads, thr, hyst_d, tta in [
        ("A单头", [lin], thr_l, 0.0, False),
        ("B集成", ens, thr_e, 0.0, False),
        ("C+滞后", ens, thr_e, best_d, False),
        ("D+TTA", ens, thr_e, best_d, True),
    ]:
        S = [amap_fn(bb, heads, im, tta) for im, _ in tests]
        ious = [iou(s >= thr, l) for s, l in zip(S, tstL)]
        bx = [(boxes_hyst(s, thr, hyst_d) if hyst_d > 0 else boxes_plain(s, thr)) for s in S]
        res[tag] = (np.mean(ious), hit_rate(list(zip(bx, tst_gt))))
    line = "  ".join(f"{t}:IoU={v[0]:.3f}/框={v[1]:.3f}" for t, v in res.items())
    print(f"{name:12s} {line}  [δ={best_d}]", flush=True)
    return res


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
    print("=== 稳定化:单头 vs 集成 vs +滞后阈值 vs +TTA(IoU/框命中)===")
    agg = {}
    for name, prep in [("pcb", lambda: prep_realiad("pcb")),
                       ("battery", lambda: prep_realiad("phone_battery")),
                       ("pill", lambda: prep_mvtec("pill", ["color"])),
                       ("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]))]:
        normals, fit, tests = prep()
        r = run_cat(name, normals, fit, tests, bb)
        for t, v in r.items():
            agg.setdefault(t, []).append(v)
    print("\n均值: " + "  ".join(f"{t}:IoU={np.mean([x[0] for x in v]):.3f}/框={np.mean([x[1] for x in v]):.3f}"
                                for t, v in agg.items()))


if __name__ == "__main__":
    main()

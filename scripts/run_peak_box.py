"""框失败诊断 + 逐峰自适应生长提框(分水岭思想):
诊断:每个未命中GT框归类(全漏/过小/过大/偏移)→ 指明分数流失在哪。
新提框:局部峰值(各自水平生长 peak-δ)→ 逐实例自适应阈值,治全局单阈值的
"弱缺陷整漏/强缺陷过割"。δ与峰值门槛在fit集标定。对比现有掩膜连通域提框。
用法:python scripts/run_peak_box.py
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
    for _ in range(steps):
        opt.zero_grad(); lossf(head((Fa - mu) / sd).squeeze(1), Ga).backward(); opt.step()
    head.eval()
    return head, mu, sd


def amap_ens(bb, heads, img):
    f = extract(bb, img)
    acc = None
    for head, mu, sd in heads:
        with torch.no_grad():
            lo = head((f - mu) / sd)
        acc = lo if acc is None else acc + lo
    lo = acc / len(heads)
    return F.interpolate(lo, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def f1_thr(S, L):
    s = np.concatenate([x.ravel() for x in S]); l = np.concatenate([x.ravel() for x in L])
    o = np.argsort(-s); ls = l[o]; ss = s[o]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    f1 = 2*(tp/np.maximum(tp+fp,1))*(tp/P)/np.maximum(tp/np.maximum(tp+fp,1)+tp/P,1e-9)
    return float(ss[int(np.argmax(f1))])


def boxes_mask(amap, thr):
    return gtb((amap >= thr).astype(np.uint8), min_a=3)


def boxes_peak(amap, floor, delta, max_peaks=8):
    """局部峰值→各自水平(peak-δ)生长连通域→框。逐实例自适应,弱缺陷不再被全局阈值漏。"""
    dil = cv2.dilate(amap, np.ones((5, 5), np.float32))
    peaks = np.argwhere((amap >= dil - 1e-6) & (amap >= floor))
    if len(peaks) == 0:
        return []
    vals = amap[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(-vals)[:max_peaks * 4]
    boxes = []
    for oi in order:
        y, x = peaks[oi]; pv = vals[oi]
        grow = (amap >= pv - delta).astype(np.uint8)
        n, lab = cv2.connectedComponents(grow, connectivity=8)
        c = lab[y, x]
        if c == 0:
            continue
        ys, xs = np.where(lab == c)
        b = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if not any(biou(b, e) > 0.5 for e in boxes):
            boxes.append(b)
        if len(boxes) >= max_peaks:
            break
    return boxes


def diagnose(preds_list, gts_list):
    """每个未命中GT框归类:全漏(0重叠)/过小/过大/偏移。"""
    cats = {"命中": 0, "全漏": 0, "过小": 0, "过大": 0, "偏移": 0}
    for preds, gts in zip(preds_list, gts_list):
        for g in gts:
            ious = [biou(p, g) for p in preds] or [0]
            best = max(ious)
            if best >= 0.5:
                cats["命中"] += 1; continue
            if best == 0:
                cats["全漏"] += 1; continue
            p = preds[int(np.argmax(ious))]
            ga = (g[2]-g[0])*(g[3]-g[1]); pa = (p[2]-p[0])*(p[3]-p[1])
            if pa < ga * 0.5:
                cats["过小"] += 1
            elif pa > ga * 2:
                cats["过大"] += 1
            else:
                cats["偏移"] += 1
    return cats


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
    print("=== 框失败诊断 + 逐峰生长提框 ===")
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
        # fit标定 peak 参数
        best, cfg = -1, (thr - 2, 1.0)
        for floor_off in [1.0, 2.0, 3.0]:
            for delta in [0.5, 1.0, 2.0]:
                h = hit_rate([(boxes_peak(s, thr - floor_off, delta), g) for s, g in zip(fitS, fit_gt)])
                if h > best:
                    best, cfg = h, (thr - floor_off, delta)
        floor, delta = cfg
        tstS = [amap_ens(bb, heads, im) for im, _ in tests]
        tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in tests]
        tst_gt = [gtb(l) for l in tstL]
        bm = [boxes_mask(s, thr) for s in tstS]
        bp = [boxes_peak(s, floor, delta) for s in tstS]
        hm = hit_rate(list(zip(bm, tst_gt))); hp = hit_rate(list(zip(bp, tst_gt)))
        dg = diagnose(bm, tst_gt)
        tot = sum(dg.values())
        dgs = " ".join(f"{k}={v}({v/max(tot,1)*100:.0f}%)" for k, v in dg.items())
        print(f"{name:10s} 掩膜框={hm:.3f} → 逐峰框={hp:.3f} (Δ{hp-hm:+.3f})  [诊断(掩膜框): {dgs}]", flush=True)


if __name__ == "__main__":
    main()

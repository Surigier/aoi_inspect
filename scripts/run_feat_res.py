"""定位优化:特征分辨率×层选择扫描(直击微小缺陷"特征格比缺陷粗"根因)。
配置:输入{320,512,640} × 层{(2,3),(1,2)},特征格40²→128²。
量:逐图IoU(监督F1阈值,生产口径)+特征延时。用法:python scripts/run_feat_res.py
"""
import glob
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from aoi.backbone import Backbone
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)

CONFIGS = [
    ("320/L23(现状)", 320, (2, 3)),
    ("512/L23", 512, (2, 3)),
    ("320/L12", 320, (1, 2)),
    ("512/L12", 512, (1, 2)),
    ("640/L12", 640, (1, 2)),
]


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def make_extractor(bb, size):
    @torch.no_grad()
    def ex(img):
        x = img.unsqueeze(0).to(bb.device) if img.dim() == 3 else img.to(bb.device)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return bb.extract(x)[0]
    return ex


def gather(ex, items, neg_per=300):
    rng = np.random.RandomState(0); Xs, ys = [], []
    for img, mk in items:
        f = ex(img); C, h, w = f.shape
        feat = f.reshape(C, -1).t()
        gt = np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST)).ravel()
        pos = np.where(gt == 1)[0]; neg = np.where(gt == 0)[0]
        if len(neg) > neg_per:
            neg = rng.choice(neg, neg_per, replace=False)
        sel = np.concatenate([pos, neg])
        Xs.append(feat[sel].cpu()); ys.append(torch.tensor(gt[sel].astype(np.int64)))
    return torch.cat(Xs).to(DEV), torch.cat(ys).to(DEV)


def train_head(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    head = nn.Linear(X.shape[1], 2).to(DEV)
    pw = torch.tensor([1.0, float((y == 0).sum()) / max(1, int((y == 1).sum()))], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    torch.manual_seed(0)
    for _ in range(400):
        opt.zero_grad(); F.cross_entropy(head(Xn), y, weight=pw).backward(); opt.step()
    return head, mu, sd


def amap_of(ex, head, mu, sd, img):
    f = ex(img); C, h, w = f.shape
    p = head(((f.reshape(C, -1).t()) - mu) / sd).softmax(1)[:, 1].reshape(1, 1, h, w)
    return F.interpolate(p, size=HW, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()


def f1_thr(scores, labels):
    s = np.concatenate(scores); l = np.concatenate(labels)
    order = np.argsort(-s); ls = l[order]; ss = s[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    f1 = 2 * (tp / np.maximum(tp + fp, 1)) * (tp / P) / np.maximum(tp / np.maximum(tp + fp, 1) + tp / P, 1e-9)
    return float(ss[int(np.argmax(f1))])


def per_img_iou(amap, gt, thr):
    pred = amap >= thr
    TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
    return TP / max(TP + FP + FN, 1)


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit = [(_load_img(p, 640), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[:k]]
    tests = [(_load_img(p, 640), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    return fit, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit = [(_load_img(R / x["image_path"], 640), _read(R / x["mask_path"], HW)) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 640), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    return fit, tests


def main():
    torch.manual_seed(0)
    jobs = [
        ("电子 pcb", lambda: prep_realiad("pcb")),
        ("电池 battery", lambda: prep_realiad("phone_battery")),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
    ]
    data = {n: p() for n, p in jobs}
    print("=== 特征分辨率×层 → 逐图IoU(监督F1阈值)===")
    for tag, size, layers in CONFIGS:
        bb = Backbone(layers=layers, pretrained=True, device=DEV)
        ex = make_extractor(bb, size)
        # 特征延时
        img0 = data["电子 pcb"][1][0][0]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10):
            ex(img0)
        torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 100
        g = ex(img0).shape
        row = []
        for name, (fit, tests) in data.items():
            X, y = gather(ex, fit); h = train_head(X, y)
            # 监督阈值(在fit上标)
            fs = [amap_of(ex, *h, im) for im, _ in fit]
            fl = [mk for _, mk in fit]
            thr = f1_thr([s.ravel() for s in fs], [m.ravel() for m in fl])
            ious = [per_img_iou(amap_of(ex, *h, im), mk, thr) for im, mk in tests]
            row.append((name, np.mean(ious)))
        detail = "  ".join(f"{n.split()[1]}={v:.3f}" for n, v in row)
        print(f"{tag:14s} 格{g[1]}²/{lat:.0f}ms  {detail}  均={np.mean([v for _, v in row]):.3f}", flush=True)
        del bb; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

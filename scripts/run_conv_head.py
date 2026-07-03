"""定位头容量实验:逐像素线性(现状,无上下文) vs 小卷积头(3×3感受野)。
假设:conv头看邻域→边界平滑/噪点少→IoU涨。全图训练(30掩膜×128²=50万像素)。
量纯定位逐图IoU(监督F1阈值)。用法:python scripts/run_conv_head.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from aoi.backbone import Backbone
from eval.mvtec import _load_img
from aoi.imageio import load_fast

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


class LinearHead(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.f = nn.Conv2d(C, 1, 1)
    def forward(self, x):
        return self.f(x)


class ConvHead(nn.Module):
    """1×1降维 + 3×3上下文 + 1×1输出(~10万参,30掩膜可训)。"""
    def __init__(self, C, mid=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(C, mid, 1), nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(mid, 1, 1))
    def forward(self, x):
        return self.f(x)


def extract(bb, img):
    x = img.unsqueeze(0).to(bb.device) if img.dim() == 3 else img.to(bb.device)
    x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
    with torch.no_grad():
        return bb.extract(x)                                # (1,C,h,w)


def train_head(bb, head_cls, fit, normals, steps=300):
    feats, gts = [], []
    with torch.no_grad():
        for img, mk in fit:
            f = extract(bb, img)
            h, w = f.shape[-2:]
            g = torch.from_numpy(np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST))).float()
            feats.append(f); gts.append(g)
        for img in normals[:20]:
            feats.append(extract(bb, img)); gts.append(torch.zeros(feats[-1].shape[-2:]))
    F_all = torch.cat(feats).to(DEV)                        # (N,C,h,w)
    G_all = torch.stack(gts).to(DEV)                        # (N,h,w)
    # 特征标准化(逐通道)
    mu = F_all.mean(dim=(0, 2, 3), keepdim=True); sd = F_all.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    Fn = (F_all - mu) / sd
    head = head_cls(F_all.shape[1]).to(DEV)
    pw = torch.tensor([(G_all == 0).sum() / max(1, (G_all == 1).sum())], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    torch.manual_seed(0)
    for _ in range(steps):
        opt.zero_grad(); lossf(head(Fn).squeeze(1), G_all).backward(); opt.step()
    head.eval()
    # 监督F1阈值(fit上)
    with torch.no_grad():
        logits = head((torch.cat(feats[:len(fit)]).to(DEV) - mu) / sd).squeeze(1)
    s = logits.cpu().numpy().ravel()
    l = np.concatenate([np.array(Image.fromarray(mk).resize(logits.shape[-2:][::-1], Image.NEAREST)).ravel() for _, mk in fit])
    order = np.argsort(-s); ls = l[order]; ss = s[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    f1 = 2 * (tp / np.maximum(tp + fp, 1)) * (tp / P) / np.maximum(tp / np.maximum(tp + fp, 1) + tp / P, 1e-9)
    thr = float(ss[int(np.argmax(f1))])
    return head, mu, sd, thr


def amap(bb, head, mu, sd, img):
    f = extract(bb, img)
    with torch.no_grad():
        lo = head((f - mu) / sd)
    return F.interpolate(lo, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


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


def prep_ad2(cat):
    root = Path(f"data/mvtec_ad_2/{cat}")
    gn = sorted(glob.glob(str(root / "train/good/*.png")))
    bad = sorted(glob.glob(str(root / "test_public/bad/*.png")))
    random.Random(0).shuffle(bad)
    def mpath(p):
        return str(root / "test_public/ground_truth/bad" / (Path(p).stem + "_mask.png"))
    normals = [load_fast(p) for p in gn[:60]]
    fit = [(load_fast(p), _read(mpath(p), HW)) for p in bad[:30]]
    tests = [(load_fast(p), _read(mpath(p), HW)) for p in bad[30:70]]
    return normals, fit, tests


def main():
    torch.manual_seed(0)
    bb = Backbone(layers=(1, 2), pretrained=True, device=DEV)
    jobs = [
        ("电子 pcb", lambda: prep_realiad("pcb")),
        ("电池 battery", lambda: prep_realiad("phone_battery")),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("大图 AD2/sheet", lambda: prep_ad2("sheet_metal")),
    ]
    print("=== 头容量:线性 vs 卷积(纯定位逐图IoU,监督阈值)===")
    L, C = [], []
    for name, prep in jobs:
        normals, fit, tests = prep()
        row = {}
        for tag, cls in [("线性", LinearHead), ("卷积", ConvHead)]:
            head, mu, sd, thr = train_head(bb, cls, fit, normals)
            ious = [iou(amap(bb, head, mu, sd, im) >= thr, mk) for im, mk in tests]
            row[tag] = np.mean(ious)
        L.append(row["线性"]); C.append(row["卷积"])
        print(f"{name:16s} 线性={row['线性']:.3f}  卷积={row['卷积']:.3f}  Δ={row['卷积']-row['线性']:+.3f}", flush=True)
    print(f"\n均值: 线性={np.mean(L):.3f}  卷积={np.mean(C):.3f}  Δ={np.mean(C)-np.mean(L):+.3f}")


if __name__ == "__main__":
    main()

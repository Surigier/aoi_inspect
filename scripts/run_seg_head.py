"""监督分割头实验:用30张标注缺陷掩膜训轻量头(EAD残差384通道上的逐像素逻辑回归),
对比无监督异常图的 pixel-AUROC。验证'迁移图带标注'这个赛题信息能否提升定位精度。
用法:python scripts/run_seg_head.py
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
from aoi.efficientad import EfficientADDetector, OUT
from eval.protocol import image_auroc
from eval.mvtec import _load_img

EVAL_HW = (256, 256)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD")
RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")


def _mask(path, hw):
    if path is None or not Path(path).exists():
        return np.zeros(hw, dtype=np.uint8)
    m = Image.open(path).convert("L").resize((hw[1], hw[0]))
    return (np.array(m) > 0).astype(np.uint8)


# ---------- 数据装载:返回 fit正常 / 缺陷(带掩膜)/ 测试(带掩膜或None) ----------
def items_mvtec(cat):
    root = Path(f"data/mvtec/{cat}")
    norm = [_load_img(p, 320) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    defs, tests = [], []
    for dt in sorted(glob.glob(str(root / "test/*"))):
        dtn = Path(dt).name
        for p in sorted(glob.glob(f"{dt}/*.png")):
            if dtn == "good":
                tests.append((_load_img(p, 320), None))
            else:
                mp = str(GT / cat / "ground_truth" / dtn / (Path(p).stem + "_mask.png"))
                defs.append((_load_img(p, 320), mp))
    return norm, defs, tests


def items_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json"))
    train_ok = [it for it in d["train"] if it["anomaly_class"] == "OK"]
    random.Random(0).shuffle(train_ok)
    norm = [_load_img(RI / cat / it["image_path"], 320) for it in train_ok[:100]]
    defs, tests = [], []
    for it in d["test"]:
        img = _load_img(RI / cat / it["image_path"], 320)
        if it["anomaly_class"] == "OK":
            tests.append((img, None))
        else:
            defs.append((img, str(RI / cat / it["mask_path"])))
    return norm, defs, tests


def items_visa(cat):
    base = Path(f"data/visa/{cat}/Data")
    gn = sorted(glob.glob(str(base / "Images/Normal/*.JPG"))); random.Random(0).shuffle(gn)
    norm = [_load_img(p, 320) for p in gn[:100]]
    tests = [(_load_img(p, 320), None) for p in gn[100:250]]
    defs = []
    for p in sorted(glob.glob(str(base / "Images/Anomaly/*.JPG"))):
        defs.append((_load_img(p, 320), str(base / "Masks/Anomaly" / (Path(p).stem + ".png"))))
    return norm, defs, tests


# ---------- 监督头:残差(C)→逐像素 logit ----------
def train_head(det, fit_defs, fit_norms):
    """采集 fit缺陷(掩膜>0为正)+ fit正常(全负)的逐像素残差特征,训 1×1 logistic。"""
    Xs, ys = [], []
    rng = np.random.RandomState(0)
    def collect(img, mpath):
        res = det.residual_map_large(img)                 # (C,h,w)
        C, h, w = res.shape
        gt = _mask(mpath, (h, w)).ravel()
        feat = res.reshape(C, -1).t().cpu().numpy()        # (h*w, C)
        pos = np.where(gt == 1)[0]; neg = np.where(gt == 0)[0]
        if len(neg) > 400: neg = rng.choice(neg, 400, replace=False)   # 控负样本
        idx = np.concatenate([pos, neg])
        Xs.append(feat[idx]); ys.append(gt[idx])
    for img, mp in fit_defs:
        collect(img, mp)
    for img in fit_norms:
        collect(img, None)
    X = torch.tensor(np.concatenate(Xs), dtype=torch.float32, device=DEV)
    y = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=DEV)
    head = nn.Linear(X.shape[1], 1).to(DEV)
    pos_w = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    for _ in range(300):
        opt.zero_grad(); loss = lossf(head(Xn).squeeze(1), y); loss.backward(); opt.step()
    return head, mu, sd


def head_map(det, head, mu, sd, img):
    res = det.residual_map_large(img)                     # (C,h,w)
    C, h, w = res.shape
    feat = ((res.reshape(C, -1).t() - mu) / sd)
    logit = head(feat).squeeze(1).reshape(1, 1, h, w)
    amap = F.interpolate(logit, size=EVAL_HW, mode="bilinear", align_corners=False)
    return amap[0, 0].detach().cpu().numpy()


def pixel_auroc(maps, masks):
    s = np.concatenate([m.ravel() for m in maps]); l = np.concatenate([m.ravel() for m in masks])
    pos = np.where(l == 1)[0]; neg = np.where(l == 0)[0]
    rng = np.random.RandomState(0)
    if len(pos) > 200000: pos = rng.choice(pos, 200000, replace=False)
    if len(neg) > 200000: neg = rng.choice(neg, 200000, replace=False)
    idx = np.concatenate([pos, neg])
    return image_auroc(s[idx], l[idx])


def run(name, norm, defs, tests):
    random.Random(0).shuffle(defs)
    fit_defs = defs[:30]; eval_defs = defs[30:]
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(norm, None)
    head, mu, sd = train_head(det, fit_defs, norm[:30])
    # 评测集 = 测试正常 + 未用于训头的缺陷
    ev = tests + eval_defs
    un_maps, su_maps, masks = [], [], []
    for img, mp in ev:
        un_maps.append(det.anomaly_map_large(img, out_hw=EVAL_HW))
        su_maps.append(head_map(det, head, mu, sd, img))
        masks.append(_mask(mp, EVAL_HW))
    au_un = pixel_auroc(un_maps, masks); au_su = pixel_auroc(su_maps, masks)
    print(f"{name:20s} 无监督={au_un:.3f}  监督头={au_su:.3f}  Δ={au_su-au_un:+.3f}", flush=True)
    return au_un, au_su


def main():
    torch.manual_seed(0)
    jobs = [
        ("mvtec/transistor", items_mvtec, "transistor"),
        ("realiad/pcb", items_realiad, "pcb"),
        ("realiad/phone_battery", items_realiad, "phone_battery"),
        ("visa/pcb1", items_visa, "pcb1"),
    ]
    uns, sus = [], []
    for name, fn, cat in jobs:
        norm, defs, tests = fn(cat)
        u, s = run(name, norm, defs, tests)
        uns.append(u); sus.append(s)
    print(f"\n均值: 无监督={np.mean(uns):.3f}  监督头={np.mean(sus):.3f}  Δ={np.mean(sus)-np.mean(uns):+.3f}")


if __name__ == "__main__":
    main()

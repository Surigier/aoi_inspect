"""双反事实增益 × 二分类缺陷分割头(type-agnostic 长尾:正常=head 缺陷=tail,不需类型标签)。
核心:每图 99%+ 是正常像素→正常永远head、缺陷永远tail。双反事实在正常/缺陷争议边界发力,
不依赖缺陷型标签,契合赛题(类型未知)。对比 朴素CE / 类平衡CE / +双反事实,测 pixel-AUROC。
用法:python scripts/run_binary_cf.py
"""
import json
import glob
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
EVAL_HW = (256, 256); NEG_PER = 300
DELTA = 1.0; CAP = 8.0; GAIN_SCALE = 1.0             # 双反事实超参


def _mask(p, hw):
    if p is None or not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def dual_cf_gain(Z, y):
    """二分类双反事实增益 g∈[0,1](detach)。y:0=正常(head) 1=缺陷(tail)。
    对缺陷像素:h*=正常;m=clamp(relu(z_norm-z_def+δ),cap);
    g_fwd=压正常后缺陷概率回升;g_bwd=抬缺陷后正常概率回落。只在缺陷像素非零。"""
    with torch.no_grad():
        N = Z.shape[0]; idx = torch.arange(N, device=Z.device)
        p = Z.softmax(1)
        hstar = 1 - y                                # 二分类:竞争类=另一类(缺陷像素→正常)
        zy = Z[idx, y]; zh = Z[idx, hstar]
        m = torch.clamp(torch.relu(zh - zy + DELTA), max=CAP)
        Zf = Z.clone(); Zf[idx, hstar] -= m
        g_fwd = torch.relu(Zf.softmax(1)[idx, y] - p[idx, y])
        Zb = Z.clone(); Zb[idx, y] += m
        g_bwd = torch.relu(p[idx, hstar] - Zb.softmax(1)[idx, hstar])
        g = torch.clamp((0.5 * g_fwd + 0.5 * g_bwd) * GAIN_SCALE, 0, 1)
        return g * (y == 1).float()                  # 只在缺陷(tail)像素


def gather(det, items):
    rng = np.random.RandomState(0); Xs, ys = [], []
    for img, mp in items:
        res = det.residual_map_large(img); C, h, w = res.shape
        feat = res.reshape(C, -1).t()
        gt = _mask(mp, (h, w)).ravel() if mp else np.zeros(h * w, np.uint8)
        pos = np.where(gt == 1)[0]; neg = np.where(gt == 0)[0]
        if len(neg) > NEG_PER:
            neg = rng.choice(neg, NEG_PER, replace=False)
        sel = np.concatenate([pos, neg])
        Xs.append(feat[sel].cpu()); ys.append(torch.tensor(gt[sel].astype(np.int64)))
    return torch.cat(Xs).to(DEV), torch.cat(ys).to(DEV)


def train(X, y, balance, use_cf, cf_scale=4.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    head = nn.Linear(X.shape[1], 2).to(DEV)
    cw = None
    if balance:
        cnt = torch.bincount(y, minlength=2).float()
        cw = (cnt.sum() / (cnt + 1)).clamp(max=50).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    torch.manual_seed(0)
    for _ in range(400):
        opt.zero_grad(); Z = head(Xn)
        ce = F.cross_entropy(Z, y, weight=cw, reduction="none")
        if use_cf:
            ce = ce * (1.0 + cf_scale * dual_cf_gain(Z.detach(), y))
        ce.mean().backward(); opt.step()
    return head, mu, sd


def auroc_eval(det, head, mu, sd, tests):
    S, L = [], []
    for img, mp in tests:
        res = det.residual_map_large(img); C, h, w = res.shape
        Z = head((res.reshape(C, -1).t() - mu) / sd)
        pc = Z.softmax(1)[:, 1].reshape(1, 1, h, w)
        amap = F.interpolate(pc, size=EVAL_HW, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        S.append(amap.ravel()); L.append(_mask(mp, EVAL_HW).ravel())
    return image_auroc(np.concatenate(S), np.concatenate(L))


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
                defs.append((_load_img(p, 320), str(GT / cat / "ground_truth" / dtn / (Path(p).stem + "_mask.png"))))
    random.Random(0).shuffle(defs)
    return norm, defs[:30], tests + defs[30:]


def items_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    norm = [_load_img(R / x["image_path"], 320) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    defs = [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], 320), None) for x in d["test"] if x["anomaly_class"] == "OK"][:150]
    tests += [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[30:70]]
    return norm, defs, tests


def main():
    torch.manual_seed(0)
    jobs = [("mvtec/transistor", items_mvtec, "transistor"),
            ("realiad/pcb", items_realiad, "pcb"),
            ("realiad/phone_battery", items_realiad, "phone_battery")]
    for name, fn, cat in jobs:
        norm, defs, tests = fn(cat)
        det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
        det.fit_fewshot(norm, None)
        X, y = gather(det, defs + [(n, None) for n in norm[:30]])
        row = {}
        for tag, bal, cf in [("朴素", False, False), ("类平衡", True, False),
                             ("朴素+CF", False, True), ("平衡+CF", True, True)]:
            head, mu, sd = train(X, y, bal, cf)
            row[tag] = auroc_eval(det, head, mu, sd, tests)
        print(f"{name:22s} 朴素={row['朴素']:.3f} 类平衡={row['类平衡']:.3f} "
              f"朴素+CF={row['朴素+CF']:.3f} 平衡+CF={row['平衡+CF']:.3f}", flush=True)


if __name__ == "__main__":
    main()

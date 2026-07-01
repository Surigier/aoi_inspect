"""双反事实增益 × 多分类缺陷型分割头(长尾验证)。
在 Real-IAD pcb(缺陷型天然长尾:PS/ZW多, AK/HS少)上,把监督分割头改逐像素多分类
(softmax over {正常,PS,ZW,AK,HS}),对比:基线加权CE vs +双反事实增益(do-干预 head/tail logit)。
测每类(尤其尾类AK/HS)pixel-AUROC。用法:python scripts/run_dual_cf.py
"""
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# phone_battery:缺陷型天然长尾 PS125/ZW110(头) vs AK38/HS28(尾),且真手机电池件最对口
RI = Path("data/_dl/Real-IAD/phone_battery"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv/phone_battery.json")
TYPES = ["PS", "ZW", "AK", "HS"]                       # 0=正常,1..4;头={PS,ZW} 尾={AK,HS}
CLS = {t: i + 1 for i, t in enumerate(TYPES)}
EVAL_HW = (256, 256)
NEG_PER = 300
# 双反事实超参(对齐用户公式)
W_BI = 0.5; DELTA = 1.0; CAP = 8.0; GAIN_SCALE = 1.0


def _mask(p, hw):
    if p is None or not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def dual_cf_gain(Z, y):
    """逐像素双反事实增益 g∈[0,1](detach,作loss权重)。Z:(N,K) logits, y:(N,) GT类。
    h*=每像素最强竞争错类;m自适应=clamp(relu(z_h*-z_y*+δ),cap);
    g_fwd=压head后y*概率回升; g_bwd=抬tail后head概率回落; 只在缺陷像素(y>0)非零。"""
    with torch.no_grad():
        N, K = Z.shape
        idx = torch.arange(N, device=Z.device)
        p = Z.softmax(1)
        zy = Z[idx, y]
        # h* = argmax over k≠y* 的概率
        pm = p.clone(); pm[idx, y] = -1
        hstar = pm.argmax(1)
        zh = Z[idx, hstar]
        m = torch.clamp(torch.relu(zh - zy + DELTA), max=CAP)          # v3s2 自适应 m
        # 前向:do(z_h* -= m)
        Zf = Z.clone(); Zf[idx, hstar] = Zf[idx, hstar] - m
        pf = Zf.softmax(1)
        g_fwd = torch.relu(pf[idx, y] - p[idx, y])
        # 后向:do(z_y* += m)
        Zb = Z.clone(); Zb[idx, y] = Zb[idx, y] + m
        pb = Zb.softmax(1)
        g_bwd = torch.relu(p[idx, hstar] - pb[idx, hstar])
        g = (1 - W_BI) * g_fwd + W_BI * g_bwd
        g = torch.clamp(g * GAIN_SCALE, 0, 1)
        g = g * (y > 0).float()                                       # 只在缺陷(尾)像素
        return g


def gather_pixels(det, items):
    """items:[(img, type_idx or 0, mask_path or None)] → (feat[N,384], label[N])。"""
    rng = np.random.RandomState(0)
    Xs, ys = [], []
    for img, cls, mp in items:
        res = det.residual_map_large(img)              # (384,h,w)
        C, h, w = res.shape
        feat = res.reshape(C, -1).t()
        gt = _mask(mp, (h, w)).ravel() if cls > 0 else np.zeros(h * w, np.uint8)
        lab = (gt * cls).astype(np.int64)              # 缺陷像素=类号,其余0
        pos = np.where(lab > 0)[0]; neg = np.where(lab == 0)[0]
        if len(neg) > NEG_PER:
            neg = rng.choice(neg, NEG_PER, replace=False)
        sel = np.concatenate([pos, neg])
        Xs.append(feat[sel].cpu()); ys.append(torch.tensor(lab[sel]))
    return torch.cat(Xs).to(DEV), torch.cat(ys).to(DEV)


def train_head(X, y, K, balance=True, use_cf=False, cf_scale=4.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    head = nn.Linear(X.shape[1], K).to(DEV)
    if balance:
        cnt = torch.bincount(y, minlength=K).float()
        cw = (cnt.sum() / (cnt + 1)).clamp(max=50).to(DEV)
    else:
        cw = None
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    torch.manual_seed(0)
    g_last = 0.0
    for it in range(400):
        opt.zero_grad()
        Z = head(Xn)
        ce = F.cross_entropy(Z, y, weight=cw, reduction="none")
        if use_cf:
            g = dual_cf_gain(Z.detach(), y)
            ce = ce * (1.0 + cf_scale * g)
            if it == 399:
                dp = (y > 0)
                g_last = float(g[dp].mean()) if dp.any() else 0.0

        ce.mean().backward(); opt.step()
    return head, mu, sd, g_last


def per_type_auroc(det, head, mu, sd, test_by_type):
    """每型:该型测试图的 P(该型) vs GT掩膜 → pixel-AUROC。"""
    out = {}
    for t, imgs_masks in test_by_type.items():
        if not imgs_masks:
            out[t] = float("nan"); continue
        c = CLS[t]
        S, L = [], []
        for img, mp in imgs_masks:
            res = det.residual_map_large(img); C, h, w = res.shape
            Z = head((res.reshape(C, -1).t() - mu) / sd)
            pc = Z.softmax(1)[:, c].reshape(1, 1, h, w)
            amap = F.interpolate(pc, size=EVAL_HW, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
            S.append(amap.ravel()); L.append(_mask(mp, EVAL_HW).ravel())
        out[t] = image_auroc(np.concatenate(S), np.concatenate(L))
    return out


def main():
    torch.manual_seed(0)
    d = json.load(open(RJ))
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(RI / x["image_path"], 320) for x in tok[:100]]
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(normals, None)

    ng = [x for x in d["test"] if x["anomaly_class"] in CLS]
    by_type = {t: [x for x in ng if x["anomaly_class"] == t] for t in TYPES}
    for t in TYPES:
        random.Random(0).shuffle(by_type[t])
    print("缺陷型分布:", {t: len(by_type[t]) for t in TYPES})

    # fit:每型取前若干(反映长尾:头类多/尾类少),共~30
    fit_items = []
    fit_n = {"PS": 12, "ZW": 11, "AK": 4, "HS": 3}
    test_by_type = {}
    for t in TYPES:
        k = fit_n[t]
        for x in by_type[t][:k]:
            fit_items.append((_load_img(RI / x["image_path"], 320), CLS[t], str(RI / x["mask_path"])))
        test_by_type[t] = [(_load_img(RI / x["image_path"], 320), str(RI / x["mask_path"]))
                           for x in by_type[t][k:k + 40]]
    # 加正常负样本
    for img in normals[:30]:
        fit_items.append((img, 0, None))

    X, y = gather_pixels(det, fit_items)
    print("训练像素:", {int(c): int((y == c).sum()) for c in y.unique()})
    K = len(TYPES) + 1
    configs = [
        ("A 朴素CE",       dict(balance=False, use_cf=False)),
        ("B 类平衡CE",     dict(balance=True,  use_cf=False)),
        ("C 朴素+双反事实", dict(balance=False, use_cf=True)),
        ("D 平衡+双反事实", dict(balance=True,  use_cf=True)),
    ]
    for tag, kw in configs:
        head, mu, sd, gl = train_head(X, y, K, **kw)
        au = per_type_auroc(det, head, mu, sd, test_by_type)
        head_m = np.mean([au["PS"], au["ZW"]]); tail_m = np.mean([au["AK"], au["HS"]])
        gtag = f" ḡ={gl:.3f}" if kw["use_cf"] else ""
        print(f"{tag:14s} | PS={au['PS']:.3f} ZW={au['ZW']:.3f} | AK={au['AK']:.3f} HS={au['HS']:.3f} "
              f"| 头均={head_m:.3f} 尾均={tail_m:.3f}{gtag}", flush=True)


if __name__ == "__main__":
    main()

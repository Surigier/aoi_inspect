"""双反事实 × 跨域泛化(赛题命根:泛化到未知域)。
头在源域(phone_battery)学"缺陷vs正常",搬到目标域(pcb/usb/sim_card,各自EAD拟合自己正常)
测像素定位。假设:CF学可迁移因果缺陷特征、朴素头学源域假相关 → CF跨域更稳。
对比 朴素CE / +双反事实 的目标域 pixel-AUROC。用法:python scripts/run_crossdomain_cf.py
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
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
EVAL_HW = (256, 256); NEG_PER = 300
DELTA = 1.0; CAP = 8.0
SRC = "phone_battery"; TGTS = ["pcb", "usb", "sim_card_set", "button_battery"]


def _mask(p, hw):
    if p is None or not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def dual_cf_gain(Z, y):
    with torch.no_grad():
        N = Z.shape[0]; idx = torch.arange(N, device=Z.device)
        p = Z.softmax(1); hstar = 1 - y
        zy = Z[idx, y]; zh = Z[idx, hstar]
        m = torch.clamp(torch.relu(zh - zy + DELTA), max=CAP)
        Zf = Z.clone(); Zf[idx, hstar] -= m
        g_fwd = torch.relu(Zf.softmax(1)[idx, y] - p[idx, y])
        Zb = Z.clone(); Zb[idx, y] += m
        g_bwd = torch.relu(p[idx, hstar] - Zb.softmax(1)[idx, hstar])
        g = torch.clamp(0.5 * g_fwd + 0.5 * g_bwd, 0, 1)
        return g * (y == 1).float()


def fit_ead(cat, n=100):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    norm = [_load_img(R / x["image_path"], 320) for x in tok[:n]]
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(norm, None)
    return det, norm


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


def train(X, y, use_cf, cf_scale=4.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    head = nn.Linear(X.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    torch.manual_seed(0)
    for _ in range(400):
        opt.zero_grad(); Z = head(Xn)
        ce = F.cross_entropy(Z, y, reduction="none")
        if use_cf:
            ce = ce * (1.0 + cf_scale * dual_cf_gain(Z.detach(), y))
        ce.mean().backward(); opt.step()
    return head, mu, sd


def eval_tgt(det, head, mu, sd, tests):
    S, L = [], []
    for img, mp in tests:
        res = det.residual_map_large(img); C, h, w = res.shape
        Z = head((res.reshape(C, -1).t() - mu) / sd)
        pc = Z.softmax(1)[:, 1].reshape(1, 1, h, w)
        amap = F.interpolate(pc, size=EVAL_HW, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        S.append(amap.ravel()); L.append(_mask(mp, EVAL_HW).ravel())
    return image_auroc(np.concatenate(S), np.concatenate(L))


def tgt_tests(det, cat, k=40):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    ok = [x for x in d["test"] if x["anomaly_class"] == "OK"][:100]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(1).shuffle(ng)
    tests = [(_load_img(R / x["image_path"], 320), None) for x in ok]
    tests += [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[:k]]
    return tests


def main():
    torch.manual_seed(0)
    # 源域:拟合 EAD + 30缺陷训头(朴素 & CF)
    det_s, norm_s = fit_ead(SRC)
    d = json.load(open(RJ / f"{SRC}.json")); R = RI / SRC
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    defs = [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[:30]]
    X, y = gather(det_s, defs + [(n, None) for n in norm_s[:30]])
    head_plain = train(X, y, False)
    head_cf = train(X, y, True)
    # 源域内自测(对照)
    src_tests = tgt_tests(det_s, SRC)
    print(f"源域内({SRC}): 朴素={eval_tgt(det_s, *head_plain, src_tests):.3f} "
          f"CF={eval_tgt(det_s, *head_cf, src_tests):.3f}")
    # 跨域:每个目标域拟合自己EAD,套源域头
    dp, dc = [], []
    for t in TGTS:
        det_t, _ = fit_ead(t)
        tests = tgt_tests(det_t, t)
        ap = eval_tgt(det_t, *head_plain, tests); ac = eval_tgt(det_t, *head_cf, tests)
        dp.append(ap); dc.append(ac)
        print(f"跨域 {SRC}→{t:14s}: 朴素={ap:.3f}  +双反事实={ac:.3f}  Δ={ac-ap:+.3f}", flush=True)
    print(f"\n跨域均值: 朴素={np.mean(dp):.3f}  +双反事实={np.mean(dc):.3f}  Δ={np.mean(dc)-np.mean(dp):+.3f}")


if __name__ == "__main__":
    main()

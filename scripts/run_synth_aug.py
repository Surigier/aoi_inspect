"""反事实缺陷合成增广 × 跨域泛化(赛题点名合成缺陷/GANs)。
监督分割头训练除30真缺陷外,加 N 张合成缺陷(CutPaste/色彩/噪声,带mask)→ 扩张训练分布。
在CF失败的同一跨域台(phone_battery训→pcb/usb/sim/button测)对比:仅真缺陷 vs +合成。
假设:合成引入多样非源域特有缺陷→头学通用缺陷特征→跨域更稳(与CF收缩边界相反)。
用法:python scripts/run_synth_aug.py
"""
import json
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from aoi.efficientad import EfficientADDetector
from eval.protocol import image_auroc
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
EVAL_HW = (256, 256); NEG_PER = 300
SRC = "phone_battery"; TGTS = ["pcb", "usb", "sim_card_set", "button_battery"]
N_SYNTH = 60


def _mask(p, hw):
    if p is None or not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def synth_defect(img, rng):
    """正常图 → 合成缺陷图 + mask。反事实"正常→缺陷":随机blob区域施加
    CutPaste(错位内容)/色彩偏移/噪声,覆盖5类缺陷的视觉代理(外观/色彩/结构)。"""
    C, H, W = img.shape
    out = img.clone()
    mask = np.zeros((H, W), np.uint8)
    for _ in range(rng.randint(1, 4)):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        ax = rng.randint(max(4, W // 20), max(6, W // 6))
        ay = rng.randint(max(4, H // 20), max(6, H // 6))
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (cx, cy), (ax, ay), rng.randint(0, 180), 0, 360, 1, -1)
        mb = torch.from_numpy(m > 0).to(img.device)
        mode = rng.choice(["cutpaste", "color", "noise"])
        if mode == "cutpaste":
            dx, dy = rng.randint(-W // 3, W // 3), rng.randint(-H // 3, H // 3)
            src = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
            out[:, mb] = src[:, mb]
        elif mode == "color":
            f = torch.tensor([rng.uniform(0.4, 1.8) for _ in range(3)], device=img.device).view(3, 1)
            out[:, mb] = (out[:, mb] * f).clamp(0, 1)
        else:
            out[:, mb] = (out[:, mb] + torch.randn_like(out[:, mb]) * 0.3).clamp(0, 1)
        mask |= m
    return out, mask


def fit_ead(cat, n=100):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    norm = [_load_img(R / x["image_path"], 320) for x in tok[:n]]
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    det.fit_fewshot(norm, None)
    return det, norm


def gather(det, items):
    rng = np.random.RandomState(0); Xs, ys = [], []
    for img, mp_or_mask in items:
        res = det.residual_map_large(img); C, h, w = res.shape
        feat = res.reshape(C, -1).t()
        if isinstance(mp_or_mask, np.ndarray):                 # 合成:直接给mask数组
            gt = cv2.resize(mp_or_mask, (w, h), interpolation=cv2.INTER_NEAREST).ravel()
        else:
            gt = _mask(mp_or_mask, (h, w)).ravel() if mp_or_mask else np.zeros(h * w, np.uint8)
        pos = np.where(gt == 1)[0]; neg = np.where(gt == 0)[0]
        if len(neg) > NEG_PER:
            neg = rng.choice(neg, NEG_PER, replace=False)
        sel = np.concatenate([pos, neg])
        Xs.append(feat[sel].cpu()); ys.append(torch.tensor(gt[sel].astype(np.int64)))
    return torch.cat(Xs).to(DEV), torch.cat(ys).to(DEV)


def train(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    head = nn.Linear(X.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    torch.manual_seed(0)
    for _ in range(400):
        opt.zero_grad(); F.cross_entropy(head(Xn), y).backward(); opt.step()
    return head, mu, sd


def evald(det, head, mu, sd, tests):
    S, L = [], []
    for img, mp in tests:
        res = det.residual_map_large(img); C, h, w = res.shape
        Z = head((res.reshape(C, -1).t() - mu) / sd)
        pc = Z.softmax(1)[:, 1].reshape(1, 1, h, w)
        amap = F.interpolate(pc, size=EVAL_HW, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        S.append(amap.ravel()); L.append(_mask(mp, EVAL_HW).ravel())
    return image_auroc(np.concatenate(S), np.concatenate(L))


def tgt_tests(cat, k=40):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    ok = [x for x in d["test"] if x["anomaly_class"] == "OK"][:100]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(1).shuffle(ng)
    return ([(_load_img(R / x["image_path"], 320), None) for x in ok]
            + [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[:k]])


def main():
    torch.manual_seed(0); rng = random.Random(0)
    det_s, norm_s = fit_ead(SRC)
    d = json.load(open(RJ / f"{SRC}.json")); R = RI / SRC
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    real = [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[:30]]
    negs = [(n, None) for n in norm_s[:30]]
    synth = [synth_defect(norm_s[rng.randrange(len(norm_s))], rng) for _ in range(N_SYNTH)]

    heads = {}
    X, y = gather(det_s, real + negs)
    heads["仅真缺陷"] = train(X, y)
    Xs, ys = gather(det_s, real + synth + negs)
    heads["真+合成"] = train(Xs, ys)
    Xso, yso = gather(det_s, synth + negs)
    heads["仅合成(0真缺陷)"] = train(Xso, yso)

    det_cache = {SRC: (det_s, tgt_tests(SRC))}
    for t in TGTS:
        det_cache[t] = (fit_ead(t)[0], tgt_tests(t))

    print(f"{'配置':16s} 源内   跨域均值")
    for tag, hd in heads.items():
        src = evald(det_cache[SRC][0], *hd, det_cache[SRC][1])
        cross = [evald(det_cache[t][0], *hd, det_cache[t][1]) for t in TGTS]
        detail = " ".join(f"{t[:3]}={c:.3f}" for t, c in zip(TGTS, cross))
        print(f"{tag:16s} {src:.3f}  {np.mean(cross):.3f}   [{detail}]", flush=True)


if __name__ == "__main__":
    main()

"""合成缺陷增广的多种子跨域评估——排除EAD运行间方差,看真实效果。
每种子:源域(phone_battery)训头(仅真 vs 真+合成)→跨域(pcb/usb/sim)测,记 delta。
报每种子 delta + 均值±std。用法:python scripts/run_synth_multiseed.py
"""
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from aoi.efficientad import EfficientADDetector
from aoi.synth import synth_defect
from eval.protocol import image_auroc
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
EVAL_HW = (256, 256); NEG_PER = 300
SRC = "phone_battery"; TGTS = ["pcb", "usb", "sim_card_set"]
N_SYNTH = 60; SEEDS = [1, 2, 3]


def _mask(p, hw):
    if p is None or not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def fit_ead(cat, seed, n=100):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    norm = [_load_img(R / x["image_path"], 320) for x in tok[:n]]
    det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000, seed=seed)
    det.fit_fewshot(norm, None)
    return det, norm


def gather(det, items):
    rng = np.random.RandomState(0); Xs, ys = [], []
    for img, mm in items:
        res = det.residual_map_large(img); C, h, w = res.shape
        feat = res.reshape(C, -1).t()
        if isinstance(mm, np.ndarray):
            import cv2
            gt = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST).ravel()
        else:
            gt = _mask(mm, (h, w)).ravel() if mm else np.zeros(h * w, np.uint8)
        pos = np.where(gt == 1)[0]; neg = np.where(gt == 0)[0]
        if len(neg) > NEG_PER:
            neg = rng.choice(neg, NEG_PER, replace=False)
        sel = np.concatenate([pos, neg])
        Xs.append(feat[sel].cpu()); ys.append(torch.tensor(gt[sel].astype(np.int64)))
    return torch.cat(Xs).to(DEV), torch.cat(ys).to(DEV)


def train(X, y, seed):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    head = nn.Linear(X.shape[1], 2).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    torch.manual_seed(seed)
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
    deltas, reals, synths = [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed); rng = random.Random(seed)
        det_s, norm_s = fit_ead(SRC, seed)
        d = json.load(open(RJ / f"{SRC}.json")); R = RI / SRC
        ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
        real = [(_load_img(R / x["image_path"], 320), str(R / x["mask_path"])) for x in ng[:30]]
        negs = [(n, None) for n in norm_s[:30]]
        synth = [synth_defect(norm_s[rng.randrange(len(norm_s))], rng) for _ in range(N_SYNTH)]
        h_real = train(*gather(det_s, real + negs), seed)
        h_synth = train(*gather(det_s, real + synth + negs), seed)
        tgts = [(fit_ead(t, seed)[0], tgt_tests(t)) for t in TGTS]
        cr = np.mean([evald(dt, *h_real, ts) for dt, ts in tgts])
        cs = np.mean([evald(dt, *h_synth, ts) for dt, ts in tgts])
        reals.append(cr); synths.append(cs); deltas.append(cs - cr)
        print(f"seed={seed}: 仅真={cr:.3f}  真+合成={cs:.3f}  Δ={cs-cr:+.3f}", flush=True)
    print(f"\n均值: 仅真={np.mean(reals):.3f}  真+合成={np.mean(synths):.3f}  "
          f"Δ={np.mean(deltas):+.3f} ± {np.std(deltas):.3f}  (n={len(SEEDS)}种子)")


if __name__ == "__main__":
    main()

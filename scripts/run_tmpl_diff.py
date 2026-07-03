"""金模板配准差分(工业AOI经典,治刚性件微小缺陷):
测试图与最近邻正常参考配准(ECC)后,头输入=concat[feat(测试), feat(测试)-feat(对齐参考)]。
5px级缺陷在差分特征里信号强(单图特征几乎不可见)。对比单特征基线:IoU+框命中。
用法:python scripts/run_tmpl_diff.py
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


def gray64(img):
    g = img.mean(0)
    return F.interpolate(g[None, None], size=(64, 64), mode="bilinear")[0, 0].numpy()


class RefBank:
    """正常参考库:最近邻挑参考 + ECC 配准。"""
    def __init__(self, normals):
        self.refs = normals
        self.keys = np.stack([gray64(n) for n in normals])

    def aligned_ref(self, img):
        q = gray64(img)
        i = int(np.argmin(((self.keys - q) ** 2).mean(axis=(1, 2))))
        ref = self.refs[i]
        # ECC 配准(灰度256,euclidean;失败回退不变换)
        try:
            g1 = cv2.resize((img.mean(0).numpy() * 255).astype(np.uint8), (256, 256))
            g2 = cv2.resize((ref.mean(0).numpy() * 255).astype(np.uint8), (256, 256))
            warp = np.eye(2, 3, dtype=np.float32)
            cv2.findTransformECC(g1, g2, warp, cv2.MOTION_EUCLIDEAN,
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4))
            H, W = ref.shape[-2:]
            scale = np.diag([W / 256, H / 256, 1]).astype(np.float32)
            w3 = np.vstack([warp, [0, 0, 1]])
            w_full = (scale @ w3 @ np.linalg.inv(scale))[:2]
            arr = ref.permute(1, 2, 0).numpy()
            out = cv2.warpAffine(arr, w_full, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            return torch.from_numpy(out).permute(2, 0, 1)
        except Exception:
            return ref


def extract(bb, img):
    x = (img.unsqueeze(0) if img.dim() == 3 else img).to(bb.device)
    x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
    with torch.no_grad():
        return bb.extract(x)


def feats_of(bb, img, bank, use_diff):
    f = extract(bb, img)
    if not use_diff:
        return f
    fr = extract(bb, bank.aligned_ref(img))
    return torch.cat([f, f - fr], dim=1)


def train(bb, fit, normals, bank, use_diff, steps=300):
    feats, gts = [], []
    with torch.no_grad():
        for img, mk in fit:
            f = feats_of(bb, img, bank, use_diff); h, w = f.shape[-2:]
            feats.append(f.half().cpu())
            gts.append(torch.from_numpy(np.array(Image.fromarray(mk).resize((w, h), Image.NEAREST))).float())
        for img in normals[:20]:
            feats.append(feats_of(bb, img, bank, use_diff).half().cpu())
            gts.append(torch.zeros(feats[-1].shape[-2:]))
    Fa = torch.cat([f.float() for f in feats]).to(DEV); Ga = torch.stack(gts).to(DEV)
    mu = Fa.mean(dim=(0, 2, 3), keepdim=True); sd = Fa.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    Fn = (Fa - mu) / sd
    head = nn.Conv2d(Fa.shape[1], 1, 1).to(DEV)
    pw = torch.tensor([(Ga == 0).sum() / max(1, (Ga == 1).sum())], device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    torch.manual_seed(0)
    for _ in range(steps):
        opt.zero_grad(); lossf(head(Fn).squeeze(1), Ga).backward(); opt.step()
    head.eval()
    return head, mu, sd


def amap_of(bb, head, mu, sd, img, bank, use_diff):
    f = feats_of(bb, img, bank, use_diff)
    with torch.no_grad():
        lo = head((f - mu) / sd)
    return F.interpolate(lo, size=HW, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def f1_thr(S, L):
    s = np.concatenate([x.ravel() for x in S]); l = np.concatenate([x.ravel() for x in L])
    o = np.argsort(-s); ls = l[o]; ss = s[o]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
    f1 = 2*(tp/np.maximum(tp+fp,1))*(tp/P)/np.maximum(tp/np.maximum(tp+fp,1)+tp/P,1e-9)
    return float(ss[int(np.argmax(f1))])


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
    print("=== 金模板配准差分 vs 单特征(刚性电子件)===")
    for cat in ["pcb", "phone_battery"]:
        normals, fit, tests = prep_realiad(cat)
        bank = RefBank(normals[:40])
        tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in tests]
        tst_gt = [gtb(l) for l in tstL]
        row = {}
        for tag, ud in [("单特征", False), ("+模板差分", True)]:
            head, mu, sd = train(bb, fit, normals, bank, ud)
            fitS = [amap_of(bb, head, mu, sd, im, bank, ud) for im, _ in fit]
            fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in fit]
            thr = f1_thr(fitS, fitL)
            S = [amap_of(bb, head, mu, sd, im, bank, ud) for im, _ in tests]
            ious = [iou(s >= thr, l) for s, l in zip(S, tstL)]
            bx = [gtb((s >= thr).astype(np.uint8), min_a=3) for s in S]
            row[tag] = (np.mean(ious), hit_rate(list(zip(bx, tst_gt))))
        print(f"{cat:14s} 单特征:IoU={row['单特征'][0]:.3f}/框={row['单特征'][1]:.3f}  "
              f"+差分:IoU={row['+模板差分'][0]:.3f}/框={row['+模板差分'][1]:.3f}  "
              f"ΔIoU={row['+模板差分'][0]-row['单特征'][0]:+.3f}", flush=True)


if __name__ == "__main__":
    main()

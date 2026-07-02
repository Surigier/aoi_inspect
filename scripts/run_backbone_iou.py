"""表征能力对定位IoU的影响:EAD残差(小PDN) vs WideResNet50 vs DINOv2。
验证假设:小模型表征弱是IoU瓶颈,换强骨干特征→监督头出更准掩膜→IoU涨。
同一监督逻辑回归头,弱类上量 best-IoU。用法:python scripts/run_backbone_iou.py
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
from aoi.efficientad import EfficientADDetector
from aoi.backbone import Backbone
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256); SIZE = 322


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def best_iou(s, l):
    order = np.argsort(-s); ls = l[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    bi = int(np.argmax(f1))
    return float(tp[bi] / max(tp[bi] + fp[bi] + (P - tp[bi]), 1))


# ---------- 三种特征提取器:img(3,H,W)[0,1] → (C,h,w) tensor ----------
class EADFeat:
    def __init__(self, normals):
        self.det = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
        self.det.fit_fewshot(normals, None)
    def __call__(self, img):
        return self.det.residual_map_large(img)


class WRNFeat:
    def __init__(self):
        self.bb = Backbone(pretrained=True, device=DEV)
    @torch.no_grad()
    def __call__(self, img):
        return self.bb.extract(img.unsqueeze(0).to(DEV))[0]


class DinoFeat:
    def __init__(self):
        import timm
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    @torch.no_grad()
    def __call__(self, img):
        x = ((img.unsqueeze(0).to(DEV) - self.mean) / self.std)
        t = self.m.forward_features(x)          # (1, 1+N, C)
        t = t[:, self.m.num_prefix_tokens:, :]  # 去cls
        n = t.shape[1]; g = int(n ** 0.5)
        return t.reshape(1, g, g, -1).permute(0, 3, 1, 2)[0]  # (C,g,g)


def gather(feat_fn, items, neg_per=300):
    rng = np.random.RandomState(0); Xs, ys = [], []
    for img, mp in items:
        f = feat_fn(img); C, h, w = f.shape
        feat = f.reshape(C, -1).t()
        gt = np.array(Image.fromarray(mp).resize((w, h), Image.NEAREST)) if mp.shape != (h, w) else mp
        gt = gt.ravel()
        pos = np.where(gt == 1)[0]; neg = np.where(gt == 0)[0]
        if len(neg) > neg_per:
            neg = rng.choice(neg, neg_per, replace=False)
        sel = np.concatenate([pos, neg])
        Xs.append(feat[sel].cpu()); ys.append(torch.tensor(gt[sel].astype(np.int64)))
    return torch.cat(Xs).to(DEV), torch.cat(ys).to(DEV)


def train_head(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xn = (X - mu) / sd
    head = nn.Linear(X.shape[1], 2).to(DEV)
    pw = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum()), 1.0], device=DEV)  # 类权重
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    torch.manual_seed(0)
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy(head(Xn), y, weight=pw.flip(0)).backward(); opt.step()
    return head, mu, sd


def evalb(feat_fn, head, mu, sd, tests):
    S, L = [], []
    for img, mp in tests:
        f = feat_fn(img); C, h, w = f.shape
        p = head(((f.reshape(C, -1).t()) - mu) / sd).softmax(1)[:, 1].reshape(1, 1, h, w)
        amap = F.interpolate(p, size=HW, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        S.append(amap.ravel()); L.append(mp.ravel())
    return best_iou(np.concatenate(S), np.concatenate(L))


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, SIZE) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [_load_img(p, SIZE) for p in sorted(glob.glob(str(root / "test/good/*.png")))]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit = [(_load_img(p, SIZE), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[:k]]
    tests = [(_load_img(p, SIZE), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW)) for p, fo in df[k:]]
    tests += [(g, np.zeros(HW, np.uint8)) for g in goods[:len(df) - k]]
    return normals, fit, tests


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], SIZE) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    fit = [(_load_img(R / x["image_path"], SIZE), _read(R / x["mask_path"], HW)) for x in ng[:30]]
    tests = [(_load_img(R / x["image_path"], SIZE), _read(R / x["mask_path"], HW)) for x in ng[30:70]]
    tests += [(_load_img(R / x["image_path"], SIZE), np.zeros(HW, np.uint8))
              for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, fit, tests


def main():
    torch.manual_seed(0)
    print("=== 表征能力 × 定位best-IoU:EAD残差 vs WRN50 vs DINOv2 ===")
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("电子 pcb", lambda: prep_realiad("pcb")),
        ("电池 phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    wrn = WRNFeat(); dino = DinoFeat()
    agg = {"EAD残差": [], "WRN50": [], "DINOv2": []}
    for name, prep in jobs:
        normals, fit, tests = prep()
        ead = EADFeat(normals)
        for tag, fn in [("EAD残差", ead), ("WRN50", wrn), ("DINOv2", dino)]:
            X, y = gather(fn, fit); h = train_head(X, y)
            iou = evalb(fn, *h, tests); agg[tag].append(iou)
            print(f"  {name:20s} {tag:8s} best-IoU={iou:.3f}", flush=True)
    print("\n均值 best-IoU:", {k: round(np.mean(v), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()

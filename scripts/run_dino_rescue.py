"""DINOv2记忆库图级补检(AnomalyDINO/SuperAD式,training-free):
洞察=v1-v6救援失败因seg-head监督信号过拟合漂移;DINOv2记忆距离training-free→阈值可迁移。
量:①DINOv2记忆图级AUROC(信号质量)②EAD漏检中DINOv2能救回多少③B半安全阈值下
   EAD-only acc vs EAD+DINO补检 acc(正常误报有没有涨)。
用法:python scripts/run_dino_rescue.py
"""
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from aoi.efficientad import EfficientADDetector
from eval.protocol import image_auroc
from eval.mvtec import _load_img

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GT = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
DINO_SZ = 518; BANK_MAX = 40000; TOPQ = 0.01


class DinoBank:
    """DINOv2 patch记忆库(AnomalyDINO式,余弦最近邻,training-free)。"""
    def __init__(self, normals):
        self.m = timm.create_model("vit_small_patch14_dinov2", pretrained=True,
                                   dynamic_img_size=True).eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        vs = [self._patches(n) for n in normals]
        V = torch.cat(vs)
        g = torch.Generator().manual_seed(0)
        if V.shape[0] > BANK_MAX:
            V = V[torch.randperm(V.shape[0], generator=g)[:BANK_MAX]]
        self.bank = F.normalize(V, dim=1).half().to(DEV)

    @torch.no_grad()
    def _patches(self, img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(DEV)
        x = F.interpolate(x, size=(DINO_SZ, DINO_SZ), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        t = self.m.forward_features(x)[:, self.m.num_prefix_tokens:, :]   # (1,N,C)
        return t[0].float().cpu()

    @torch.no_grad()
    def score(self, img):
        q = F.normalize(self._patches(img).to(DEV), dim=1).half()
        d = []
        for i in range(0, q.shape[0], 2048):
            sim = q[i:i + 2048] @ self.bank.t()
            d.append(1 - sim.max(dim=1).values.float())
        dm = torch.cat(d).cpu().numpy()
        k = max(1, int(len(dm) * TOPQ))
        return float(np.sort(dm)[-k:].mean())              # top-1% patch距离均值


def _read(p, hw):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def prep_mvtec(cat, folders):
    root = Path(f"data/mvtec/{cat}")
    normals = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [_load_img(p, 640) for p in sorted(glob.glob(str(root / "test/good/*.png")))[:40]]
    defs = []
    for fo in folders:
        defs += [_load_img(p, 640) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(defs)
    return normals, defs, goods


def prep_realiad(cat):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [_load_img(R / x["image_path"], 640) for x in tok[:100]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    defs = [_load_img(R / x["image_path"], 640) for x in ng]
    goods = [_load_img(R / x["image_path"], 640) for x in d["test"] if x["anomaly_class"] == "OK"][:40]
    return normals, defs, goods


def run(name, normals, defs, goods):
    fit_n, fit_d = normals[:60], defs[:30]
    test_d, test_g = defs[30:70], goods
    # EAD 图级
    ead = EfficientADDetector(model_size="small", device=DEV, train_steps=8000)
    ead.fit_fewshot(fit_n, None)
    ns = [ead._image_score(x)[0] for x in fit_n]
    ds = [ead._image_score(x)[0] for x in fit_d]
    from aoi.fewshot import FewShotAdapter
    thr = FewShotAdapter._calibrate(ns, ds)
    # DINOv2 记忆库
    bank = DinoBank(fit_n[:40])
    dn = [bank.score(x) for x in fit_n]
    # B半安全阈值(A建线 max+尾宽, B验零误报, 需≥3 fit缺陷可救)
    A, B = np.array(dn[:30]), np.array(dn[30:])
    dbar = float(A.max() + (A.max() - np.percentile(A, 90)) + 1e-6)
    fit_d_ead = [ead._image_score(x)[0] for x in fit_d]
    fit_d_dino = [bank.score(x) for x in fit_d]
    miss_fit = [(e, dv) for e, dv in zip(fit_d_ead, fit_d_dino) if e < thr]   # EAD漏的
    b_false = (B > dbar).sum()
    rescuable = sum(dv > dbar for _, dv in miss_fit)
    enable = (b_false == 0 and rescuable >= 3)
    # 测试
    ead_d = [ead._image_score(x)[0] for x in test_d]; dino_d = [bank.score(x) for x in test_d]
    ead_g = [ead._image_score(x)[0] for x in test_g]; dino_g = [bank.score(x) for x in test_g]
    # DINOv2 独立图级AUROC
    au = image_auroc(dino_d + dino_g, [1]*len(dino_d) + [0]*len(dino_g))
    # EAD-only vs +DINO补检
    def acc(rescue):
        cor = 0
        for e, dv in zip(ead_d, dino_d):
            det = e >= thr or (rescue and e >= 0.8*thr and dv > dbar)
            cor += det
        for e, dv in zip(ead_g, dino_g):
            det = e >= thr or (rescue and e >= 0.8*thr and dv > dbar)
            cor += (not det)
        return cor / (len(ead_d) + len(ead_g))
    a0, a1 = acc(False), acc(True)
    # EAD漏检里DINOv2实际救回率 + 正常误翻
    miss_t = [dv for e, dv in zip(ead_d, dino_d) if e < thr]
    resc_t = sum(dv > dbar and e >= 0.8*thr for e, dv in zip(ead_d, dino_d) if e < thr)
    print(f"{name:14s} DINO-AUROC={au:.3f} | EAD漏检{len(miss_t)}张 DINO救回{resc_t}张 | "
          f"补检启用={enable} acc {a0:.3f}→{a1:.3f}", flush=True)
    return au, a0, a1


def main():
    torch.manual_seed(0)
    print("=== DINOv2记忆库图级补检(training-free,治EAD漏检)===")
    jobs = [("pcb", lambda: prep_realiad("pcb")),
            ("battery", lambda: prep_realiad("phone_battery")),
            ("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
            ("pill", lambda: prep_mvtec("pill", ["color"])),
            ("cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"]))]
    for name, prep in jobs:
        run(name, *prep())


if __name__ == "__main__":
    main()

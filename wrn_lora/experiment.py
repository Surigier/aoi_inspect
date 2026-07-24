"""WRN-LoRA三组受控对照实验:①冻结基线(n_late_blocks=0,无LoRA)②layer2最后1块
conv2,rank=2③layer2最后2块conv2,rank=4。三组用【同一套训练循环/损失/超参数】,
唯一变量是LoRA配置——不同时改损失/增广/阈值,才能判断增益真的来自LoRA。

真实数据(AD2 sheet_metal/walnuts/fruit_jelly,已有掩膜,今天反复用过的验证床)×
3个随机种子,报中位数IoU+最差类别IoU双门槛:只有中位数提升且最差类别不明显下降
才算通过,单点最优值不算数。
用法:PYTHONPATH=. python wrn_lora/experiment.py
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from wrn_lora.lora_backbone import build_lora_wrn
from scripts.run_seg_head_ab import prep_ad2, HW

CATS = ["sheet_metal", "walnuts", "fruit_jelly"]
SEEDS = [0, 1, 2]
SEG_IN = 512


def _mask_to(mask_hw, h, w):
    m = torch.from_numpy(mask_hw.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="nearest")[0, 0]
    return (m.numpy() > 0.5).astype(np.uint8)


def _linear_head(C):
    return nn.Conv2d(C, 1, 1)


def _conv_head(C, mid=64):
    return nn.Sequential(nn.Conv2d(C, mid, 1), nn.ReLU(True),
                         nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(True),
                         nn.Conv2d(mid, 1, 1))


class _Ensemble(nn.Module):
    def __init__(self, *heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        return torch.stack([h(x) for h in self.heads], 0).mean(0)


def per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def run_one(cat, seed, n_late_blocks, r, lora_lr=2e-4, head_lr=5e-3, steps=150, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); random.seed(seed)
    normals, fit_i, fit_m, test_defs = prep_ad2(cat)

    bb, extractor, lora_modules, lora_params = build_lora_wrn(
        device=device, n_late_blocks=max(n_late_blocks, 1) if n_late_blocks > 0 else 1, r=r)
    if n_late_blocks == 0:
        for lm in lora_modules:                                # 基线组:LoRA权重锁死在零初始化(=纯冻结base)
            for p in list(lm.down.parameters()) + list(lm.up.parameters()):
                p.requires_grad_(False)
        lora_params = []
    elif n_late_blocks == 1:
        lora_modules = lora_modules[-1:]                        # build_lora_wrn(n_late_blocks=1)已只建1个,这里保险裁剪
        lora_params = []
        for lm in lora_modules:
            lora_params += list(lm.down.parameters()) + list(lm.up.parameters())

    probe = extractor(fit_i[0].to(device))
    C, gh, gw = probe.shape

    torch.manual_seed(seed)
    lin = _linear_head(C).to(device); cnv = _conv_head(C).to(device)
    head = _Ensemble(lin, cnv)
    param_groups = [{"params": list(head.parameters()), "lr": head_lr}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr})
    opt = torch.optim.Adam(param_groups, weight_decay=1e-4)

    gts_native = fit_m
    n = len(fit_i)
    pos_total = sum(float(_mask_to(m, gh, gw).sum()) for m in gts_native)
    neg_total = n * gh * gw - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    all_imgs = fit_i + normals[:20]
    all_gts = [torch.from_numpy(_mask_to(m, gh, gw).astype(np.float32)) for m in gts_native] + \
              [torch.zeros(gh, gw) for _ in range(min(20, len(normals)))]

    g = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(steps):
        i = int(torch.randint(0, len(all_imgs), (1,), generator=g).item())
        img = all_imgs[i].to(device)
        feat = extractor(img)[None]                            # 有梯度(LoRA参数需要),backbone其余部分冻结
        logit = head(feat)
        y = all_gts[i][None].to(device)
        loss = lossf(logit.squeeze(1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    with torch.no_grad():
        S, L = [], []
        for img, mk in zip(fit_i, fit_m):
            feat = extractor(img.to(device))[None]
            logit = head(feat)[0, 0].cpu().numpy()
            S.append(logit.ravel()); L.append(_mask_to(mk, gh, gw).ravel())
        s = np.concatenate(S); l = np.concatenate(L)
        order = np.argsort(-s); ls = l[order]; ss = s[order]
        tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = max(int(ls.sum()), 1)
        f1 = 2 * (tp / np.maximum(tp + fp, 1)) * (tp / P) / np.maximum(tp / np.maximum(tp + fp, 1) + tp / P, 1e-9)
        thr = float(ss[int(np.argmax(f1))])

        ious = []
        for img, mk in test_defs:
            feat = extractor(img.to(device))[None]
            logit = head(feat)[0, 0].cpu().numpy()
            pred = (logit >= thr).astype(np.uint8)
            gt = _mask_to(mk, gh, gw)
            ious.append(per_image_iou(pred, gt))
    return float(np.mean(ious))


def main():
    groups = [("baseline(冻结,无LoRA)", 0, 2), ("LoRA_r2_last1block", 1, 2), ("LoRA_r4_last2blocks", 2, 4)]
    results = {name: [] for name, _, _ in groups}
    for cat in CATS:
        for seed in SEEDS:
            row = []
            for name, nb, r in groups:
                iou = run_one(cat, seed, nb, r)
                results[name].append(iou)
                row.append(f"{name}={iou:.3f}")
            print(f"{cat} seed={seed}  " + "  ".join(row), flush=True)

    print("\n=== 汇总(跨3类×3种子=9次) ===", flush=True)
    for name, _, _ in groups:
        vals = results[name]
        print(f"{name:24s} 中位数={np.median(vals):.3f}  均值={np.mean(vals):.3f}  最差={min(vals):.3f}", flush=True)
    base_med = np.median(results["baseline(冻结,无LoRA)"])
    base_min = min(results["baseline(冻结,无LoRA)"])
    for name, _, _ in groups[1:]:
        med, mn = np.median(results[name]), min(results[name])
        verdict = "✅通过(中位数提升且最差类别不明显下降)" if (med > base_med and mn >= base_min - 0.01) else "❌不通过"
        print(f"{name}: Δ中位数={med-base_med:+.3f} Δ最差={mn-base_min:+.3f}  {verdict}", flush=True)


if __name__ == "__main__":
    main()

"""WRN-LoRA封存前三项收尾诊断(不再跑大搜索):判断是否真的产生了可测的更新,还是
保守配置(lr=2e-4/150步/batch=1)下适配器没有真正动起来。
①每个LoRA层 ||ΔW||_F/||W_base||_F(合并后权重相对变化幅度)
②同一张测试图 mean/p95(|logit_lora - logit_base|)(特征/输出层面的真实变化)
③fit IoU vs test IoU(baseline vs LoRA,区分"没学""学了但只在fit上过拟合""真的没价值")
外加唯一一次压力测试(fruit_jelly seed1, r4/last2, lr=1e-3, steps=300,只为确认
LoRA机制本身能不能真的改变模型,不作为成绩证据)。
用法:PYTHONPATH=. python wrn_lora/diagnose.py
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from wrn_lora.lora_backbone import build_lora_wrn
from wrn_lora.experiment import _mask_to, _linear_head, _conv_head, _Ensemble, per_image_iou
from scripts.run_seg_head_ab import prep_ad2


def train_and_collect(cat, seed, n_late_blocks, r, lora_lr=2e-4, head_lr=5e-3, steps=150, device=None,
                       prep_fn=prep_ad2):
    """训一组,返回(head, extractor, lora_modules, fit_iou, test_iou, thr, test_defs)。
    prep_fn(cat)默认prep_ad2(仅测大图延时/形状用途),换成prep_mvtec等生产类目
    数据源时传入prep_fn即可,不改变其余逻辑。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); random.seed(seed)
    normals, fit_i, fit_m, test_defs = prep_fn(cat)

    bb, extractor, lora_modules, lora_params = build_lora_wrn(
        device=device, n_late_blocks=max(n_late_blocks, 1) if n_late_blocks > 0 else 1, r=r)
    if n_late_blocks == 0:
        for lm in lora_modules:
            for p in list(lm.down.parameters()) + list(lm.up.parameters()):
                p.requires_grad_(False)
        lora_params, lora_modules = [], []
    elif n_late_blocks == 1:
        lora_modules = lora_modules[-1:]
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

    pos_total = sum(float(_mask_to(m, gh, gw).sum()) for m in fit_m)
    neg_total = len(fit_i) * gh * gw - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    all_imgs = fit_i + normals[:20]
    all_gts = [torch.from_numpy(_mask_to(m, gh, gw).astype(np.float32)) for m in fit_m] + \
              [torch.zeros(gh, gw) for _ in range(min(20, len(normals)))]

    g = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(steps):
        i = int(torch.randint(0, len(all_imgs), (1,), generator=g).item())
        img = all_imgs[i].to(device)
        feat = extractor(img)[None]
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

        fit_ious = []
        for img, mk in zip(fit_i, fit_m):
            feat = extractor(img.to(device))[None]
            logit = head(feat)[0, 0].cpu().numpy()
            pred = (logit >= thr).astype(np.uint8)
            fit_ious.append(per_image_iou(pred, _mask_to(mk, gh, gw)))

        test_ious = []
        for img, mk in test_defs:
            feat = extractor(img.to(device))[None]
            logit = head(feat)[0, 0].cpu().numpy()
            pred = (logit >= thr).astype(np.uint8)
            test_ious.append(per_image_iou(pred, _mask_to(mk, gh, gw)))

    return head, extractor, lora_modules, float(np.mean(fit_ious)), float(np.mean(test_ious)), thr, test_defs, device


def weight_delta_ratio(lora_modules):
    out = []
    for lm in lora_modules:
        merged = lm.merged_conv()
        base_norm = lm.base.weight.norm().item()
        delta_norm = (merged.weight - lm.base.weight).norm().item()
        out.append(delta_norm / max(base_norm, 1e-9))
    return out


@torch.no_grad()
def logit_delta_stats(head_base, ext_base, head_lora, ext_lora, test_defs, device, n=10):
    diffs = []
    for img, _ in test_defs[:n]:
        img = img.to(device)
        fb = ext_base(img)[None]; lb = head_base(fb)[0, 0].cpu().numpy()
        fl = ext_lora(img)[None]; ll = head_lora(fl)[0, 0].cpu().numpy()
        diffs.append(np.abs(lb - ll).ravel())
    all_d = np.concatenate(diffs)
    return float(all_d.mean()), float(np.percentile(all_d, 95))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== 三项收尾诊断(sheet_metal/walnuts seed0, fruit_jelly seed1, LoRA_r4_last2) ===", flush=True)
    for cat, seed in [("sheet_metal", 0), ("walnuts", 0), ("fruit_jelly", 1)]:
        h_base, e_base, _, fit_b, test_b, thr_b, test_defs, _ = train_and_collect(cat, seed, 0, 2, device=device)
        h_lora, e_lora, lms, fit_l, test_l, thr_l, _, _ = train_and_collect(cat, seed, 2, 4, device=device)

        ratios = weight_delta_ratio(lms)
        mean_d, p95_d = logit_delta_stats(h_base, e_base, h_lora, e_lora, test_defs, device)
        print(f"{cat} seed={seed}", flush=True)
        print(f"  ①各层||ΔW||/||W_base||: {[f'{r:.2e}' for r in ratios]}", flush=True)
        print(f"  ②logit差异 mean={mean_d:.4f} p95={p95_d:.4f}", flush=True)
        print(f"  ③fit IoU: base={fit_b:.3f} lora={fit_l:.3f} (Δ={fit_l-fit_b:+.3f})  |  "
              f"test IoU: base={test_b:.3f} lora={test_l:.3f} (Δ={test_l-test_b:+.3f})", flush=True)

    print("\n=== 唯一压力测试:fruit_jelly seed1, r4/last2, lr=1e-3, steps=300 ===", flush=True)
    h_base, e_base, _, fit_b, test_b, _, test_defs, _ = train_and_collect("fruit_jelly", 1, 0, 2, device=device)
    h_lora, e_lora, lms, fit_l, test_l, _, _, _ = train_and_collect(
        "fruit_jelly", 1, 2, 4, lora_lr=1e-3, steps=300, device=device)
    ratios = weight_delta_ratio(lms)
    mean_d, p95_d = logit_delta_stats(h_base, e_base, h_lora, e_lora, test_defs, device)
    print(f"①各层||ΔW||/||W_base||: {[f'{r:.2e}' for r in ratios]}", flush=True)
    print(f"②logit差异 mean={mean_d:.4f} p95={p95_d:.4f}", flush=True)
    print(f"③fit IoU: base={fit_b:.3f} lora={fit_l:.3f} (Δ={fit_l-fit_b:+.3f})  |  "
          f"test IoU: base={test_b:.3f} lora={test_l:.3f} (Δ={test_l-test_b:+.3f})", flush=True)


if __name__ == "__main__":
    main()

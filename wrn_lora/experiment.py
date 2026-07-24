"""WRN-LoRA三组受控对照实验:①冻结基线(n_late_blocks=0,无LoRA)②layer2最后1块
conv2,rank=2③layer2最后2块conv2,rank=4。三组用【同一套训练循环/损失/超参数】,
唯一变量是LoRA配置——不同时改损失/增广/阈值,才能判断增益真的来自LoRA。

真实数据(AD2 sheet_metal/walnuts/fruit_jelly,已有掩膜,今天反复用过的验证床)×
3个随机种子,报配对margin判定(见main()末尾):median(Δ)>=0.005 且 mean(Δ)>0 且
>=2/3类中位数为正 且 min(Δ)>=-0.01,单点最优值不算数。

【已判负,结论封存】9次跑(3类×3种子)结果:LoRA_r2 median(Δ)=0.000 mean(Δ)=+0.003
类中位数=[0.000,0.000,0.000](0/3为正) min(Δ)=-0.005 → 不通过;LoRA_r4
median(Δ)=+0.001 mean(Δ)=+0.004 类中位数=[+0.001,0.000,+0.013](2/3为正)
min(Δ)=0.000 → 不通过(中位数增益远低于0.005门槛)。sheet_metal/walnuts跨种子
稳定打平,fruit_jelly出现的+0.013~+0.034离群正向不能归因为稳定LoRA收益(该类
本身是对头部/阈值高方差敏感类,新旧分割头A/B也在该类出现过约0.092反号差异)。

封存前诊断(见diagnose.py,排除"配置太保守/适配器没动"的可能性):①各层
||ΔW||_F/||W_base||_F 在保守配置(lr=2e-4,150步)下已达1.3e-3~1.7e-3,远高于
"没动"的1e-4门槛,LoRA权重确实产生了可测更新;②同测试图logit差异sheet_metal/
walnuts mean仅0.006~0.009(权重动了但输出几乎不变),fruit_jelly mean=0.28
(该类对同样幅度的权重扰动格外敏感,与其本身高方差体质一致);③fit/test IoU
保守配置下同向小幅变化(无fit涨test不涨的过拟合特征)。唯一压力测试(fruit_jelly
seed1,r4/last2,lr=1e-3,steps=300,仅为验证LoRA机制能否真的移动模型,不作为
成绩证据)显示权重可大幅移动(ΔW/W达2.9e-2~4.5e-2)且fit(+0.108)/test(+0.114)
同向大涨——fruit_jelly上确实可能存在类别特定的真实收益,压力测试没有排除这一点。
**结论(措辞更正,避免过度概括):不能说"WRN表示已经足够"或"绝非超参数问题"——
能确定的只是,在已测的配置范围内收益不广泛、不稳定(2/3类打平),继续在这个方向
调参(找更适合fruit_jelly这类高方差类别的超参/rank)的竞赛期望值低,默认关闭,
保留为零时延opt-in研究件。** 资源转回Top-1参考ROI精修和RDDN-YOLO。

用法:PYTHONPATH=. python wrn_lora/experiment.py(封存实验,不再新增大搜索)
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
    baseline = np.array(results["baseline(冻结,无LoRA)"])
    for name, _, _ in groups[1:]:
        candidate = np.array(results[name])
        deltas = candidate - baseline                          # 配对差(同cat/同seed一一对应),不是组间中位数相减
        ns = len(SEEDS)
        category_medians = [np.median(deltas[i * ns:(i + 1) * ns]) for i in range(len(CATS))]  # results按cat外层/seed内层循环填充,连续ns个一段=同一类别
        passed = (
            np.median(deltas) >= 0.005
            and np.mean(deltas) > 0
            and sum(cm > 0 for cm in category_medians) >= 2
            and np.min(deltas) >= -0.01
        )
        verdict = "✅通过(有margin的配对判定)" if passed else "❌不通过"
        print(f"{name}: Δ中位数(配对)={np.median(deltas):+.3f} Δ均值={np.mean(deltas):+.3f} "
              f"各类Δ中位数={[f'{cm:+.3f}' for cm in category_medians]} Δ最小={np.min(deltas):+.3f}  {verdict}",
              flush=True)


if __name__ == "__main__":
    main()

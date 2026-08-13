"""诊断pcb为什么在激进配置(900步/1e-2)下比base(300步/5e-3)差,不是直接排除
这一类,是先搞清楚:是步数太多学过头了,还是学习率太高不稳定,还是两者都有份。
用pcb真实fit数据(和gated_train.py同一套train_sub/val_sub切分),沿训练过程
打checkpoint记录val_sub IoU曲线,拆成两条对照:
  A: lr固定5e-3(base学习率),步数300→900看纯步数的影响
  B: 步数固定300(base步数),lr从5e-3→1e-2看纯学习率的影响
用法:PYTHONPATH=. python seghead_tuning/diag_pcb.py
"""
import random
import numpy as np
import torch
import torch.nn as nn
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to
from seghead_tuning.probe_aggressive_train import _calibrate_thr, _per_image_iou, BATCH, SEED, DEV
from global_context.eval_global_branch import prep_realiad

CKPTS = [100, 200, 300, 450, 600, 750, 900]


def train_with_checkpoints(feats, gts, pos_w, lr, device=DEV):
    """一次训练里在CKPTS各个步数打点,返回{steps: (head,mu,sd)}(head是深拷贝)。"""
    import copy
    C = feats[0].shape[0]
    mu = torch.stack([f.float().mean(dim=(1, 2)) for f in feats]).mean(0).view(1, -1, 1, 1).to(device)
    sd = (torch.stack([f.float().std(dim=(1, 2)) for f in feats]).mean(0) + 1e-6).view(1, -1, 1, 1).to(device)
    all_feats = [f.to(device) for f in feats]
    all_gts = [g.to(device) for g in gts]

    torch.manual_seed(SEED)
    lin = _ProdHead._linear_head(C).to(device)
    cnv = _ProdHead._conv_head(C).to(device)
    head = _Ensemble(lin, cnv)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(SEED)
    n = len(all_feats)
    ckpts = {}
    for step in range(1, max(CKPTS) + 1):
        sel = torch.randperm(n, generator=g)[:BATCH].tolist()
        X = torch.stack([all_feats[i] for i in sel])
        X = (X - mu) / sd
        Y = torch.stack([all_gts[i] for i in sel])
        opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
        if step in CKPTS:
            head.eval()
            ckpts[step] = copy.deepcopy(head)
            head.train()
    return ckpts, mu, sd


@torch.no_grad()
def eval_head(extractor, head, mu, sd, defs, device=DEV):
    head.eval()
    ious = []
    for img, mk in defs:
        f = extractor(img)[None].to(device).float()
        logit = head((f - mu) / sd)
        gh, gw = logit.shape[-2:]
        pred = (logit[0, 0].cpu().numpy() >= 0).astype(np.uint8)  # 用0(logit中性点)近似看趋势,不需要精确thr
        ious.append(_per_image_iou(pred, _mask_to(mk, gh, gw)))
    return float(np.mean(ious)) if ious else 0.0


def main():
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs = prep_realiad("pcb")
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    extractor = det.seg_head.extractor

    n = len(fit_i)
    idx = list(range(n)); random.Random(0).shuffle(idx)
    n_val = max(1, int(round(n * 0.3)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_i = [fit_i[i] for i in train_idx]; train_m = [fit_m[i] for i in train_idx]
    val_defs = [(fit_i[i], fit_m[i]) for i in val_idx]
    print(f"pcb fit={n} -> train_sub={len(train_idx)} val_sub={len(val_idx)}", flush=True)

    with torch.no_grad():
        feats = [extractor(im) for im in train_i]
    grid_hw = feats[0].shape[-2:]
    gts = [torch.from_numpy(_mask_to(m, grid_hw[0], grid_hw[1]).astype(np.float32)) for m in train_m]
    pos_total = sum(float(g.sum()) for g in gts)
    neg_total = len(gts) * grid_hw[0] * grid_hw[1] - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=DEV)

    print("\n=== A: lr固定5e-3(base学习率),步数300->900看纯步数影响 ===", flush=True)
    ckpts_a, mu_a, sd_a = train_with_checkpoints(feats, gts, pos_w, lr=5e-3)
    for step in CKPTS:
        v = eval_head(extractor, ckpts_a[step], mu_a, sd_a, val_defs)
        print(f"  steps={step:4d} lr=5e-3  val_iou(近似,thr=0)={v:.3f}", flush=True)

    print("\n=== B: 步数固定300(base步数),lr从5e-3变到1e-2看纯学习率影响 ===", flush=True)
    for lr in [5e-3, 7e-3, 1e-2]:
        ckpts_b, mu_b, sd_b = train_with_checkpoints(feats, gts, pos_w, lr=lr)
        v = eval_head(extractor, ckpts_b[300], mu_b, sd_b, val_defs)
        print(f"  steps=300 lr={lr:.0e}  val_iou(近似,thr=0)={v:.3f}", flush=True)


if __name__ == "__main__":
    main()

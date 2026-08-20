"""验证LoRA实验的意外副产物:**双头联合训练 vs 各自独立训**。

生产`_seg_head_old_ae5fbbb.fit()`把线性头和卷积头各自独立训300步
(`_train_one(lin)`、`_train_one(cnv)`),训完才包成`_Ensemble`——两个头训练期间
互不影响。而LoRA探针脚本里是把`_Ensemble`当整体联合训(梯度同时流过两个头,
让它们协同分工)。LoRA空转那次(LoRA实际没生效,等于纯粹的头训练方式对照)
测出median ΔIoU=+0.003、框命中+0.026,其中pill框命中0.426→0.603(+0.177)在
LoRA生效/不生效两次跑里数字分毫不差——说明这个提升确实来自训练方式差异。

这个改动不碰骨干、推理结构完全不变(还是同一个_Ensemble)、零延时、零假阳性
风险(seg_head只在is_defect=True后才调用),是目前唯一未验证的免费候选。

判据同session口径:median(Δ)>=0.005 且 min(Δ)>=-0.01,同时Δacc>=-0.01。

用法:PYTHONPATH=. python seghead_tuning/probe_joint_ensemble.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR, BATCH, SEED = 300, 5e-3, 8, 0        # 和生产_train_one完全一致


def train_joint(feats, gts, pos_w, device=DEV):
    """把_Ensemble当整体联合训——唯一和生产不同的地方(生产是lin/cnv各训各的)。
    其余loss/优化器/步数/lr/batch/种子/归一化全部照抄生产_train_one。"""
    C = feats[0].shape[0]
    mu = torch.stack([f.float().mean(dim=(1, 2)) for f in feats]).mean(0).view(1, -1, 1, 1).to(device)
    sd = (torch.stack([f.float().std(dim=(1, 2)) for f in feats]).mean(0) + 1e-6).view(1, -1, 1, 1).to(device)
    all_feats = [f.to(device) for f in feats]
    all_gts = [g.to(device) for g in gts]
    torch.manual_seed(SEED)
    lin = _ProdHead._linear_head(C).to(device)
    cnv = _ProdHead._conv_head(C).to(device)
    head = _Ensemble(lin, cnv)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(SEED)
    n = len(all_feats)
    head.train()
    for _ in range(STEPS):
        sel = torch.randperm(n, generator=g)[:BATCH].tolist()
        X = (torch.stack([all_feats[i] for i in sel]) - mu) / sd
        Y = torch.stack([all_gts[i] for i in sel])
        opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
    head.eval()
    return head, mu, sd


@torch.no_grad()
def calibrate_thr(det, head, mu, sd, imgs, masks, out_hw=(256, 256)):
    extractor = det.seg_head.extractor
    S, L = [], []
    for img, mk in zip(imgs, masks):
        f = extractor(img)[None].to(DEV).float()
        logit = head((f - mu) / sd)
        amap = torch.nn.functional.interpolate(logit, size=out_hw, mode="bilinear", align_corners=False)
        S.append(amap[0, 0].cpu().numpy().ravel())
        L.append(_mask_to(mk, out_hw[0], out_hw[1]).ravel())
    s = np.concatenate(S); l = np.concatenate(L)
    order = np.argsort(-s); ls = l[order]; ss = s[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    if P == 0:
        return None
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(ss[int(np.argmax(f1))])


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def eval_pipeline(det, test_defs, goods):
    ious, hits = [], []
    tp_def = 0
    for img, gt in test_defs:
        o = det.locate(img)
        if o["is_defect"]:
            tp_def += 1
        if o.get("mask") is None:
            ious.append(0.0); hits.append(0.0); continue
        mask = o["mask"]
        gt_r = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        ious.append(_per_image_iou(mask, gt_r))
        hits.append(box_hit(o["boxes"], gt_boxes(gt)) or 0.0)
    tn_good = sum(1 for img in goods if not det.locate(img)["is_defect"])
    acc = (tp_def + tn_good) / max(len(test_defs) + len(goods), 1)
    return float(np.mean(ious)), float(np.mean(hits)), acc


def run_one(name, normals, fit_i, fit_m, test_defs, goods):
    t0 = time.time()
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"  [{name}] fit完成 {time.time()-t0:.0f}s", flush=True)
    if det.seg_head.head is None:
        print(f"{name}: seg_head未训成功,跳过", flush=True); return None
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)

    b_iou, b_hit, b_acc = eval_pipeline(det, test_defs, goods)
    print(f"  [{name}] baseline(各自独立训) IoU={b_iou:.3f} hit={b_hit:.3f} acc={b_acc:.3f}", flush=True)

    extractor = det.seg_head.extractor
    with torch.no_grad():
        feats = [extractor(im) for im in fit_i]
    gh, gw = feats[0].shape[-2:]
    gts = [torch.from_numpy(_mask_to(m, gh, gw).astype(np.float32)) for m in fit_m]
    pos_total = sum(float(g.sum()) for g in gts)
    neg_total = len(gts) * gh * gw - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=DEV)

    orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
    try:
        head, mu, sd = train_joint(feats, gts, pos_w)
        thr = calibrate_thr(det, head, mu, sd, fit_i, fit_m)
        if thr is None:
            print(f"{name}: 阈值标定失败,跳过", flush=True); return None
        det.seg_head.head, det.seg_head.mu, det.seg_head.sd = head, mu, sd
        det.seg_head.thr = thr; det.seg_head.rams = None
        j_iou, j_hit, j_acc = eval_pipeline(det, test_defs, goods)
    finally:
        (det.seg_head.head, det.seg_head.mu, det.seg_head.sd,
         det.seg_head.thr, det.seg_head.rams) = orig

    print(f"{name:20s} 独立训 IoU={b_iou:.3f}/hit={b_hit:.3f}/acc={b_acc:.3f} | "
          f"联合训 IoU={j_iou:.3f}/hit={j_hit:.3f}/acc={j_acc:.3f} | "
          f"ΔIoU={j_iou-b_iou:+.3f} Δhit={j_hit-b_hit:+.3f} Δacc={j_acc-b_acc:+.3f}", flush=True)
    return dict(base=(b_iou, b_hit, b_acc), joint=(j_iou, j_hit, j_acc))


def main():
    torch.manual_seed(0)
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("生产:pcb", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
    ]
    names, di, dh, da = [], [], [], []
    for name, prep in jobs:
        normals, fit_i, fit_m, test_defs = prep()
        goods = normals[-20:]
        row = run_one(name, normals[:-20], fit_i, fit_m, test_defs, goods)
        if row is None:
            continue
        names.append(name)
        di.append(row["joint"][0] - row["base"][0])
        dh.append(row["joint"][1] - row["base"][1])
        da.append(row["joint"][2] - row["base"][2])
    if di:
        a = np.array(di); h = np.array(dh); c = np.array(da)
        passed = (np.median(a) >= 0.005 and np.mean(a) > 0
                 and (a > 0).sum() >= max(1, len(a) // 2 + 1) and np.min(a) >= -0.01
                 and np.min(c) >= -0.01)
        print(f"\n=== 汇总(n={len(a)}) === ΔIoU median={np.median(a):+.3f} mean={np.mean(a):+.3f} "
              f"min={np.min(a):+.3f} | Δhit mean={np.mean(h):+.3f} | Δacc min={np.min(c):+.3f} | "
              f"{'通过' if passed else '不通过'}", flush=True)
        print("ΔIoU:", dict(zip(names, [round(float(x), 3) for x in a])), flush=True)
        print("Δhit:", dict(zip(names, [round(float(x), 3) for x in h])), flush=True)


if __name__ == "__main__":
    main()

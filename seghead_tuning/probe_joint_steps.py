"""联合训练的步数没重新调过——现在的300步/5e-3是当年针对"两个头各自独立训"调出来的。
联合训练后优化目标变了(两个头一起训、可训参数翻倍、还要学会协同分工),300步可能
不够。之前测过的"激进配置900步/1e-2"是在**独立训**基础上测的(min=-0.017勉强不
过关),那个结论对联合训练不一定成立。

这里在**已上生产的联合训练**基础上,只变步数(lr保持5e-3不变,避免两个变量混在一起):
300(生产现状)vs 600 vs 900。零部署成本(推理结构完全不变)。

判据同session口径:median(Δ)>=0.005 且 min(Δ)>=-0.01,同时Δacc>=-0.01。

用法:PYTHONPATH=. python seghead_tuning/probe_joint_steps.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LR, BATCH, SEED = 5e-3, 8, 0
STEP_CANDIDATES = [300, 600, 900]        # 300=生产现状(基线)


def train_joint(feats, gts, pos_w, steps, device=DEV):
    C = feats[0].shape[0]
    mu = torch.stack([f.float().mean(dim=(1, 2)) for f in feats]).mean(0).view(1, -1, 1, 1).to(device)
    sd = (torch.stack([f.float().std(dim=(1, 2)) for f in feats]).mean(0) + 1e-6).view(1, -1, 1, 1).to(device)
    all_feats = [f.to(device) for f in feats]
    all_gts = [g.to(device) for g in gts]
    torch.manual_seed(SEED)
    head = _Ensemble(_ProdHead._linear_head(C).to(device), _ProdHead._conv_head(C).to(device))
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(SEED)
    n = len(all_feats)
    head.train()
    for _ in range(steps):
        sel = torch.randperm(n, generator=g)[:BATCH].tolist()
        X = (torch.stack([all_feats[i] for i in sel]) - mu) / sd
        Y = torch.stack([all_gts[i] for i in sel])
        opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
    head.eval()
    return head, mu, sd


@torch.no_grad()
def calibrate_thr(det, head, mu, sd, imgs, masks, out_hw=(256, 256)):
    ext = det.seg_head.extractor
    S, L = [], []
    for img, mk in zip(imgs, masks):
        logit = head((ext(img)[None].to(DEV).float() - mu) / sd)
        amap = torch.nn.functional.interpolate(logit, size=out_hw, mode="bilinear", align_corners=False)
        S.append(amap[0, 0].cpu().numpy().ravel()); L.append(_mask_to(mk, out_hw[0], out_hw[1]).ravel())
    s = np.concatenate(S); l = np.concatenate(L)
    order = np.argsort(-s); ls = l[order]; ss = s[order]
    tp = np.cumsum(ls); fp = np.cumsum(1 - ls); P = int(ls.sum())
    if P == 0:
        return None
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(ss[int(np.argmax(f1))])


def _iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def eval_pipeline(det, test_defs, goods):
    ious, hits = [], []
    tp = 0
    for img, gt in test_defs:
        o = det.locate(img)
        if o["is_defect"]:
            tp += 1
        if o.get("mask") is None:
            ious.append(0.0); hits.append(0.0); continue
        m = o["mask"]
        gr = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=m.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        ious.append(_iou(m, gr)); hits.append(box_hit(o["boxes"], gt_boxes(gt)) or 0.0)
    tn = sum(1 for img in goods if not det.locate(img)["is_defect"])
    return float(np.mean(ious)), float(np.mean(hits)), (tp + tn) / max(len(test_defs) + len(goods), 1)


def run_one(name, normals, fit_i, fit_m, test_defs, goods):
    t0 = time.time()
    det = CompetitionLargeDetector()          # joint_ensemble默认已开
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"  [{name}] fit完成 {time.time()-t0:.0f}s", flush=True)
    if det.seg_head.head is None:
        print(f"{name}: seg_head未训成功,跳过", flush=True); return None
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)

    ext = det.seg_head.extractor
    with torch.no_grad():
        feats = [ext(im) for im in fit_i]
    gh, gw = feats[0].shape[-2:]
    gts = [torch.from_numpy(_mask_to(m, gh, gw).astype(np.float32)) for m in fit_m]
    pos = sum(float(g.sum()) for g in gts); neg = len(gts) * gh * gw - pos
    pos_w = torch.tensor([neg / max(pos, 1)], device=DEV)

    orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
    out = {}
    try:
        for st in STEP_CANDIDATES:
            head, mu, sd = train_joint(feats, gts, pos_w, st)
            thr = calibrate_thr(det, head, mu, sd, fit_i, fit_m)
            if thr is None:
                continue
            det.seg_head.head, det.seg_head.mu, det.seg_head.sd = head, mu, sd
            det.seg_head.thr = thr; det.seg_head.rams = None
            out[st] = eval_pipeline(det, test_defs, goods)
            print(f"  [{name}] steps={st}: IoU={out[st][0]:.3f} hit={out[st][1]:.3f} acc={out[st][2]:.3f}", flush=True)
    finally:
        (det.seg_head.head, det.seg_head.mu, det.seg_head.sd,
         det.seg_head.thr, det.seg_head.rams) = orig
    return out


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
    names, rows = [], []
    for name, prep in jobs:
        normals, fit_i, fit_m, test_defs = prep()
        r = run_one(name, normals[:-20], fit_i, fit_m, test_defs, normals[-20:])
        if r and 300 in r:
            names.append(name); rows.append(r)
    for st in STEP_CANDIDATES[1:]:
        di = [r[st][0] - r[300][0] for r in rows if st in r]
        dh = [r[st][1] - r[300][1] for r in rows if st in r]
        da = [r[st][2] - r[300][2] for r in rows if st in r]
        if not di:
            continue
        a = np.array(di)
        ok = (np.median(a) >= 0.005 and np.mean(a) > 0
              and (a > 0).sum() >= max(1, len(a) // 2 + 1)
              and np.min(a) >= -0.01 and np.min(np.array(da)) >= -0.01)
        print(f"\n=== steps={st} vs 300(生产) n={len(a)} === ΔIoU median={np.median(a):+.3f} "
              f"mean={np.mean(a):+.3f} min={np.min(a):+.3f} | Δhit mean={np.mean(dh):+.3f} "
              f"| Δacc min={np.min(da):+.3f} | {'通过' if ok else '不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in a])), flush=True)


if __name__ == "__main__":
    main()

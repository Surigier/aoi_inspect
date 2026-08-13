"""给seg_head核心训练配方(保守300/5e-3 vs 激进900/1e-2)加OOF门控,和wrn_lora/
gated_validate.py同一套思路:30张fit缺陷图内部切train_sub/val_sub,两种配置都只在
train_sub上训,val_sub上比谁强就用谁,决定后再看真正held-out test_defs的表现。
动机:probe_aggressive_train.py测出来是6/10类受益、2类(wood/pcb)因为过拟合明显
变差(wood -0.089破了止损线)——和LoRA那次一模一样的模式,同样的门控解法应该适用。

用法:PYTHONPATH=. python seghead_tuning/gated_train.py
"""
import random
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import _mask_to
from seghead_tuning.probe_aggressive_train import (
    _train_head, _calibrate_thr, _per_image_iou,
    LIGHT_STEPS, LIGHT_LR, BASE_STEPS, BASE_LR, AGG_STEPS, AGG_LR, DEV,
)
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit
from scripts.run_scorecard_5types import prep_mvtec_color


@torch.no_grad()
def _eval_head_on(extractor, head, mu, sd, thr, defs, device=DEV):
    ious = []
    for img, mk in defs:
        f = extractor(img)[None].to(device).float()
        logit = head((f - mu) / sd)
        gh, gw = logit.shape[-2:]
        pred = (logit[0, 0].cpu().numpy() >= thr).astype(np.uint8)
        ious.append(_per_image_iou(pred, _mask_to(mk, gh, gw)))
    return float(np.mean(ious)) if ious else 0.0


def _fit_head_full(extractor, imgs, masks, steps, lr, device=DEV):
    """在给定图集上训一个头(和_train_head一样的loss/优化器/batch/种子),返回
    (head,mu,sd,thr)。供pick_gated_head两处复用:一次在train_sub上做门控决策,
    一次在决策完的全量fit集上重训最终部署的头。"""
    with torch.no_grad():
        feats = [extractor(im) for im in imgs]
    grid_hw = feats[0].shape[-2:]
    gts = [torch.from_numpy(_mask_to(m, grid_hw[0], grid_hw[1]).astype(np.float32)) for m in masks]
    pos_total = sum(float(g.sum()) for g in gts)
    neg_total = len(gts) * grid_hw[0] * grid_hw[1] - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=device)
    head, mu, sd = _train_head(feats, gts, pos_w, steps, lr, device=device)
    thr = _calibrate_thr(extractor, head, mu, sd, imgs, masks)
    return head, mu, sd, thr


CANDIDATES = [("轻量", LIGHT_STEPS, LIGHT_LR), ("保守", BASE_STEPS, BASE_LR), ("激进", AGG_STEPS, AGG_LR)]


def pick_gated_head(name, extractor, fit_i, fit_m, val_frac=0.3, seed=0, device=DEV):
    """内部切train_sub/val_sub,三档配置(轻量100/5e-3、保守300/5e-3、激进900/1e-2)
    都只在train_sub上训,val_sub上自检三选一——这只是"决策"阶段,便宜、不浪费太多
    计算。决策完之后,**用全量fit集重新训一遍胜出的配置**(以及保守配置的
    full-data版本当公平对照),避免"门控决策训练用了更少数据,导致和真实
    baseline(用全量fit集训)比较时不公平"这个混淆——这个bug在combined_probe.py
    组合测试里实际发生过。

    三档而非两档:diag_pcb.py诊断出pcb这类微小缺陷的真正训练甜点(~100步)比
    现有"保守"基线(300步)还早,"保守vs激进"这两个候选都已经过了它的峰值,门控
    怎么选都是次优——轻量档专门补上这个空缺。

    返回((h_base_full,mu,sd,thr), (h_final,mu,sd,thr,选中的档位名))——两者都是
    在全量fit_i/fit_m上训出来的,数据量对等,只有配置(steps/lr)不同。"""
    n = len(fit_i)
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_i = [fit_i[i] for i in train_idx]; train_m = [fit_m[i] for i in train_idx]
    val_defs = [(fit_i[i], fit_m[i]) for i in val_idx]
    print(f"  [{name}] fit={n} -> train_sub={len(train_idx)} val_sub={len(val_idx)}(仅用于门控决策)", flush=True)

    def decide(steps, lr):
        head, mu, sd, thr = _fit_head_full(extractor, train_i, train_m, steps, lr, device=device)
        if thr is None:
            return 0.0
        return _eval_head_on(extractor, head, mu, sd, thr, val_defs, device=device)

    vals = {tag: decide(steps, lr) for tag, steps, lr in CANDIDATES}
    winner = max(vals, key=vals.get)
    val_str = " ".join(f"{tag}={vals[tag]:.3f}" for tag, _, _ in CANDIDATES)
    print(f"  [{name}] val({val_str}) gate={winner}  -> 用全量fit={n}张重训部署版本", flush=True)

    # 决策完,保守(baseline对照)和胜出档都用全量fit集重训(数据量对等,唯一变量是steps/lr)
    base_cfg = _fit_head_full(extractor, fit_i, fit_m, BASE_STEPS, BASE_LR, device=device)
    if base_cfg[-1] is None:
        print(f"{name}: 全量重训阈值标定失败,跳过", flush=True)
        return None
    if winner == "保守":
        gated_cfg = (*base_cfg, winner)
    else:
        w_steps, w_lr = next((s, l) for tag, s, l in CANDIDATES if tag == winner)
        gated_full = _fit_head_full(extractor, fit_i, fit_m, w_steps, w_lr, device=device)
        if gated_full[-1] is None:
            print(f"{name}: 全量重训阈值标定失败,跳过", flush=True)
            return None
        gated_cfg = (*gated_full, winner)
    return base_cfg, gated_cfg


def _test_iou_with_head(det, head, mu, sd, thr, test_defs):
    orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd = head, mu, sd
    det.seg_head.thr = thr; det.seg_head.rams = None
    ious, hits = [], []
    for img, gt in test_defs:
        o = det.locate(img)
        if o.get("mask") is None:
            ious.append(0.0); hits.append(0.0); continue
        mask = o["mask"]
        gt_r = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        ious.append(_per_image_iou(mask, gt_r))
        hits.append(box_hit(o["boxes"], gt_boxes(gt)) or 0.0)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams = orig
    return float(np.mean(ious)), float(np.mean(hits))


def run_gated(name, normals, fit_i, fit_m, test_defs, val_frac=0.3, seed=0, device=DEV):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det.seg_head.head is None:
        print(f"{name}: 生产seg_head未训成功,跳过", flush=True)
        return None
    # DINO门可能被_calibrate_latency在GPU瞬时负载下砍掉(已知风险),不补会导致
    # test_base/test_gated用的判定路径跨进程不一致,IoU数字跟着运气跳——这正是
    # pcb在combined_probe.py和这里数字对不上(-0.016 vs -0.032)的根因之一。
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)
    extractor = det.seg_head.extractor

    picked = pick_gated_head(name, extractor, fit_i, fit_m, val_frac=val_frac, seed=seed, device=device)
    if picked is None:
        return None
    base_cfg, gated_cfg = picked
    h_gated, mu_g, sd_g, thr_g, winner = gated_cfg

    test_base, hit_base = _test_iou_with_head(det, *base_cfg, test_defs)
    test_gated, hit_gated = _test_iou_with_head(det, h_gated, mu_g, sd_g, thr_g, test_defs)
    d = test_gated - test_base
    print(f"{name}: gate={winner}  "
          f"test(base={test_base:.3f} gated={test_gated:.3f}) Δ(gated-base)={d:+.3f}", flush=True)
    return d


def main():
    torch.manual_seed(0)
    print("=== OOF门控seg_head训练配方:留出集自检,过拟合自动退回保守配置 ===", flush=True)
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("色彩 carpet", lambda: prep_mvtec_color("carpet")[:4]),
        ("色彩 leather", lambda: prep_mvtec_color("leather")[:4]),
        ("色彩 metal_nut", lambda: prep_mvtec_color("metal_nut")[:4]),
        ("色彩 wood", lambda: prep_mvtec_color("wood")[:4]),
        ("生产:pcb(微小缺陷)", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
    ]
    names, deltas = [], []
    for name, prep in jobs:
        d = run_gated(name, *prep())
        if d is None:
            continue
        names.append(name); deltas.append(d)

    d = np.array(deltas)
    passed = (np.median(d) >= 0.005 and np.mean(d) > 0
              and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
    print(f"\n=== 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} mean(Δ)={np.mean(d):+.3f} "
          f"min(Δ)={np.min(d):+.3f}  {'通过' if passed else '不通过'}", flush=True)
    print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

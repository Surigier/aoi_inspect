"""测"加深WRN语义信息"这个猜想:主分割头现在只用浅层(1,2)@512(run_feat_res.py
扫描出的选择,+47% IoU),没测过"浅层+深层拼在一起"这种叠加方案。项目里已有layer3
基础设施(_bb_l3,目前只给RAMS-R修正支用),这里直接拼进主extractor测。

假阳性风险结构上不存在(不需要重测):seg_head/WRN只在locate()里is_defect已经为
True之后才被调用,不参与图级判定本身,不会像GCAD-EmbedAE那样引入正常图误报。
唯一要额外测的是延时(多一次layer3前向的开销)。

用法:PYTHONPATH=. python seghead_tuning/probe_deep_extractor.py
"""
import time
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.backbone import Backbone
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to
from aoi.seg_head import map_to_boxes, merge_boxes
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit
from scripts.run_scorecard_5types import prep_mvtec_color

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR, BATCH, SEED = 300, 5e-3, 8, 0  # 和生产_train_one完全同配方,唯一变量是extractor


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def make_deep_extractor(det, l3_in=None):
    """浅层(1,2)+layer3拼接,按_rams_scales同款resize逻辑对齐到同一个格。
    l3_in:layer3自己的输入分辨率(None=和主特征一样用det._seg_in=512)。传更小的值
    (如256)可以把layer3前向成本降到~1/4——hazelnut实测全分辨率layer3让p90从184ms
    涨到242ms,爆了200ms预算,所以必须同时测便宜版能不能既保住精度又付得起。"""
    if det._bb_l3 is None:
        det._bb_l3 = Backbone(layers=(3,), device=det._bb_loc.device)
    size = l3_in or det._seg_in

    def extractor(img):
        f12 = det._wrn_feats(img)
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(det._bb_loc.device)
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        f3 = det._bb_l3.extract(x)
        f3 = torch.nn.functional.interpolate(f3, size=f12.shape[-2:], mode="bilinear", align_corners=False)[0]
        return torch.cat([f12, f3], dim=0)
    return extractor


def _train_head(extractor, feats, gts, pos_w, device=DEV):
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
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(SEED)
    n = len(all_feats)
    for _ in range(STEPS):
        sel = torch.randperm(n, generator=g)[:BATCH].tolist()
        X = torch.stack([all_feats[i] for i in sel]); X = (X - mu) / sd
        Y = torch.stack([all_gts[i] for i in sel])
        opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
    head.eval()
    return head, mu, sd


@torch.no_grad()
def _calibrate_thr(extractor, head, mu, sd, imgs, masks, out_hw=(256, 256)):
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


def run_one(name, normals, fit_i, fit_m, test_defs):
    t0 = time.time()
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"  [{name}] fit_fewshot完成,耗时{time.time()-t0:.0f}s "
          f"lat_trimmed={getattr(det, 'lat_trimmed', None)}", flush=True)
    if det.seg_head.head is None:
        print(f"{name}: 生产seg_head未训成功,跳过", flush=True)
        return None
    # DINO门可能被_calibrate_latency在GPU负载/热降频下砍掉(已知风险)。不补的话
    # cable这类完全依赖DINO门的类目baseline会崩(实测0.811→0.171,正好对上记录里
    # "弃DINO门代价-0.6+量级"),测出来的Δ就不代表生产状态了。
    if det._dino is None:
        print(f"  [{name}] DINO门被延时自适应砍了,强制重新标定(本实验只关心定位精度)", flush=True)
        det._calibrate_dino_gate(normals, fit_i)

    # baseline:生产原样(浅层1,2)
    t1 = time.time()
    base_iou, base_hit, base_lat = _eval_with_head(det, det.seg_head.extractor,
                                                    det.seg_head.head, det.seg_head.mu, det.seg_head.sd,
                                                    det.seg_head.thr, test_defs)
    print(f"  [{name}] baseline评测完成,耗时{time.time()-t1:.0f}s,IoU={base_iou:.3f}", flush=True)

    # 深层版:重训一个新头(通道数变了,不能复用旧头权重)。两个变体一次测完:
    # full=layer3走全分辨率(512,精度上限但已知爆预算);half=layer3走256(成本~1/4)
    out = {"base": (base_iou, base_hit, base_lat)}
    for tag, l3_in in [("full", None), ("half", 256)]:
        t2 = time.time()
        deep_ext = make_deep_extractor(det, l3_in=l3_in)
        with torch.no_grad():
            feats = [deep_ext(im) for im in fit_i]
        grid_hw = feats[0].shape[-2:]
        gts = [torch.from_numpy(_mask_to(m, grid_hw[0], grid_hw[1]).astype(np.float32)) for m in fit_m]
        pos_total = sum(float(g.sum()) for g in gts)
        neg_total = len(gts) * grid_hw[0] * grid_hw[1] - pos_total
        pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=DEV)
        head_d, mu_d, sd_d = _train_head(deep_ext, feats, gts, pos_w)
        thr_d = _calibrate_thr(deep_ext, head_d, mu_d, sd_d, fit_i, fit_m)
        if thr_d is None:
            print(f"{name}({tag}): 阈值标定失败,跳过", flush=True)
            continue
        d_iou, d_hit, d_lat = _eval_with_head(det, deep_ext, head_d, mu_d, sd_d, thr_d, test_defs)
        out[tag] = (d_iou, d_hit, d_lat)
        print(f"  [{name}] layer3-{tag}完成,耗时{time.time()-t2:.0f}s "
              f"IoU={d_iou:.3f} hit={d_hit:.3f} 延时p90={d_lat:.0f}ms Δ={d_iou-base_iou:+.3f}", flush=True)

    print(f"{name:20s} baseline IoU={base_iou:.3f}/p90={base_lat:.0f}ms | "
          + " | ".join(f"L3-{t} IoU={out[t][0]:.3f}/p90={out[t][2]:.0f}ms/Δ={out[t][0]-base_iou:+.3f}"
                       for t in ("full", "half") if t in out), flush=True)
    return out


def _eval_with_head(det, extractor, head, mu, sd, thr, test_defs):
    orig = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr,
            det.seg_head.extractor, det.seg_head.rams)
    det.seg_head.head, det.seg_head.mu, det.seg_head.sd = head, mu, sd
    det.seg_head.thr = thr; det.seg_head.extractor = extractor; det.seg_head.rams = None
    ious, hits, lat = [], [], []
    for img, gt in test_defs:
        t0 = time.time()
        o = det.locate(img)
        lat.append((time.time() - t0) * 1000)
        if o.get("mask") is None:
            ious.append(0.0); hits.append(0.0); continue
        mask = o["mask"]
        gt_r = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        ious.append(_per_image_iou(mask, gt_r))
        hits.append(box_hit(o["boxes"], gt_boxes(gt)) or 0.0)
    (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr,
     det.seg_head.extractor, det.seg_head.rams) = orig
    return float(np.mean(ious)), float(np.mean(hits)), float(np.percentile(lat, 90))


def main():
    torch.manual_seed(0)
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("生产:pcb(微小缺陷)", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
    ]
    names, rows = [], []
    for name, prep in jobs:
        row = run_one(name, *prep())
        if row is None:
            continue
        names.append(name); rows.append(row)
    for tag in ("full", "half"):
        deltas = [r[tag][0] - r["base"][0] for r in rows if tag in r]
        lats = [r[tag][2] for r in rows if tag in r]
        if not deltas:
            continue
        d = np.array(deltas)
        passed = (np.median(d) >= 0.005 and np.mean(d) > 0
                 and (d > 0).sum() >= max(1, len(d) // 2 + 1) and np.min(d) >= -0.01)
        print(f"\n=== layer3-{tag} 汇总(n={len(d)}) === median(Δ)={np.median(d):+.3f} "
              f"mean(Δ)={np.mean(d):+.3f} min(Δ)={np.min(d):+.3f} 延时p90均值={np.mean(lats):.0f}ms "
              f"{'精度通过' if passed else '精度不通过'}", flush=True)
        print(dict(zip(names, [round(float(x), 3) for x in d])), flush=True)


if __name__ == "__main__":
    main()

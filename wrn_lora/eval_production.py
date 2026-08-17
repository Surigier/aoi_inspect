"""WRN-LoRA的**生产管线**验证——此前所有LoRA数字(判负的和激进配置判正的)都是在
wrn_lora/diagnose.py那套独立测试台上测的"裸分割头IoU",从没走过生产det.locate()
完整链路(没有图级is_defect门、没有SAM精化、没有框合并)。今天GCAD-EmbedAE的教训
就是隔离测试好看、进生产崩掉,所以LoRA必须补这一课才能谈转正。

和layer3加深语义那条路的关键差别:LoRA训完可以merged_conv()合并回普通卷积,
**推理零增量延时**——layer3那条路正是死在+70ms上(生产locate已经149~207ms,
再加70ms必然破200ms硬预算)。LoRA没这个问题。

结构上不存在GCAD那种正常图误报风险:LoRA改的是det._bb_loc(定位骨干),只被
seg_head用;而seg_head只在locate()判定is_defect=True之后才被调用,不参与图级
判定本身。所以图级acc不会因为LoRA变化(会在结果里实测确认这一点)。

用法:PYTHONPATH=. python wrn_lora/eval_production.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
from aoi.competition import CompetitionLargeDetector
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as _ProdHead, _Ensemble, _mask_to
from wrn_lora.lora_backbone import LoRAConv2d
from global_context.eval_global_branch import prep_loco, prep_mvtec, prep_realiad, gt_boxes, box_hit
from scripts.run_scorecard_5types import prep_mvtec_color

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HEAD_STEPS, HEAD_LR, LORA_LR, BATCH, SEED = 300, 5e-3, 1e-3, 8, 0
N_LATE_BLOCKS, RANK = 2, 4          # 激进配置实测最好的那组(见CLAUDE.md重要修正条目)


def apply_lora(det, n_late_blocks=N_LATE_BLOCKS, r=RANK):
    """在生产骨干det._bb_loc.model上做LoRA手术,返回(lora_modules, 还原用的原始conv)。
    注意:_wrn_feats和_wrn_feats_diff内部都调用同一个_bb_loc,所以两种特征模式会
    自动一起继承LoRA,不需要分别处理。"""
    model = det._bb_loc.model
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()                                    # BN全程冻结(小批量会破坏预训练统计量)
    # timm features_only包装后,真实层在model里的名字可能带前缀,先定位layer2
    layer2 = None
    for attr in ("layer2",):
        if hasattr(model, attr):
            layer2 = getattr(model, attr)
            break
    if layer2 is None:
        raise RuntimeError("找不到layer2,无法做LoRA手术")
    mods, originals = [], []
    for bi in ([3] if n_late_blocks == 1 else [2, 3]):
        block = layer2[bi]
        originals.append((block, block.conv2))
        wrapped = LoRAConv2d(block.conv2, r=r).to(det._bb_loc.device)
        block.conv2 = wrapped
        mods.append(wrapped)
    return mods, originals


def restore(originals):
    for block, conv in originals:
        block.conv2 = conv


def merge_lora(originals, mods):
    """把LoRA旁路合并回普通卷积——推理零增量延时的关键。"""
    for (block, _), lm in zip(originals, mods):
        block.conv2 = lm.merged_conv()


def grad_extractor(det):
    """LoRA训练专用的特征前向:和det._wrn_feats / _wrn_feats_diff等价,但**不带
    @torch.no_grad()**——生产的_wrn_feats是no_grad的(推理路径本来就不需要梯度),
    直接拿它训LoRA会导致LoRA参数收不到任何梯度、up权重永远停在零初始化(=恒等
    映射),整个训练静默空转:Adam对grad=None的参数直接跳过,不报错、不告警,
    跑出来的"提升"其实全是重训分割头带来的,和LoRA无关。这个坑2026-08-14踩过一次。"""
    def _feats(img):
        x = img.unsqueeze(0) if img.dim() == 3 else img
        x = x.to(det._bb_loc.device)
        x = torch.nn.functional.interpolate(x, size=(det._seg_in, det._seg_in),
                                             mode="bilinear", align_corners=False)
        feats = det._bb_loc.model(x)
        size = feats[0].shape[-2:]
        feats = [torch.nn.functional.interpolate(f, size=size, mode="bilinear", align_corners=False)
                 for f in feats]
        return torch.cat(feats, dim=1)[0]

    if getattr(det, "feat_mode", "single") != "tmpl_diff":
        return _feats

    def _feats_diff(img):                      # 和_wrn_feats_diff同款:concat[f, f-f_ref]
        f = _feats(img)
        ref = det._ref_bank.aligned_ref(img if img.dim() == 3 else img[0])
        with torch.no_grad():                  # 参考图分支不需要梯度
            fr = _feats(ref)
        return torch.cat([f, f - fr], dim=0)
    return _feats_diff


def train_lora_and_head(det, fit_i, fit_m, lora_params, device=DEV):
    """LoRA参数和分割头联合训练(同生产_train_one的loss/优化器/batch/种子/归一化,
    LoRA用单独的lr)。注意extractor必须实时算(LoRA在变),不能预先缓存特征。

    ⚠️归一化必须和生产_train_one完全一致:**固定的per-channel mu/sd**(从全部fit
    特征算一次),不能用每个batch实时算的统计量——训练时归一化标准飘、评测时又换成
    固定的,会系统性地坑训练效果,得出的负结果是实现bug不是机制本身的问题。
    LoRA起点是恒等(up零初始化),所以训练前的特征≡baseline特征,用它算的mu/sd
    对训练全程都是合理估计(LoRA权重相对变化量级只有1e-2)。"""
    extractor = grad_extractor(det)            # 必须用带梯度的版本,见grad_extractor docstring
    with torch.no_grad():
        init_feats = [extractor(im) for im in fit_i]
    C, gh, gw = init_feats[0].shape
    # 和生产_train_one同款:per-channel统计,全fit算一次,训练+评测全程固定
    mu = torch.stack([f.float().mean(dim=(1, 2)) for f in init_feats]).mean(0).view(1, -1, 1, 1).to(device)
    sd = (torch.stack([f.float().std(dim=(1, 2)) for f in init_feats]).mean(0) + 1e-6).view(1, -1, 1, 1).to(device)
    del init_feats

    gts = [torch.from_numpy(_mask_to(m, gh, gw).astype(np.float32)).to(device) for m in fit_m]
    pos_total = sum(float(g.sum()) for g in gts)
    neg_total = len(gts) * gh * gw - pos_total
    pos_w = torch.tensor([neg_total / max(pos_total, 1)], device=device)

    torch.manual_seed(SEED)
    lin = _ProdHead._linear_head(C).to(device)
    cnv = _ProdHead._conv_head(C).to(device)
    head = _Ensemble(lin, cnv)
    opt = torch.optim.Adam([{"params": head.parameters(), "lr": HEAD_LR},
                             {"params": lora_params, "lr": LORA_LR}], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    g = torch.Generator().manual_seed(SEED)
    n = len(fit_i)
    up0 = [p.detach().clone() for p in lora_params]
    head.train()
    for _ in range(HEAD_STEPS):
        sel = torch.randperm(n, generator=g)[:BATCH].tolist()
        feats = torch.stack([extractor(fit_i[i]) for i in sel])      # LoRA在训,必须实时前向
        X = (feats - mu) / sd
        Y = torch.stack([gts[i] for i in sel])
        opt.zero_grad(); lossf(head(X).squeeze(1), Y).backward(); opt.step()
    head.eval()
    # 硬断言:LoRA参数必须真的动了。不加这条的话,梯度断掉时Adam对grad=None静默跳过,
    # 整个训练空转但不报错,跑出来的"提升"其实全是重训分割头的效果(2026-08-14踩过)。
    moved = max(float((p.detach() - q).abs().max()) for p, q in zip(lora_params, up0))
    if moved < 1e-9:
        raise RuntimeError("LoRA参数训练后没有任何变化——梯度可能被no_grad截断,本次结果无效")
    print(f"    (LoRA权重最大变化量={moved:.3e},确认真的训起来了)", flush=True)
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
    """走生产det.locate()完整链路:含漏检IoU + 框命中 + 图级acc(正常图误报率一起测,
    今天GCAD的教训:只测缺陷图召回会漏掉致命的假阳性问题)+ 延时p90。"""
    ious, hits, lat = [], [], []
    tp_def = 0
    for img, gt in test_defs:
        t0 = time.time()
        o = det.locate(img)
        lat.append((time.time() - t0) * 1000)
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
    return (float(np.mean(ious)), float(np.mean(hits)), acc,
            float(np.percentile(lat, 90)))


def run_one(name, normals, fit_i, fit_m, test_defs, goods):
    t0 = time.time()
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"  [{name}] fit完成 {time.time()-t0:.0f}s lat_trimmed={getattr(det,'lat_trimmed',None)}", flush=True)
    if det.seg_head.head is None:
        print(f"{name}: seg_head未训成功,跳过", flush=True); return None
    if det._dino is None:
        det._calibrate_dino_gate(normals, fit_i)

    b_iou, b_hit, b_acc, b_lat = eval_pipeline(det, test_defs, goods)
    print(f"  [{name}] baseline IoU={b_iou:.3f} hit={b_hit:.3f} acc={b_acc:.3f} p90={b_lat:.0f}ms", flush=True)

    mods, originals = apply_lora(det)
    lora_params = []
    for lm in mods:
        lora_params += list(lm.down.parameters()) + list(lm.up.parameters())
    orig_head = (det.seg_head.head, det.seg_head.mu, det.seg_head.sd, det.seg_head.thr, det.seg_head.rams)
    try:
        head, mu, sd = train_lora_and_head(det, fit_i, fit_m, lora_params)
        merge_lora(originals, mods)                       # 合并回普通卷积(推理零增量)
        thr = calibrate_thr(det, head, mu, sd, fit_i, fit_m)
        if thr is None:
            print(f"{name}: LoRA版阈值标定失败,跳过", flush=True)
            return None
        det.seg_head.head, det.seg_head.mu, det.seg_head.sd = head, mu, sd
        det.seg_head.thr = thr; det.seg_head.rams = None
        l_iou, l_hit, l_acc, l_lat = eval_pipeline(det, test_defs, goods)
    finally:
        restore(originals)
        (det.seg_head.head, det.seg_head.mu, det.seg_head.sd,
         det.seg_head.thr, det.seg_head.rams) = orig_head

    print(f"{name:20s} base IoU={b_iou:.3f}/hit={b_hit:.3f}/acc={b_acc:.3f}/p90={b_lat:.0f}ms | "
          f"LoRA IoU={l_iou:.3f}/hit={l_hit:.3f}/acc={l_acc:.3f}/p90={l_lat:.0f}ms | "
          f"ΔIoU={l_iou-b_iou:+.3f} Δacc={l_acc-b_acc:+.3f}", flush=True)
    return dict(base=(b_iou, b_hit, b_acc), lora=(l_iou, l_hit, l_acc))


def main():
    torch.manual_seed(0)
    jobs = [
        ("外观 hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
        ("缺件 cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("色彩 pill", lambda: prep_mvtec("pill", ["color"])),
        ("生产:pcb", lambda: prep_realiad("pcb")),
        ("生产:phone_battery", lambda: prep_realiad("phone_battery")),
    ]
    names, d_iou, d_acc = [], [], []
    for name, prep in jobs:
        normals, fit_i, fit_m, test_defs = prep()
        goods = normals[-20:]                # 留20张正常图测假阳性(不参与fit的后20张)
        row = run_one(name, normals[:-20], fit_i, fit_m, test_defs, goods)
        if row is None:
            continue
        names.append(name)
        d_iou.append(row["lora"][0] - row["base"][0])
        d_acc.append(row["lora"][2] - row["base"][2])
    if d_iou:
        di, da = np.array(d_iou), np.array(d_acc)
        passed = (np.median(di) >= 0.005 and np.mean(di) > 0
                 and (di > 0).sum() >= max(1, len(di) // 2 + 1) and np.min(di) >= -0.01
                 and np.min(da) >= -0.01)
        print(f"\n=== 汇总(n={len(di)}) === ΔIoU median={np.median(di):+.3f} mean={np.mean(di):+.3f} "
              f"min={np.min(di):+.3f} | Δacc min={np.min(da):+.3f} | {'通过' if passed else '不通过'}", flush=True)
        print("ΔIoU:", dict(zip(names, [round(float(x), 3) for x in di])), flush=True)
        print("Δacc:", dict(zip(names, [round(float(x), 3) for x in da])), flush=True)


if __name__ == "__main__":
    main()

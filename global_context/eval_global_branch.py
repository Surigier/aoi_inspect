"""GCAD风格全局上下文分支——充分对比,不是跑一两次就下结论。

对比对象(同一fit/同一test,唯一变量是图级门融合项):
  baseline    : max(z_EAD, z_DINO)                          (现生产)
  +PixelAE    : max(z_EAD, z_DINO, z_pixelAE)                (整图像素级瓶颈重建,忠实GCAD原论文)
  +EmbedAE    : max(z_EAD, z_DINO, z_embedAE)                (DINOv2 CLS token瓶颈重建,测语义是否有额外增益)

覆盖面(避免"看几次实验结果就下结论"):
  5类LOCO logical_anomalies(缺件/错位/组合,这条支路的目标场景)
  2类LOCO structural_anomalies(breakfast_box/juice_bottle,回归检查——纹理类缺陷
    现有EAD/WRN/DINO已经够用,新支路不该在这类上添乱)
  2类现有生产类目(cable/pcb,更广的回归安全检查,不限于LOCO代理数据)
9类目×3配置=27次fit_fewshot+评测,margin配对判定(同wrn_lora口径):
  median(Δ)>=0.005 且 mean(Δ)>0 且 >=多数类中位数为正 且 min(Δ)>=-0.01 才算通过。

【已验证,判负封存】真实9类目结果(ΔIoU,见global_context/eval_global_branch_resume.py
补跑的剩余5类合并):PixelAE median=+0.002 min=-0.021(juice_bottle逻辑异常);EmbedAE
median=0.000 min=-0.014(screw_bag逻辑异常)。两者都远低于0.005门槛且跌破-0.01最差
类别底线,即使只看5类目标场景(logical_anomalies)中位数也只有+0.003,同样跌破底线。
5/9(PixelAE)、4/9(EmbedAE)类目有正向移动(breakfast_box逻辑异常+juice_bottle结构
异常两个变体都表现不错),但不足以支撑广泛稳定收益——和WRN-LoRA、Top-1 ROI同一个
故事:局部有信号,整体不过关。**顺带回答了驱动这次实验的问题("语义容量是不是
瓶颈"):EmbedAE(DINOv2语义嵌入)中位数(0.000)并不明显优于PixelAE(纯像素,
+0.002)——说明"加更强语义表示"本身不是这里的瓶颈,GCAD式"整图瓶颈重建判构图"这个
架构思路本身收益有限,不是换更大模型就能解决的。** 默认不接入competition.py,
代码留opt-in研究件。过程中还意外发现并修复了一个真实confound:GPU连轴运行导致
`_calibrate_latency`的硬线延时自适应会把已标定好的DINO门砍掉(lat_trimmed里带
"dino_gate"),9类里有5类命中这个情况——已在run_one()里加了强制重新标定,不影响
本实验结论,但提醒:评委机器如果也处于高负载/高温状态,生产DINO门也可能被这条
路径误伤,值得后续单独复查`_calibrate_latency`的硬线阈值是否设得过紧。

用法:PYTHONPATH=. python global_context/eval_global_branch.py(封存实验,不再新增大搜索)
"""
import glob
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aoi.competition import CompetitionLargeDetector
from aoi.fewshot import FewShotAdapter
from aoi.seg_head import map_to_boxes, merge_boxes
from aoi.imageio import load_fast
from global_context.autoencoder import PixelAE, EmbedAE, fit_ae, calibrate_zscore
from global_context.dino_cls import DinoCLS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOCO_ROOT = Path("data/_dl/mvtec_loco")
RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)


# ---------- 数据准备:LOCO + MVTec(cable) + Real-IAD(pcb) 三种口径统一到 (normals, fit_i, fit_m, test_defs) ----------

def _union_mask(gt_dir, hw=HW):
    m = None
    for mp in sorted(gt_dir.glob("*.png")):
        arr = np.array(Image.open(mp).convert("L").resize((hw[1], hw[0]))) > 0
        m = arr.astype(np.uint8) if m is None else (m | arr.astype(np.uint8))
    return m if m is not None else np.zeros(hw, np.uint8)


def prep_loco(cat, anomaly_type, n_norm=100, n_fit=15, seed=0):
    root = LOCO_ROOT / cat
    normals = [load_fast(p) for p in sorted((root / "train" / "good").glob("*.png"))[:n_norm]]
    imgs = sorted((root / "test" / anomaly_type).glob("*.png"))
    random.Random(seed).shuffle(imgs)
    fit_p, test_p = imgs[:n_fit], imgs[n_fit:]
    fit_i = [load_fast(p) for p in fit_p]
    fit_m = [_union_mask(root / "ground_truth" / anomaly_type / p.stem) for p in fit_p]
    test_defs = [(load_fast(p), _union_mask(root / "ground_truth" / anomaly_type / p.stem))
                for p in test_p]
    return normals, fit_i, fit_m, test_defs


def _read_mvtec(p, hw=HW):
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def prep_mvtec(cat, folders, n_norm=100, seed=0):
    root = Path(f"data/mvtec/{cat}")
    gt = Path("data/_dl/_gt_stage/mvtech_anomaly_detection")
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:n_norm]]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(seed).shuffle(df); k = max(5, len(df) // 3)
    fit_i = [load_fast(p) for p, _ in df[:k]]
    fit_m = [_read_mvtec(gt / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png")) for p, fo in df[:k]]
    test_defs = [(load_fast(p), _read_mvtec(gt / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png")))
                for p, fo in df[k:]]
    return normals, fit_i, fit_m, test_defs


def prep_realiad(cat, n_norm=100, n_fit=30, seed=0):
    import json
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(seed).shuffle(tok)
    normals = [load_fast(R / x["image_path"]) for x in tok[:n_norm]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(seed).shuffle(ng)
    fit_i = [load_fast(R / x["image_path"]) for x in ng[:n_fit]]
    fit_m = [_read_mvtec(R / x["mask_path"]) for x in ng[:n_fit]]
    test_defs = [(load_fast(R / x["image_path"]), _read_mvtec(R / x["mask_path"]))
                for x in ng[n_fit:n_fit + 40]]
    return normals, fit_i, fit_m, test_defs


# ---------- 评测:自定义图级融合决定is_defect,下游定位管线原样复用 ----------

def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def gt_boxes(mask):
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in
            (stats[i] for i in range(1, n)) if a >= 4]


def box_hit(pred_boxes, gtbs, thr=0.5):
    if not gtbs:
        return None
    hit = sum(1 for g in gtbs if any(_box_iou(p[:4], g) >= thr for p in pred_boxes))
    return hit / len(gtbs)


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def fit_global_branches(det, normals, defects, dino, ae_steps=300, ae_lr=1e-3):
    """在同一批fit数据上标定两个全局分支的z-score统计量+融合阈值(与_calibrate_dino_gate
    完全同款流程:全fit标定mu/sd,阈值用FewShotAdapter._calibrate在z-fused值上找)。
    ae_steps/ae_lr默认300/1e-3(原判负用的配置),传入更激进的值可复查是不是训练
    强度不够(和WRN-LoRA/seg_head同一个"配置太保守"假说)。"""
    pix_ae = fit_ae(PixelAE().to(DEV), normals + defects[:len(normals) // 2 or 1],
                     steps=ae_steps, lr=ae_lr, device=DEV)
    cls_normals = [dino.cls(n) for n in normals]
    cls_defects = [dino.cls(d) for d in defects]
    emb_ae = fit_ae(EmbedAE().to(DEV), cls_normals, steps=ae_steps, lr=ae_lr, device=DEV)

    pix_mu, pix_sd = calibrate_zscore(pix_ae, normals)
    emb_mu, emb_sd = calibrate_zscore(emb_ae, cls_normals, is_embed=True)

    ead = det.branches[0]
    en = np.array([ead.score(n) for n in normals])
    ed = np.array([ead.score(d) for d in defects])
    dn = np.array([det._dino.score(n) for n in normals])
    dd = np.array([det._dino.score(d) for d in defects])
    emu, esd, dmu, dsd = det._dino_stats

    pn = np.array([pix_ae.score(n[None].to(DEV)) for n in normals])
    pd = np.array([pix_ae.score(d[None].to(DEV)) for d in defects])
    gn = np.array([emb_ae.score(c[None].to(DEV)) for c in cls_normals])
    gd = np.array([emb_ae.score(c[None].to(DEV)) for c in cls_defects])

    def z(v, mu, sd):
        return (v - mu) / (sd + 1e-9)

    fused_base_n = np.maximum(z(en, emu, esd), z(dn, dmu, dsd))
    fused_base_d = np.maximum(z(ed, emu, esd), z(dd, dmu, dsd))
    fused_pix_n = np.maximum(fused_base_n, z(pn, pix_mu, pix_sd))
    fused_pix_d = np.maximum(fused_base_d, z(pd, pix_mu, pix_sd))
    fused_emb_n = np.maximum(fused_base_n, z(gn, emb_mu, emb_sd))
    fused_emb_d = np.maximum(fused_base_d, z(gd, emb_mu, emb_sd))

    thr_base = FewShotAdapter._calibrate(list(fused_base_n), list(fused_base_d))
    thr_pix = FewShotAdapter._calibrate(list(fused_pix_n), list(fused_pix_d))
    thr_emb = FewShotAdapter._calibrate(list(fused_emb_n), list(fused_emb_d))
    # EmbedAE独立标定自己的阈值(不参与base的联合重标定),供"emb_or"变体用——
    # diag_interaction_pcb.py诊断出:三信号取max()后联合重标定一个阈值,会把base
    # (EAD+DINO)原有的判定边界跟着抬高,导致原本能正确抓到的图反而漏检(pcb上1张
    # 净损失换4张"抓到了但掩膜本身没用"的空欢喜,net亏)。emb_or用OR逻辑代替
    # max()+联合重标定:base该抓的一定还抓得到(阈值不变),EmbedAE只能新增覆盖,
    # 不能收窄base已有的判定边界。
    thr_emb_only = FewShotAdapter._calibrate(list(z(gn, emb_mu, emb_sd)), list(z(gd, emb_mu, emb_sd)))

    def _base_fn(img):
        return max(z(ead.score(img), emu, esd), z(det._dino.score(img), dmu, dsd))

    def _emb_or_fn(img):
        z_emb_img = z(emb_ae.score(dino.cls(img)[None].to(DEV)), emb_mu, emb_sd)
        return max(_base_fn(img) - thr_base, z_emb_img - thr_emb_only)

    def _emb_only_fn(img):
        return z(emb_ae.score(dino.cls(img)[None].to(DEV)), emb_mu, emb_sd)

    return dict(
        base=(_base_fn, thr_base),
        emb_only=(_emb_only_fn, thr_emb_only),
        pix=(lambda img: max(z(ead.score(img), emu, esd), z(det._dino.score(img), dmu, dsd),
                             z(pix_ae.score(img[None].to(DEV)), pix_mu, pix_sd)), thr_pix),
        emb=(lambda img: max(z(ead.score(img), emu, esd), z(det._dino.score(img), dmu, dsd),
                             z(emb_ae.score(dino.cls(img)[None].to(DEV)), emb_mu, emb_sd)), thr_emb),
        emb_or=(_emb_or_fn, 0.0),
    )


def evaluate_variant(det, fused_fn, thr, test_defs):
    ious, hits = [], []
    for img, gt in test_defs:
        s = fused_fn(img)
        is_def = s >= thr
        if is_def:
            amap = det.segment(img)
            th = det.pix_thr if det.pix_thr is not None else float(amap.mean() + 3 * amap.std())
            mask = (amap >= th).astype(np.uint8)
            gt_r = (torch.nn.functional.interpolate(
                torch.from_numpy(gt.astype(np.float32))[None, None],
                size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
            ious.append(_per_image_iou(mask, gt_r))
            boxes = merge_boxes(map_to_boxes(mask.astype(np.float32), 0.5, min_area_frac=0.0002, close=0),
                                getattr(det, "box_merge_d", 0))
            hits.append(box_hit(boxes, gt_boxes(gt)) or 0.0)
        else:
            ious.append(0.0); hits.append(0.0)
    return float(np.mean(ious)), float(np.mean(hits))


def run_one(name, normals, fit_i, fit_m, test_defs, ae_steps=300, ae_lr=1e-3):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    if det._dino is None:
        # _calibrate_latency可能在硬线超时时把已标定好的DINO门砍掉(见competition.py
        # "④超真硬线才弃DINO门"那段)——这是延时预算自适应的正常行为,但和本实验要测的
        # "融合机制本身准不准"无关(本实验不测真实延时达标性),这里强制重新标定一次,
        # 避免GPU忙/热导致的延时探针抖动把样本量吃掉一半以上。
        print(f"{name}: DINO门被延时自适应裁剪掉了(lat_trimmed={getattr(det,'lat_trimmed',None)}),"
              f"本实验只关心准确率不关心延时预算,强制重新标定", flush=True)
        det._calibrate_dino_gate(normals, fit_i)
    if det._dino is None:
        print(f"{name}: 强制重标后DINO门仍未启用(样本量真的不够),跳过", flush=True)
        return None
    dino = DinoCLS(device=DEV)
    fns = fit_global_branches(det, normals, fit_i, dino, ae_steps=ae_steps, ae_lr=ae_lr)
    row = {}
    for tag in ["base", "pix", "emb"]:
        fn, thr = fns[tag]
        iou, hit = evaluate_variant(det, fn, thr, test_defs)
        row[tag] = (iou, hit)
    print(f"{name:34s} baseline IoU={row['base'][0]:.3f}/hit={row['base'][1]:.3f}  "
          f"+PixelAE IoU={row['pix'][0]:.3f}/hit={row['pix'][1]:.3f}  "
          f"+EmbedAE IoU={row['emb'][0]:.3f}/hit={row['emb'][1]:.3f}", flush=True)
    return row


def main():
    torch.manual_seed(0)
    jobs = [
        ("logical:breakfast_box", lambda: prep_loco("breakfast_box", "logical_anomalies")),
        ("logical:juice_bottle", lambda: prep_loco("juice_bottle", "logical_anomalies")),
        ("logical:pushpins", lambda: prep_loco("pushpins", "logical_anomalies")),
        ("logical:screw_bag", lambda: prep_loco("screw_bag", "logical_anomalies")),
        ("logical:splicing_connectors", lambda: prep_loco("splicing_connectors", "logical_anomalies")),
        ("structural:breakfast_box(回归检查)", lambda: prep_loco("breakfast_box", "structural_anomalies")),
        ("structural:juice_bottle(回归检查)", lambda: prep_loco("juice_bottle", "structural_anomalies")),
        ("生产:cable(回归检查)", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
        ("生产:pcb(回归检查)", lambda: prep_realiad("pcb")),
    ]
    results = {"pix": [], "emb": []}
    names = []
    for name, prep in jobs:
        row = run_one(name, *prep())
        if row is None:
            continue
        names.append(name)
        for tag in ["pix", "emb"]:
            results[tag].append(row[tag][0] - row["base"][0])   # ΔIoU vs baseline(配对,同fit)

    print("\n=== 汇总(margin配对判定,同wrn_lora口径)===", flush=True)
    ns = len(names)
    for tag, label in [("pix", "+PixelAE(整图像素瓶颈)"), ("emb", "+EmbedAE(DINO CLS瓶颈)")]:
        deltas = np.array(results[tag])
        if len(deltas) == 0:
            print(f"{label}: 无有效样本"); continue
        passed = (np.median(deltas) >= 0.005 and np.mean(deltas) > 0
                 and (deltas > 0).sum() >= max(1, ns // 2 + 1) and np.min(deltas) >= -0.01)
        print(f"{label}: median(Δ)={np.median(deltas):+.3f} mean(Δ)={np.mean(deltas):+.3f} "
              f"正类目数={int((deltas>0).sum())}/{ns} min(Δ)={np.min(deltas):+.3f}  "
              f"{'✅通过' if passed else '❌不通过'}", flush=True)
        print(f"  逐类目Δ: {dict(zip(names, [round(float(d),3) for d in deltas]))}", flush=True)


if __name__ == "__main__":
    main()

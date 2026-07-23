"""YOLO候选框提议器可行性探针(先别急着训练):差异通道(Lab色差+梯度差+局部SSIM)
在Real-IAD电子/手机件真实数据上到底有没有判别信号?
①最近邻正常模板配对+ECC对齐+算差异通道(aoi/diff_channels.py)
②量两个诚实指标(不用AUROC,用项目一贯的IoU口径):
   - mask区域 vs 背景 差异强度均值比值(信号强度粗判)
   - 差异通道直接阈值化当预测的"纯定位IoU上限"(和crop_cascade当初-0.059的候选生成器
     同一评估口径,直接可比——crop_cascade当时只在ViSA pcb1单类测,这次是真实电子/
     手机件多类,数据量和域都更贴近赛题)
这个IoU上限如果比crop_cascade当初的±0还差或差不多,说明差异通道本身信息量不够,
YOLO训练也救不回来(garbage in garbage out);如果明显更好,才值得投入YOLO训练管线。
用法:PYTHONPATH=. python scripts/diag_diff_signal.py
"""
import glob
import json
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from aoi.diff_channels import build_6ch
from aoi.imageio import load_fast

RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)
CATS = ["phone_battery", "pcb", "sim_card_set", "usb", "switch"]


def _read_mask(p, hw):
    if p is None or not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def _gray_key(img, size=32):
    import cv2
    arr = (img.mean(0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(arr, (size, size)).astype(np.float32)


def per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def _boxes_from_mask(mask, min_area=4):
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in (stats[i] for i in range(1, n)) if a >= min_area]


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def box_recall_at(pred_mask, gt_mask, iou_thr=0.3):
    """候选框召回:naive阈值给出的连通域框里,有没有任意一个和GT框IoU>=thr(宽松,匹配
    "候选框提议"这个用途——只要求"框对大概位置",不要求像素级精确,精确交给下游YOLO+精修)。"""
    gtb = _boxes_from_mask(gt_mask)
    if not gtb:
        return None
    predb = _boxes_from_mask(pred_mask)
    if not predb:
        return 0.0
    hit = sum(1 for g in gtb if any(_box_iou(p, g) >= iou_thr for p in predb))
    return hit / len(gtb)


def main():
    torch.manual_seed(0)
    all_ratios, all_best_ious, all_recalls = [], [], []
    for cat in CATS:
        d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
        tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]
        random.Random(0).shuffle(tok)
        templates = [load_fast(R / x["image_path"]) for x in tok[:30]]
        tmpl_keys = np.stack([_gray_key(t) for t in templates])

        ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]
        random.Random(1).shuffle(ng)
        test_items = ng[:25]

        ratios, best_ious, recalls_fixed = [], [], []
        for x in test_items:
            img = load_fast(R / x["image_path"])
            gt = _read_mask(R / x["mask_path"], HW)
            if gt.sum() < 4:
                continue
            qk = _gray_key(img)
            i = int(np.argmin(((tmpl_keys - qk) ** 2).mean(axis=(1, 2))))
            ch6 = build_6ch(img, templates[i])                 # (6,H,W)
            diff = ch6[3:]                                     # (3,H,W): lab/grad/ssim
            combined = diff.max(axis=0)                        # (H,W) 融合差异强度
            import cv2
            combined_256 = cv2.resize(combined, (HW[1], HW[0]))
            m = gt.astype(bool)
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            ratio = float(combined_256[m].mean() / (combined_256[~m].mean() + 1e-6))
            ratios.append(ratio)
            best_iou = 0.0
            for pct in range(50, 100, 2):
                thr = np.percentile(combined_256, pct)
                pred = (combined_256 >= thr).astype(np.uint8)
                best_iou = max(best_iou, per_image_iou(pred, gt))
            best_ious.append(best_iou)
            # 固定分位阈值(不逐图挑最优,更诚实)+ 候选框召回(宽松IoU0.3,匹配"提议候选"用途)
            thr_fixed = np.percentile(combined_256, 92)
            pred_fixed = (combined_256 >= thr_fixed).astype(np.uint8)
            r = box_recall_at(pred_fixed, gt, iou_thr=0.3)
            if r is not None:
                recalls_fixed.append(r)
        print(f"{cat:14s} mask/背景差异强度比值中位={np.median(ratios):.2f}  "
              f"逐图最优阈值IoU均值={np.mean(best_ious):.3f}  "
              f"固定阈值候选框召回@IoU0.3={np.mean(recalls_fixed):.3f}  (n={len(ratios)})", flush=True)
        all_ratios += ratios; all_best_ious += best_ious; all_recalls += recalls_fixed
    print(f"\n=== 均值(5类合并) === mask/背景比值中位={np.median(all_ratios):.2f}  "
          f"逐图最优阈值IoU均值={np.mean(all_best_ious):.3f}  候选框召回@IoU0.3均值={np.mean(all_recalls):.3f}  "
          f"| 对照:crop_cascade当初ViSA pcb1实测-0.059(相对raw基线)", flush=True)


if __name__ == "__main__":
    main()

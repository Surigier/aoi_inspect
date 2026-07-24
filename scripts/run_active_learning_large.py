"""用户反馈驱动优化——真实验证ActiveLearningLoop接到生产大图架构CompetitionLargeDetector
后确实能工作(赛题"用户反馈驱动的优化"是三条重点解决问题之一,此前ActiveLearningLoop只
和demo_app.py的旧记忆库适配器接过,从未验证过对当前生产架构生效)。

场景:初始只给10张缺陷现场迁移(模拟"刚部署,标注还没跟上"),在真实留出测试集上评一次;
然后模拟操作员陆续把5张漏检/误检图反馈进去(loop.feedback(img, is_defect=True, mask=...)),
每次反馈都重跑fit_fewshot(带掩膜,监督分割头/SAM/crop_cascade/component_graph等全部真实
重新标定,不是玩具级"重建记忆库"),反馈完在**同一个**留出测试集上再评一次,对比前后。

用法:PYTHONPATH=. python scripts/run_active_learning_large.py --cat phone_battery
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from aoi.active_learning import ActiveLearningLoop
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast

RI = Path("data/_dl/Real-IAD"); RJ = Path("data/_dl/realiad_jsons/realiad_jsons_sv")
HW = (256, 256)


def _read(p, hw=HW):
    from PIL import Image
    if not Path(p).exists():
        return np.zeros(hw, np.uint8)
    return (np.array(Image.open(p).convert("L").resize((hw[1], hw[0]))) > 0).astype(np.uint8)


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def _gt_boxes(mask):
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(x, y, x + w, y + h) for x, y, w, h, a in
            (stats[i] for i in range(1, n)) if a >= 4]


def _box_hit(pred_boxes, gtbs, thr=0.5):
    if not gtbs:
        return None
    hit = sum(1 for g in gtbs if any(_box_iou(p[:4], g) >= thr for p in pred_boxes))
    return hit / len(gtbs)


def _per_image_iou(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    TP = int((p & g).sum()); FP = int((p & ~g).sum()); FN = int((~p & g).sum())
    return TP / max(TP + FP + FN, 1)


def prep(cat, n_norm=100, n_init=10, n_feedback=5, n_test=30):
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [load_fast(R / x["image_path"]) for x in tok[:n_norm]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    init_i = [load_fast(R / x["image_path"]) for x in ng[:n_init]]
    init_m = [_read(R / x["mask_path"]) for x in ng[:n_init]]
    fb_i = [load_fast(R / x["image_path"]) for x in ng[n_init:n_init + n_feedback]]
    fb_m = [_read(R / x["mask_path"]) for x in ng[n_init:n_init + n_feedback]]
    test_defs = [(load_fast(R / x["image_path"]), _read(R / x["mask_path"]))
                for x in ng[n_init + n_feedback:n_init + n_feedback + n_test]]
    return normals, init_i, init_m, fb_i, fb_m, test_defs


def evaluate_on(det, test_defs):
    ious, hits = [], []
    for img, gt in test_defs:
        o = det.locate(img)
        if o.get("mask") is None:
            ious.append(0.0); hits.append(0.0)
            continue
        mask = o["mask"]
        gt_r = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=mask.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        ious.append(_per_image_iou(mask, gt_r))
        hits.append(_box_hit(o["boxes"], _gt_boxes(gt)) or 0.0)
    return float(np.mean(ious)), float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="phone_battery")
    args = ap.parse_args()
    torch.manual_seed(0)

    normals, init_i, init_m, fb_i, fb_m, test_defs = prep(args.cat)
    print(f"{args.cat}: normals={len(normals)} 初始fit缺陷={len(init_i)} "
          f"待反馈缺陷={len(fb_i)} 留出test={len(test_defs)}", flush=True)

    det = CompetitionLargeDetector()
    loop = ActiveLearningLoop(det, normals, init_i, defect_masks=init_m)
    iou0, hit0 = evaluate_on(det, test_defs)
    print(f"反馈前(仅{len(init_i)}张缺陷现场迁移): 含漏检IoU={iou0:.3f} 框命中@0.5={hit0:.3f}", flush=True)

    for i, (img, mk) in enumerate(zip(fb_i, fb_m)):
        n_norm, n_def = loop.feedback(img, is_defect=True, mask=mk)
        print(f"  反馈第{i+1}张(操作员标记漏检)→ 重跑fit_fewshot,当前缺陷集={n_def}张", flush=True)

    iou1, hit1 = evaluate_on(det, test_defs)
    print(f"反馈后(共{len(init_i)+len(fb_i)}张缺陷): 含漏检IoU={iou1:.3f} 框命中@0.5={hit1:.3f}", flush=True)
    print(f"Δ含漏检IoU={iou1-iou0:+.3f}  Δ框命中@0.5={hit1-hit0:+.3f}", flush=True)
    print(f"结论:{'反馈机制在生产大图架构上验证有效(同一留出测试集上定位精度随反馈提升)' if iou1 > iou0 else '本次反馈样本未带来提升(需检查样本质量或增大反馈样本量)'}", flush=True)


if __name__ == "__main__":
    main()

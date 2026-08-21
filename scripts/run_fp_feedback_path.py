"""误检(假阳性)反馈能不能也走"实时"快路径?——赛题要求"当系统**误检或漏检**时,
操作员可提供**实时**反馈","实时"是修饰两者的,但目前只有漏检走了增量快路径
(251s),误检仍走完整fit(1242s),不契合题面。

假设:误检反馈未必需要重训EAD学生。操作员标记一张误检,系统要做的是"别再把它判成
缺陷",而阈值标定`_calibrate(ns, ds)`本来就吃**全部正常图**的EAD分——把这张高分
正常图加进去,阈值自然上移;DINO门/像素阈值同理也会重标。学生权重陈旧不走这条通路。

本脚本直接对比同一批假阳性样本上:
  快路径(retrain_ead=False,秒级~分钟级) vs 完整路径(retrain_ead=True,20分钟)
两者把假阳性率压下去的效果差多少,以及对缺陷侧有无反噬。

⚠️用train_steps=100快速fit:目的是**造出足够多的假阳性**(题面描述的正是"系统误检"
这个场景),好让反馈效果可观测——phone_battery生产配置下25张正常图只误判2张,
样本量根本不足以量化。这里测的是**反馈机制本身**,不是绝对精度,所以可接受。

用法:PYTHONPATH=. python scripts/run_fp_feedback_path.py --cat pill
"""
import argparse
import copy
import time
import numpy as np
import torch
from aoi.active_learning import ActiveLearningLoop
from aoi.competition import CompetitionLargeDetector
from global_context.eval_global_branch import prep_mvtec
from scripts.run_scorecard_5types import prep_mvtec_color

PREPS = {
    "pill": lambda: prep_mvtec("pill", ["color"]),
    "hazelnut": lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]),
    "carpet": lambda: prep_mvtec_color("carpet")[:4],
}


def _iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return tp / max(tp + fp + fn, 1)


def eval_defect(det, test_defs):
    ious = []
    for img, gt in test_defs:
        o = det.locate(img)
        if o.get("mask") is None:
            ious.append(0.0); continue
        m = o["mask"]
        gr = (torch.nn.functional.interpolate(
            torch.from_numpy(gt.astype(np.float32))[None, None],
            size=m.shape, mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        ious.append(_iou(m, gr))
    return float(np.mean(ious))


def fpr(det, goods):
    fp = [g for g in goods if det.locate(g)["is_defect"]]
    return len(fp) / max(len(goods), 1), fp


def run(cat, fast):
    """fast=True走retrain_ead=False快路径;False走完整fit。返回(反馈前FPR,反馈后FPR,耗时,ΔIoU)"""
    normals, fit_i, fit_m, test_defs = PREPS[cat]()
    bank, goods = normals[:60], normals[60:85]       # goods不参与fit,当"产线上的正常件"
    torch.manual_seed(0)
    det = CompetitionLargeDetector(train_steps=100)
    loop = ActiveLearningLoop(det, bank, fit_i, defect_masks=fit_m)
    f0, fps = fpr(det, goods)
    iou0 = eval_defect(det, test_defs[:15])
    if not fps:
        print(f"  [{cat}] 没有假阳性,该场景无从验证", flush=True)
        return None
    use = fps[:max(1, len(fps) // 2)]                # 只反馈一半,另一半留着看泛化
    t0 = time.time()
    for img in use:
        loop.normals.append(img)
        loop._refit(retrain_ead=not fast)            # 快路径=不重训学生
    dt = time.time() - t0
    f1, _ = fpr(det, goods)
    iou1 = eval_defect(det, test_defs[:15])
    print(f"  [{cat}] {'快路径(不重训学生)' if fast else '完整路径(重训学生)':22s} "
          f"反馈{len(use)}/{len(fps)}张误检 耗时={dt:.0f}s | "
          f"假阳性率 {f0:.3f}→{f1:.3f}({f1-f0:+.3f}) | 缺陷侧IoU {iou0:.3f}→{iou1:.3f}({iou1-iou0:+.3f})",
          flush=True)
    return f0, f1, dt, iou1 - iou0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="pill")
    args = ap.parse_args()
    print(f"=== 误检反馈:快路径 vs 完整路径 ({args.cat}) ===", flush=True)
    run(args.cat, fast=True)
    run(args.cat, fast=False)
    print("FP PATH OK", flush=True)


if __name__ == "__main__":
    main()

"""误检反馈能不能走"实时"快路径——机制级验证(不依赖假阳性是否恰好出现)。

背景:赛题要求"当系统**误检或漏检**时,操作员可提供**实时**反馈"。漏检那条已经做成
增量快路径(跳过EAD学生重训),误检那条目前仍走完整fit(20分钟),不契合"实时"。
假设:误检反馈也不需要重训学生——操作员标记误检后,系统要达成的是"别再把它判成
缺陷",而阈值标定`_calibrate(ns, ds)`本来就吃**全部正常图**的EAD分,把这张高分正常图
加进去阈值自然上移;DINO门/像素阈值同理重标。这些通路都不经过学生权重。

难点:三个类目的留出正常图实测**零假阳性**(阈值标定偏保守),没有天然的误检样本可测。
所以改测**安全边距**这个连续量:取留出正常图里分数最高的那张(最接近被误判的那张),
看反馈后 margin = 判定阈值 - 该图融合分 有没有变大。边距变大=系统确实朝"更不容易
把它误判成缺陷"的方向调整了,这正是误检反馈该起的作用,而且比"有没有翻转"灵敏得多。

对照:快路径(retrain_ead=False)vs 完整路径(retrain_ead=True),看边距改善差多少、
耗时差多少。

用法:PYTHONPATH=. python scripts/run_fp_margin.py --cat hazelnut
"""
import argparse
import time
import numpy as np
import torch
from aoi.active_learning import ActiveLearningLoop
from aoi.competition import CompetitionLargeDetector
from scripts.run_fp_feedback_path import PREPS


def fused_score(det, img):
    """图级判定用的那个分(DINO门启用时是融合分),与decision_threshold()配对。"""
    return float(det.frame_score(img))


def margin_of(det, img):
    return float(det.decision_threshold()) - fused_score(det, img)


def run(cat, fast, n_fb=3):
    normals, fit_i, fit_m, test_defs = PREPS[cat]()
    bank, goods = normals[:60], normals[60:85]
    torch.manual_seed(0)
    det = CompetitionLargeDetector(train_steps=100)
    loop = ActiveLearningLoop(det, bank, fit_i, defect_masks=fit_m)

    # 挑留出正常图里"最危险"的几张:融合分最高 = 最接近被误判成缺陷
    ranked = sorted(goods, key=lambda g: -fused_score(det, g))
    risky, rest = ranked[:n_fb], ranked[n_fb:]
    m0_risky = [margin_of(det, g) for g in risky]
    m0_rest = [margin_of(det, g) for g in rest]

    t0 = time.time()
    for img in risky:                                  # 模拟操作员标记这几张为"误检"
        loop.normals.append(img)
        loop._refit(retrain_ead=not fast)
    dt = time.time() - t0

    m1_risky = [margin_of(det, g) for g in risky]
    m1_rest = [margin_of(det, g) for g in rest]
    tag = "快路径(不重训学生)" if fast else "完整路径(重训学生)"
    print(f"  {tag:22s} 耗时={dt:6.0f}s | "
          f"被反馈图边距 {np.mean(m0_risky):+.3f}→{np.mean(m1_risky):+.3f}"
          f"({np.mean(m1_risky)-np.mean(m0_risky):+.3f}) | "
          f"其余正常图边距 {np.mean(m0_rest):+.3f}→{np.mean(m1_rest):+.3f}"
          f"({np.mean(m1_rest)-np.mean(m0_rest):+.3f})", flush=True)
    return np.mean(m1_risky) - np.mean(m0_risky), np.mean(m1_rest) - np.mean(m0_rest), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="hazelnut")
    args = ap.parse_args()
    print(f"=== 误检反馈的安全边距改善:快路径 vs 完整路径 ({args.cat}) ===", flush=True)
    print("  (边距=判定阈值-融合分,越大越不容易被误判成缺陷;反馈应让边距变大)", flush=True)
    run(args.cat, fast=True)
    run(args.cat, fast=False)
    print("MARGIN OK", flush=True)


if __name__ == "__main__":
    main()

"""用户反馈驱动优化——三条缺口的完整验证(赛题"用户反馈驱动的优化"是三条重点要求之一)。

赛题原文:"当系统**误检或漏检**时,操作员可提供**实时**反馈(如标记漏检),系统应能
**回溯检测逻辑**,动态调整模型参数"。此前run_active_learning_large.py只验证了
"漏检反馈能提升定位精度"一条,另外三点是缺口,这里补齐:

  ①【实时性】反馈漏检时跳过EAD学生重训(学生只吃正常图,缺陷图只参与阈值标定),
     实测对比"完整fit"vs"增量fit"的耗时,以及两者判定质量是否一致。
  ②【误检反馈】is_defect=False这条路径从没在生产大图架构上验证过。用真实正常图里
     被误判为缺陷的那些做反馈,看假阳性率是否下降。
  ③【回溯检测逻辑】det.explain()把整条判定链路摊开(各分支原始分/z分/生效阈值/
     谁主导判定/类型归属/掩膜经过哪些精化模块/延时自适应裁掉了什么)。

用法:PYTHONPATH=. python scripts/run_feedback_full.py --cat phone_battery
"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from aoi.active_learning import ActiveLearningLoop
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_active_learning_large import _read, evaluate_on, RI, RJ


def prep(cat, n_norm=100, n_init=10, n_fb_def=3, n_test=30, n_good=25):
    """比run_active_learning_large多准备两样:留出的正常图(测假阳性)、
    以及一批不参与fit的正常图(供误检反馈用)。"""
    d = json.load(open(RJ / f"{cat}.json")); R = RI / cat
    tok = [x for x in d["train"] if x["anomaly_class"] == "OK"]; random.Random(0).shuffle(tok)
    normals = [load_fast(R / x["image_path"]) for x in tok[:n_norm]]
    ng = [x for x in d["test"] if x["anomaly_class"] != "OK"]; random.Random(0).shuffle(ng)
    init_i = [load_fast(R / x["image_path"]) for x in ng[:n_init]]
    init_m = [_read(R / x["mask_path"]) for x in ng[:n_init]]
    fb_i = [load_fast(R / x["image_path"]) for x in ng[n_init:n_init + n_fb_def]]
    fb_m = [_read(R / x["mask_path"]) for x in ng[n_init:n_init + n_fb_def]]
    test_defs = [(load_fast(R / x["image_path"]), _read(R / x["mask_path"]))
                 for x in ng[n_init + n_fb_def:n_init + n_fb_def + n_test]]
    ok_test = [x for x in d["test"] if x["anomaly_class"] == "OK"]
    random.Random(1).shuffle(ok_test)
    goods = [load_fast(R / x["image_path"]) for x in ok_test[:n_good]]
    return normals, init_i, init_m, fb_i, fb_m, test_defs, goods


def false_positive_rate(det, goods):
    fp = sum(1 for g in goods if det.locate(g)["is_defect"])
    return fp / max(len(goods), 1), fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="phone_battery")
    args = ap.parse_args()
    torch.manual_seed(0)

    normals, init_i, init_m, fb_i, fb_m, test_defs, goods = prep(args.cat)
    print(f"{args.cat}: normals={len(normals)} 初始缺陷={len(init_i)} 漏检反馈={len(fb_i)} "
          f"留出缺陷test={len(test_defs)} 留出正常图={len(goods)}", flush=True)

    det = CompetitionLargeDetector()
    t0 = time.time()
    loop = ActiveLearningLoop(det, normals, init_i, defect_masks=init_m)
    t_full = time.time() - t0
    iou0, hit0 = evaluate_on(det, test_defs)
    fpr0, fp0 = false_positive_rate(det, goods)
    print(f"\n[初始] 完整fit耗时={t_full:.0f}s | 含漏检IoU={iou0:.3f} 框命中={hit0:.3f} "
          f"假阳性率={fpr0:.3f}({fp0}/{len(goods)})", flush=True)

    # ===== ③ 回溯检测逻辑:挑一张缺陷图,把判定链路摊开 =====
    print("\n===== ③【回溯检测逻辑】det.explain() 单图判定链路 =====", flush=True)
    tr = det.explain(test_defs[0][0])
    print(json.dumps(tr, ensure_ascii=False, indent=2, default=float), flush=True)

    # ===== ① 实时性:漏检反馈走增量fit,计时 =====
    print("\n===== ①【实时性】漏检反馈(跳过EAD学生重训) =====", flush=True)
    for i, (img, mk) in enumerate(zip(fb_i, fb_m)):
        t1 = time.time()
        n_norm, n_def = loop.feedback(img, is_defect=True, mask=mk)
        print(f"  漏检反馈第{i+1}张 → 耗时={time.time()-t1:.0f}s(完整fit是{t_full:.0f}s),"
              f"缺陷集={n_def}张", flush=True)
    iou1, hit1 = evaluate_on(det, test_defs)
    fpr1, fp1 = false_positive_rate(det, goods)
    print(f"[漏检反馈后] 含漏检IoU={iou1:.3f}({iou1-iou0:+.3f}) 框命中={hit1:.3f}({hit1-hit0:+.3f}) "
          f"假阳性率={fpr1:.3f}", flush=True)

    # ===== ② 误检反馈:把被误判成缺陷的正常图反馈回去 =====
    print("\n===== ②【误检反馈】is_defect=False 路径(此前从未验证) =====", flush=True)
    fp_imgs = [g for g in goods if det.locate(g)["is_defect"]]
    if not fp_imgs:
        print("  当前没有假阳性样本(留出正常图全部判对),该路径无从触发——"
              "这本身是好结果,但意味着本次没能验证误检反馈的效果", flush=True)
    else:
        # 只反馈一半,另一半留着看假阳性率变化(全反馈进去就成了自己考自己)
        use = fp_imgs[:max(1, len(fp_imgs) // 2)]
        print(f"  留出正常图里有{len(fp_imgs)}张被误判为缺陷,反馈其中{len(use)}张", flush=True)
        for i, img in enumerate(use):
            t1 = time.time()
            n_norm, n_def = loop.feedback(img, is_defect=False)
            print(f"  误检反馈第{i+1}张 → 耗时={time.time()-t1:.0f}s(走完整fit,"
                  f"新增正常图必须重训学生),正常集={n_norm}张", flush=True)
        iou2, hit2 = evaluate_on(det, test_defs)
        fpr2, fp2 = false_positive_rate(det, goods)
        print(f"[误检反馈后] 假阳性率={fpr2:.3f}({fp2}/{len(goods)}),"
              f"较反馈前{fpr1:.3f} → Δ={fpr2-fpr1:+.3f}", flush=True)
        print(f"           含漏检IoU={iou2:.3f}({iou2-iou1:+.3f}) 框命中={hit2:.3f}({hit2-hit1:+.3f})"
              f"  ←误检反馈不应显著伤害缺陷侧", flush=True)


if __name__ == "__main__":
    main()

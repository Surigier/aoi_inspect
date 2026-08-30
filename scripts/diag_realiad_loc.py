"""定位崩塌定位器:phone_battery 含漏检IoU 从0.399(8-20基线)掉到0.033,
图级acc几乎不变(0.925→0.912)——判决没坏,掩膜坏了。

代码diff里定位链路几乎没动,所以嫌疑集中在**fit期的自动门控**(它们是数据驱动的,
可能翻向):模板差分开关(_select_feat_mode)、SAM受控精化(OOF门控)、像素阈值标定、
分割头是否训成功。本脚本把这些决策全部打出来,再逐图对比GT面积与预测面积,
把"掩膜太大/太小/位置不对"三种失效分开。

用法:PYTHONPATH=. python scripts/diag_realiad_loc.py [类目=phone_battery]
"""
import sys
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_realiad, gt_boxes, box_hit


def main(cat="phone_battery"):
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs, goods = prep_realiad(cat)
    print(f"{cat}: 正常{len(normals)} fit缺陷{len(fit_i)} 测试缺陷{len(test_defs)} 测试正常{len(goods)}", flush=True)
    gt_area = np.mean([m.mean() for m in fit_m])
    print(f"fit掩膜平均占图比例 {gt_area:.4%}", flush=True)

    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)

    print("\n=== fit期定位决策 ===", flush=True)
    print(f"  分割头训成功       : {det.seg_head.head is not None}", flush=True)
    print(f"  分割头阈值 thr     : {det.seg_head.thr}", flush=True)
    print(f"  像素阈值 pix_thr   : {det.pix_thr}", flush=True)
    print(f"  模板差分模式       : {getattr(det.seg_head, 'feat_mode', '(无该属性)')}", flush=True)
    print(f"  SAM 启用           : {det.sam is not None}", flush=True)
    if det.sam is not None:
        for a in ("enabled", "gate", "min_iou", "accept_frac", "_gate"):
            if hasattr(det.sam, a):
                print(f"    sam.{a} = {getattr(det.sam, a)}", flush=True)
    print(f"  boundary_refiner   : {det.boundary_refiner is not None}", flush=True)
    print(f"  crop_cascade       : {det.crop_cascade is not None}", flush=True)
    print(f"  comp_graph         : {det.comp_graph is not None}", flush=True)
    print(f"  图级阈值 threshold : {det.threshold}", flush=True)
    print(f"  DINO门             : {det._dino is not None}", flush=True)

    print("\n=== 逐图(前12张)GT面积 vs 预测面积 ===", flush=True)
    ious, gtf, pdf = [], [], []
    for i, (img, gt) in enumerate(test_defs[:12]):
        o = det.locate(img)
        if o.get("mask") is None:
            print(f"  #{i:2d} 无掩膜(is_defect={o['is_defect']})", flush=True); continue
        mk = o["mask"].astype(bool); g = gt.astype(bool)
        inter = int((mk & g).sum()); union = int((mk | g).sum())
        iou = inter / max(union, 1)
        ious.append(iou); gtf.append(g.mean()); pdf.append(mk.mean())
        print(f"  #{i:2d} GT占比{g.mean():.4%}  预测占比{mk.mean():.4%}  "
              f"预测/GT={mk.sum()/max(g.sum(),1):6.1f}×  交集{inter:5d}  IoU={iou:.3f}  框{len(o['boxes'])}个",
              flush=True)
    # 全量40张,用与 run_scorecard.evaluate 完全相同的算法复算,直接和它的0.033对质
    ig, ip, hs, ndet = [], [], [], 0
    for img, gt in test_defs:
        o = det.locate(img)
        if o.get("mask") is not None:
            pred = o["mask"].astype(bool)
            TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum())
            FN = int((~pred & (gt == 1)).sum())
            iou = TP / max(TP + FP + FN, 1)
        else:
            iou = 0.0
        ip.append(iou)
        if o["is_defect"]:
            ndet += 1; ig.append(iou)
            h = box_hit(o["boxes"], gt_boxes(gt)); hs.append(h if h is not None else 0.0)
        else:
            ig.append(0.0); hs.append(0.0)
    print(f"\n=== 全量{len(test_defs)}张,evaluate同款算法 ===", flush=True)
    print(f"  检出 {ndet}/{len(test_defs)}  含漏检IoU={np.mean(ig):.3f}  纯定位IoU={np.mean(ip):.3f}  "
          f"框命中={np.mean(hs):.3f}", flush=True)
    print(f"  对质:run_scorecard 报的是 含漏检0.033/纯定位0.033/框命中0.013", flush=True)
    print(f"  SAM门控本次 = {getattr(det.sam, 'gate', None) if det.sam else 'SAM未启用'}", flush=True)

    # 同一进程里再走一遍 run_scorecard.evaluate,和上面的手算并排对质。
    # 若两者不同 → 问题在 evaluate 内部;若相同 → 是运行间不确定性(fit期门控在跳)。
    from scripts.run_scorecard import evaluate
    print("\n=== 同进程内直接调 run_scorecard.evaluate(它自己重新fit一个检测器)===", flush=True)
    ev = evaluate(cat + "(evaluate)", normals, fit_i, fit_m, test_defs, goods)
    print(f"  evaluate 返回: acc={ev[0]:.3f} 含漏检IoU={ev[1]:.3f} 纯定位={ev[2]:.3f} 框命中={ev[3]:.3f}",
          flush=True)
    print(f"  上面手算的是: 含漏检IoU={np.mean(ig):.3f} 纯定位={np.mean(ip):.3f} 框命中={np.mean(hs):.3f}",
          flush=True)

    if ious:
        print(f"\n均值(前12张): IoU={np.mean(ious):.3f}  GT占比={np.mean(gtf):.4%}  预测占比={np.mean(pdf):.4%}  "
              f"预测/GT面积比={np.mean(pdf)/max(np.mean(gtf),1e-9):.1f}×", flush=True)
        print("判读:面积比≫1 → 掩膜过大(阈值太松/SAM扩张);≪1 → 掩膜过小或空;"
              "面积比≈1但IoU低 → 位置不对(分割头训坏)", flush=True)
    print("DIAG OK", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "phone_battery")

"""缺陷子类型细分成绩单:leon要看"检测头本身(不是VLM类型头)对不同缺陷类型是不是
一视同仁,还是有的类型系统性漏检/定位不准"。按产品类目内部真实的缺陷子文件夹名
(比如grid的bent/broken/glue/metal_contamination/thread)分别统计检出率/含漏检
IoU/框命中,而不是只看类目整体均值——run_scorecard_extra.py那份成绩单只报了类目
均值,盖住了子类型间的差异。

复用run_scorecard_extra.py的MVTEC_EXTRA(8类),只改测试循环:test_defs多带一个
子类型字段,按(类目,子类型)分组统计。fit/口径其余不变。

同时按leon要求:再输出一张"数据集原生缺陷子类型 x 系统预测缺陷类型(VLM/规则)"
的原始对照表(同eval_phone_box.py的思路,不做映射评判,系统输出是什么就是什么)。

用法:PYTHONPATH=. python scripts/run_scorecard_by_defect_type.py
"""
import collections
import glob
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import GT, HW, _read, box_hit, gt_boxes, img_iou
from scripts.run_scorecard_extra import MVTEC_EXTRA


def prep_mvtec_typed(cat, folders):
    """同run_scorecard_extra.prep_mvtec,test_defs多带一个子类型字段。"""
    root = Path(f"data/mvtec/{cat}")
    normals = [load_fast(p) for p in sorted(glob.glob(str(root / "train/good/*.png")))[:100]]
    goods = [(load_fast(p), None) for p in sorted(glob.glob(str(root / "test/good/*.png")))[:40]]
    df = []
    for fo in folders:
        df += [(p, fo) for p in sorted(glob.glob(str(root / "test" / fo / "*.png")))]
    random.Random(0).shuffle(df); k = max(5, len(df) // 3)
    fit_i = [load_fast(p) for p, _ in df[:k]]
    fit_m = [_read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW) for p, fo in df[:k]]
    test_defs = [(load_fast(p), _read(GT / cat / "ground_truth" / fo / (Path(p).stem + "_mask.png"), HW), fo)
                 for p, fo in df[k:]]
    return normals, fit_i, fit_m, test_defs, goods


def evaluate_typed(name, normals, fit_i, fit_m, test_defs, test_goods):
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    rows = defaultdict(lambda: {"n": 0, "hit_ok": 0, "ious": [], "hits": []})
    type_table = defaultdict(collections.Counter)   # 数据集原生子类型 -> 系统预测缺陷类型计数
    for img, gt, subtype in test_defs:
        o = det.locate(img)
        r = rows[subtype]
        r["n"] += 1
        if o.get("mask") is not None:
            pred = o["mask"].astype(bool)
            TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
            iou = TP / max(TP + FP + FN, 1)
        else:
            iou = img_iou(det.segment(img), gt, det.pix_thr)
        if o["is_defect"]:
            r["hit_ok"] += 1
            r["ious"].append(iou)
            h = box_hit(o["boxes"], gt_boxes(gt))
            r["hits"].append(h if h is not None else 0.0)
            type_table[subtype][o["defect_type"]] += 1
        else:
            r["ious"].append(0.0)          # 漏检=定位0分,同run_scorecard口径
            r["hits"].append(0.0)
            type_table[subtype]["normal(漏检)"] += 1
    tn = sum(1 for img, _ in test_goods if not det.locate(img)["is_defect"])
    print(f"\n=== {name} ===  正常图检出率(TN)={tn}/{len(test_goods)}={tn/max(len(test_goods),1):.3f}", flush=True)
    for subtype, r in sorted(rows.items()):
        print(f"  {subtype:22s} n={r['n']:3d} 检出率={r['hit_ok']/r['n']:.3f} "
              f"含漏检IoU={np.mean(r['ious']):.3f} 框命中@0.5={np.mean(r['hits']):.3f}", flush=True)
    print("  数据集原生子类型 x 系统预测缺陷类型(VLM/规则,原始对照表不做映射评判):", flush=True)
    for subtype, c in sorted(type_table.items()):
        print(f"    {subtype}(共{sum(c.values())}张): " + ", ".join(f"{k}={v}" for k, v in c.most_common()), flush=True)
    return rows, type_table


def main():
    torch.manual_seed(0)
    print("=== 缺陷子类型细分成绩单(产品类目内部,检测头本身,非VLM) ===")
    for cat, folders in MVTEC_EXTRA:
        evaluate_typed(cat, *prep_mvtec_typed(cat, folders))


if __name__ == "__main__":
    main()

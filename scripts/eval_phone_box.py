"""手机屏box-IoU对照:正常图硬约束只有20张(MSD官方good.zip量),15fit/5test维持不变;
缺陷图按leon要求从30张fit压到10张,把省下的20张挪去测试,尽量多测。
手机屏没有像素级GT掩膜(GT本来就是YOLO框),定位口径直接用框对框IoU,不转掩膜再算。

用法:PYTHONPATH=. python scripts/eval_phone_box.py
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import make_exam_data as m
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import box_hit, box_iou

RNG = random.Random(1)
msd = sorted(m.MSD_GOOD.glob("*.png")); RNG.shuffle(msd)
pb = m.phone_best_defects(10 + 10000)   # 请求量超上限,函数会原样返回全部可用的

normals = [load_fast(p) for p in msd[:15]]
fit_items = pb[:10]
defects = [load_fast(f) for f, _, _ in fit_items]
masks = [mk for _, mk, _ in fit_items]

print(f"fit: 正常{len(normals)} 缺陷{len(defects)}(20张官方正常图硬约束,不可再多)", flush=True)
det = CompetitionLargeDetector(compile_infer=True)
det.fit_fewshot(normals, defects, defect_masks=masks)
print(f"fit完成,阈值={det.decision_threshold():.4f}", flush=True)

test_items = pb[10:]
ious, hits, recall_n = [], [], 0
import collections
type_table = collections.defaultdict(collections.Counter)   # 官方原始类型 -> 系统预测类别计数
for f, mk, gt_type in test_items:
    img = load_fast(f)
    o = det.locate(img)
    ys, xs = np.where(mk > 0)
    gtb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    if o["is_defect"]:
        recall_n += 1
        mkpred = o.get("mask")
        if mkpred is not None and o.get("boxes"):
            mh, mw = mkpred.shape[:2]
            H0, W0 = mk.shape
            sx, sy = W0 / max(mw, 1), H0 / max(mh, 1)
            pred_boxes = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in o["boxes"]]
            iou = max((box_iou(p, gtb) for p in pred_boxes), default=0.0)
            h = box_hit(pred_boxes, [gtb])
        else:
            iou, h = 0.0, 0.0
        type_table[gt_type][o["defect_type"]] += 1
    else:
        iou, h = 0.0, 0.0
        type_table[gt_type]["normal(漏检)"] += 1
    ious.append(iou); hits.append(h if h is not None else 0.0)

n = len(test_items)
print(f"\n测试{n}张(全是缺陷图,手机屏无更多正常图可测)", flush=True)
print(f"召回={recall_n/n:.3f} 含漏检IoU(框)={np.mean(ious):.3f} 框命中@0.5={np.mean(hits):.3f}", flush=True)
print("\n官方原始类型(oil/scratch/stain) x 系统预测类别,原始对照表不做映射评判:", flush=True)
for gt_type, c in sorted(type_table.items()):
    print(f"  {gt_type}(共{sum(c.values())}张): " + ", ".join(f"{k}={v}" for k, v in c.most_common()), flush=True)

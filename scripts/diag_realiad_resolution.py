"""诊断:pcb/battery(本项目历史最弱两类)垫底,是否是Real-IAD原生256²分辨率的
benchmark artifact,而非架构真实短板?今天成绩单里hazelnut/cable/pill原生800-1024²
(含漏检0.44-0.81),pcb/battery原生仅256²(含漏检0.25-0.37)——垫底两类恰好是原生
分辨率最低的两类,值得验证而非继续加架构补丁。

方法:同一批Real-IAD 256²原生图,分别按原生256²和上采样640²喂管线(其余配置完全
相同),对比纯定位IoU/框命中。若上采样后明显提升→瓶颈是128²特征格下采样比例,
赛题真实2500²(真实高分辨率非插值)应显著好于此benchmark;若不提升→缺陷本身难,
与分辨率无关。
用法:PYTHONPATH=. python scripts/diag_realiad_resolution.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_realiad, gt_boxes, box_hit


def _upsize(imgs, size):
    out = []
    for img in imgs:
        x = img.unsqueeze(0) if img.dim() == 3 else img
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        out.append(x[0])
    return out


def eval_iou_hit(det, test_defs):
    ious, hits = [], []
    for img, gt in test_defs:
        o = det.locate(img)
        pix = (o["mask"].astype(bool) if o.get("mask") is not None else (det.segment(img) >= det.pix_thr))
        TP = int((pix & (gt == 1)).sum()); FP = int((pix & (gt == 0)).sum()); FN = int((~pix & (gt == 1)).sum())
        ious.append(TP / max(TP + FP + FN, 1))
        if o["is_defect"]:
            h = box_hit(o["boxes"], gt_boxes(gt))
            hits.append(h if h is not None else 0.0)
        else:
            hits.append(0.0)
    return float(np.mean(ious)), float(np.mean(hits))


def main():
    torch.manual_seed(0)
    for cat in ["pcb", "phone_battery"]:
        normals, fit_i, fit_m, test_defs, _goods = prep_realiad(cat)
        print(f"{cat}: 原生尺寸样例={tuple(normals[0].shape[-2:])}", flush=True)
        for size, tag in [(None, "原生256"), (640, "上采样640")]:
            if size is None:
                n2, fi2, td2 = normals, fit_i, test_defs
            else:
                n2 = _upsize(normals, size)
                fi2 = _upsize(fit_i, size)
                td2 = [(_upsize([img], size)[0], gt) for img, gt in test_defs]
            det = CompetitionLargeDetector(train_steps=3000, ead_students=1)
            det.fit_fewshot(n2, fi2, defect_masks=fit_m)
            iou, hit = eval_iou_hit(det, td2)
            print(f"  {tag}: 纯定位IoU={iou:.3f} 框命中={hit:.3f}", flush=True)


if __name__ == "__main__":
    main()

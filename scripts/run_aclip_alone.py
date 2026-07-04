"""ACLIP单独的严格指标(探针AUROC虚高警惕):fit标F1阈值→逐图IoU+框命中,
与监督头同口径对比。若单独赢→按类选路(而非加法融合)。
用法:python scripts/run_aclip_alone.py
"""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/srj/yolo/anomalyclip")
from scripts.run_aclip_fusion import (ACLIPMap, prep_realiad, prep_mvtec, HW, ACLIP,
                                      f1_thr, gtb, hit_rate, iou)


def main():
    torch.manual_seed(0)
    print("=== ACLIP单独严格指标(fit F1阈值)===")
    jobs = [
        ("pcb", lambda: prep_realiad("pcb"), f"{ACLIP}/checkpoints/9_12_4_multiscale/epoch_15.pth"),
        ("battery", lambda: prep_realiad("phone_battery"), f"{ACLIP}/checkpoints/9_12_4_multiscale/epoch_15.pth"),
        ("pill", lambda: prep_mvtec("pill", ["color"]), f"{ACLIP}/checkpoints/9_12_4_multiscale_visa/epoch_15.pth"),
        ("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"]), f"{ACLIP}/checkpoints/9_12_4_multiscale_visa/epoch_15.pth"),
    ]
    cur, ac = None, None
    for name, prep, ck in jobs:
        if ck != cur:
            ac = ACLIPMap(ck); cur = ck
        normals, fit, tests = prep()
        fitS = [ac.map(im) for im, _ in fit]
        fitL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in fit]
        thr = f1_thr(fitS, fitL)
        tstS = [ac.map(im) for im, _ in tests]
        tstL = [np.array(Image.fromarray(mk).resize(HW[::-1], Image.NEAREST)) for _, mk in tests]
        tst_gt = [gtb(l) for l in tstL]
        ious = [iou(s >= thr, l) for s, l in zip(tstS, tstL)]
        bh = hit_rate([(gtb((s >= thr).astype(np.uint8), 3), g) for s, g in zip(tstS, tst_gt)])
        print(f"{name:10s} ACLIP单独: IoU={np.mean(ious):.3f}  框命中={bh:.3f}", flush=True)


if __name__ == "__main__":
    main()

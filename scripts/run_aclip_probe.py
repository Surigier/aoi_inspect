"""AnomalyCLIP集成正确性探针:它的零样本图独立的pixel-AUROC(对GT掩膜)。
~0.5=集成有bug;0.7-0.9(论文水平)=集成正确,α=0是"不叠加"的诚实结论。
用法:python scripts/run_aclip_probe.py
"""
import sys
import glob
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/srj/yolo/anomalyclip")
from scripts.run_aclip_fusion import ACLIPMap, prep_realiad, prep_mvtec, HW
from eval.protocol import image_auroc

ACLIP = "/home/srj/yolo/anomalyclip"


def main():
    torch.manual_seed(0)
    print("=== AnomalyCLIP 零样本图独立 pixel-AUROC(集成正确性验证)===")
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
        S, L = [], []
        for img, mk in tests[:30]:
            S.append(ac.map(img).ravel()); L.append(mk.ravel())
        for img in normals[40:55]:
            S.append(ac.map(img).ravel()); L.append(np.zeros(HW, np.uint8).ravel())
        au = image_auroc(np.concatenate(S), np.concatenate(L))
        print(f"{name:10s} ACLIP独立 pixel-AUROC={au:.3f}  {'✅集成OK(不叠加是事实)' if au > 0.65 else '⚠️疑似集成bug或模型盲'}", flush=True)


if __name__ == "__main__":
    main()

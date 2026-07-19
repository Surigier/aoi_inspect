"""pcb/battery新头崩塌(0.251→0.028 / 0.374→0.122)鉴别诊断:头坏了还是阈值选坏了?
同一个训好的新头,分别用thr_iou(默认)/thr_f1(旧口径)/thr_boxhit评IoU;外加旧头对照。
若new@thr_f1≈old→罪在_oof_calibrate_thr对微小缺陷的阈值选择;仍崩→罪在soft target/loss。
用法:PYTHONPATH=. python scripts/diag_tiny_collapse.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from aoi.backbone import Backbone
from aoi.seg_head import SupervisedSegHead as NewHead
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as OldHead
from scripts.run_scorecard import prep_realiad

SEG_IN = 512
HW = (256, 256)


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(layers=(1, 2), device=dev)

    @torch.no_grad()
    def extractor(img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(dev)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        return bb.extract(x)[0]

    for cat in ["pcb", "phone_battery"]:
        normals, fit_i, fit_m, test_defs, _ = prep_realiad(cat)
        new = NewHead(device=dev, extractor=extractor)
        old = OldHead(device=dev, extractor=extractor)
        new.fit(None, fit_i, fit_m, normals[:30])
        old.fit(None, fit_i, fit_m, normals[:30])

        def ev(head, thr):
            if thr is None:
                return float("nan")
            ious = []
            for img, gt in test_defs:
                amap = head.map(None, img, HW)
                pred = (amap >= thr)
                TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
                ious.append(TP / max(TP + FP + FN, 1))
            return float(np.mean(ious))

        print(f"{cat}: 阈值 thr_iou={new.thr_iou} thr_f1={new.thr_f1} thr_boxhit={new.thr_boxhit}", flush=True)
        print(f"  new@thr_iou={ev(new, new.thr_iou):.3f}  new@thr_f1={ev(new, new.thr_f1):.3f}  "
              f"new@thr_boxhit={ev(new, new.thr_boxhit):.3f}  |  old@自身thr={ev(old, old.thr):.3f}", flush=True)
        # 附:GT掩膜在128²特征格上的soft target峰值分布(target稀释假说的直接证据)
        from aoi.seg_head import _mask_to_soft
        peaks = [float(_mask_to_soft(m, 128, 128).max()) for m in fit_m]
        print(f"  soft-target峰值: 中位={np.median(peaks):.3f} 最小={min(peaks):.3f} "
              f"<0.1占比={np.mean([p < 0.1 for p in peaks]):.2f}", flush=True)


if __name__ == "__main__":
    main()

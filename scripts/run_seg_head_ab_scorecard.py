"""紧急归因:全量成绩单IoU均值0.484(历史)→0.413(新栈)的回归来源定位。
新旧seg_head的A/B此前只在3个AD2类做过(结论净平)——成绩单5类(hazelnut/cable/pill/
pcb/battery)从未直接对比。此脚本在这5类上隔离对比(同extractor,只换头),
若旧头显著赢→回归来自seg_head换代,需按类处理;若打平→回归来自其他改动(阈值口径/
SAM离场/…),继续排查。
用法:PYTHONPATH=. python scripts/run_seg_head_ab_scorecard.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from aoi.backbone import Backbone
from aoi.seg_head import SupervisedSegHead as NewHead
from aoi._seg_head_old_ae5fbbb import SupervisedSegHead as OldHead
from scripts.run_scorecard import prep_mvtec, prep_realiad

SEG_IN = 512
HW = (256, 256)

JOBS = [
    ("hazelnut", lambda: prep_mvtec("hazelnut", ["crack", "cut", "hole"])),
    ("cable", lambda: prep_mvtec("cable", ["missing_cable", "missing_wire"])),
    ("pill", lambda: prep_mvtec("pill", ["color"])),
    ("pcb", lambda: prep_realiad("pcb")),
    ("phone_battery", lambda: prep_realiad("phone_battery")),
]


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = Backbone(layers=(1, 2), device=dev)

    @torch.no_grad()
    def extractor(img):
        x = (img.unsqueeze(0) if img.dim() == 3 else img).to(dev)
        x = F.interpolate(x, size=(SEG_IN, SEG_IN), mode="bilinear", align_corners=False)
        return bb.extract(x)[0]

    results = {}
    for name, prep in JOBS:
        normals, fit_i, fit_m, test_defs, _goods = prep()
        old = OldHead(device=dev, extractor=extractor)
        new = NewHead(device=dev, extractor=extractor)
        ok_o = old.fit(None, fit_i, fit_m, normals[:30])
        ok_n = new.fit(None, fit_i, fit_m, normals[:30])

        def ev(head):
            thr = getattr(head, "thr", None)
            ious = []
            for img, gt in test_defs:
                amap = head.map(None, img, HW)
                if amap is None or thr is None:
                    ious.append(0.0); continue
                pred = (amap >= thr)
                TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
                ious.append(TP / max(TP + FP + FN, 1))
            return float(np.mean(ious))

        o, n = ev(old), ev(new)
        results[name] = (o, n)
        print(f"{name:14s} old(双头pooledF1)={o:.3f}  new(bagging+OOF-IoU)={n:.3f}  Δ(new-old)={n-o:+.3f}", flush=True)
    om = np.mean([v[0] for v in results.values()])
    nm = np.mean([v[1] for v in results.values()])
    print(f"\n=== 均值 === old={om:.3f}  new={nm:.3f}  Δ={nm-om:+.3f}", flush=True)


if __name__ == "__main__":
    main()

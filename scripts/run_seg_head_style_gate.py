"""验证新加的_select_seg_head_style(competition.py):按类留出集选新式(4头bagging+OOF-IoU
阈值)vs旧式(双头pooled-F1,ae5fbbb存档)——run_seg_head_ab.py暴露均值净平(0.598≈0.598)
掩盖了逐类反号方差(sheet_metal新+0.111/walnuts旧+0.020/fruit_jelly旧+0.092),按类选优
留出验证应逼近max(old,new)≈0.635。
只需构造CompetitionLargeDetector(不需要真EAD训练——_eff()只需对象图存在,_select_feat_mode/
_select_seg_head_style的extractor走WRN不碰EAD/self.branches[0].det.det只是属性链)。
用法:PYTHONPATH=. python scripts/run_seg_head_style_gate.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_seg_head_ab import prep_ad2

CATS = ["sheet_metal", "walnuts", "fruit_jelly"]


def main():
    torch.manual_seed(0)
    results = {}
    for cat in CATS:
        normals, fit_i, fit_m, test_defs = prep_ad2(cat)
        det = CompetitionLargeDetector()   # 不调fit_fewshot,只构造对象图(EAD不训练,省~6分钟/类)
        det._select_feat_mode(fit_i, fit_m, normals)
        det._select_seg_head_style(fit_i, fit_m, normals)
        det.seg_head.fit(det._eff(), fit_i, fit_m, normals[:30])
        style = getattr(det, "seg_head_style", None)
        thr = getattr(det.seg_head, "thr", None)
        ious = []
        for img, gt in test_defs:
            amap = det.seg_head.map(det._eff(), img, det.seg_eval_hw)
            if amap is None or thr is None:
                ious.append(0.0); continue
            pred = amap >= thr
            TP = int((pred & (gt == 1)).sum()); FP = int((pred & (gt == 0)).sum()); FN = int((~pred & (gt == 1)).sum())
            ious.append(TP / max(TP + FP + FN, 1))
        iou = float(np.mean(ious))
        results[cat] = iou
        print(f"{cat:14s} 选中style={style}  test集IoU={iou:.3f}", flush=True)
    print(f"\n均值={np.mean(list(results.values())):.3f}  (对照:run_seg_head_ab.py old均值=0.598 new均值=0.598)", flush=True)


if __name__ == "__main__":
    main()

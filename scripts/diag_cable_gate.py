"""cable图级acc=0.236归因(定位0.805正常,纯检测门问题;上一轮同代码acc=0.873,
且cable@640历史就有"小样本台架artifact"检测脆弱前科)。
生产同配置fit,打:EAD阈值/DINO门状态/融合阈值/正常vs缺陷分数分布/两种门各自的acc。
用法:PYTHONPATH=. python scripts/diag_cable_gate.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from scripts.run_scorecard import prep_mvtec


def main():
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs, goods = prep_mvtec("cable", ["missing_cable", "missing_wire"])
    det = CompetitionLargeDetector()                       # 生产同配置(10000步,双学生)
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"EAD阈值={det.threshold}  DINO门={'开' if det._dino is not None else '关'}  "
          f"DINO阈值={getattr(det, '_dino_thr', None)}", flush=True)

    ead_d = [det.branches[0].score(img) for img, _ in test_defs]
    ead_g = [det.branches[0].score(img) for img, _ in goods]
    print(f"EAD分: 缺陷 中位={np.median(ead_d):.4f} p10={np.percentile(ead_d,10):.4f} | "
          f"正常 中位={np.median(ead_g):.4f} p90={np.percentile(ead_g,90):.4f}", flush=True)

    def acc_with(dino_on):
        dino_bak = det._dino
        if not dino_on:
            det._dino = None
        ok = 0; tot = 0
        for img, _ in test_defs:
            tot += 1; ok += int(det.predict(img)["is_defect"])
        for img, _ in goods:
            tot += 1; ok += int(not det.predict(img)["is_defect"])
        det._dino = dino_bak
        return ok / tot

    acc_dino = acc_with(True)
    acc_ead = acc_with(False)
    print(f"acc(生产=当前门)= {acc_dino:.3f}   acc(强制EAD-only)= {acc_ead:.3f}", flush=True)
    if det._dino is not None:
        fs_d = [det._dino_fuse(det.branches[0].score(img), det._dino.score(img)) for img, _ in test_defs[:20]]
        fs_g = [det._dino_fuse(det.branches[0].score(img), det._dino.score(img)) for img, _ in goods[:20]]
        print(f"融合分(前20): 缺陷 中位={np.median(fs_d):.4f} | 正常 中位={np.median(fs_g):.4f} | "
              f"阈值={det._dino_thr:.4f}", flush=True)


if __name__ == "__main__":
    main()

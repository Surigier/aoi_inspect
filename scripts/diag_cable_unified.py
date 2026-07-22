"""cable统一口径(load_fast)下反向异常:acc=0.691尚可,但含漏检0.048/框0.067
(纯定位0.811历史最高)——绝大多数真缺陷图被漏判成正常(与640口径下"正常图全报警"
方向相反)。诊断:load_fast保长宽比后cable图不再是640×640,EAD/DINO阈值标定在
新几何下可能站错位置;打印测试集is_defect命中率+分数分布确认。
用法:PYTHONPATH=. python scripts/diag_cable_unified.py
"""
import numpy as np
import torch
from aoi.competition import CompetitionLargeDetector
from aoi.imageio import load_fast
from scripts.run_scorecard import prep_mvtec


def main():
    torch.manual_seed(0)
    normals, fit_i, fit_m, test_defs, goods = prep_mvtec("cable", ["missing_cable", "missing_wire"])
    print("正常图尺寸样例:", [tuple(n.shape[-2:]) for n in normals[:5]], flush=True)
    print("缺陷图尺寸样例:", [tuple(d.shape[-2:]) for d, _ in test_defs[:5]], flush=True)
    det = CompetitionLargeDetector()
    det.fit_fewshot(normals, fit_i, defect_masks=fit_m)
    print(f"EAD阈值={det.threshold}  DINO门={'开' if det._dino is not None else '关'}  "
          f"DINO阈值={getattr(det, '_dino_thr', None)}", flush=True)

    n_hit = 0
    scores_def, scores_good = [], []
    for img, gt in test_defs:
        o = det.predict(img)
        scores_def.append(o["score"])
        n_hit += int(o["is_defect"])
    n_hit_good = 0
    for img, _ in goods:
        o = det.predict(img)
        scores_good.append(o["score"])
        n_hit_good += int(not o["is_defect"])
    print(f"test缺陷图 判缺陷 {n_hit}/{len(test_defs)}  |  test正常图 判正常 {n_hit_good}/{len(goods)}", flush=True)
    print(f"EAD分: 缺陷 中位={np.median(scores_def):.4f} p10={np.percentile(scores_def,10):.4f} | "
          f"正常 中位={np.median(scores_good):.4f} p90={np.percentile(scores_good,90):.4f}", flush=True)

    if det._dino is not None:
        fd = [det._dino_fuse(det.branches[0].score(img), det._dino.score(img)) for img, _ in test_defs]
        fg = [det._dino_fuse(det.branches[0].score(img), det._dino.score(img)) for img, _ in goods]
        print(f"融合分: 缺陷 中位={np.median(fd):.4f} p10={np.percentile(fd,10):.4f} | "
              f"正常 中位={np.median(fg):.4f} p90={np.percentile(fg,90):.4f} | 阈值={det._dino_thr:.4f}", flush=True)


if __name__ == "__main__":
    main()
